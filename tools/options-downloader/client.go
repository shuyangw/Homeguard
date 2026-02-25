package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/url"
	"sync/atomic"
	"time"
)

var (
	ErrRateLimited = errors.New("rate limited")
	ErrNoData      = errors.New("no data")
	ErrNotRunning  = errors.New("theta terminal not running")
)

// rateLimitUntil is a global timestamp (unix seconds) until which all
// workers should wait before making new requests.
var rateLimitUntil atomic.Int64

// ThetaClient is a simple HTTP client for the ThetaData REST API.
// Concurrency is controlled externally by the worker pool size -- no semaphore.
type ThetaClient struct {
	baseURL    string
	httpClient *http.Client
	timeoutSec int
}

// NewThetaClient creates a new client with the given timeout.
func NewThetaClient(timeoutSec int) *ThetaClient {
	return &ThetaClient{
		baseURL: thetaBaseURL,
		httpClient: &http.Client{
			Timeout: time.Duration(timeoutSec) * time.Second,
			Transport: &http.Transport{
				MaxIdleConns:          100,
				MaxIdleConnsPerHost:   100,
				IdleConnTimeout:       90 * time.Second,
			},
		},
		timeoutSec: timeoutSec,
	}
}

// request makes an HTTP GET with retry and rate-limit handling.
// Downloads the full response body and returns it as bytes.
// No semaphore -- concurrency is controlled by the caller (worker pool).
func (c *ThetaClient) request(endpoint string, params url.Values) ([]byte, error) {
	u := c.baseURL + endpoint + "?" + params.Encode()

	for attempt := 0; attempt < maxRetries; attempt++ {
		// Check global rate limit
		if until := rateLimitUntil.Load(); until > 0 {
			wait := until - time.Now().Unix()
			if wait > 0 {
				slog.Debug("Waiting for rate limit", "seconds", wait)
				time.Sleep(time.Duration(wait) * time.Second)
			}
		}

		// Create per-request context with deadline
		reqCtx, reqCancel := context.WithTimeout(context.Background(), time.Duration(c.timeoutSec)*time.Second)
		req, reqErr := http.NewRequestWithContext(reqCtx, "GET", u, nil)
		if reqErr != nil {
			reqCancel()
			return nil, fmt.Errorf("create request: %w", reqErr)
		}

		resp, err := c.httpClient.Do(req)
		if err != nil {
			reqCancel()
			if isConnectionError(err) {
				return nil, ErrNotRunning
			}
			if reqCtx.Err() != nil {
				slog.Warn("Request timeout", "endpoint", endpoint, "attempt", attempt+1)
			} else {
				slog.Warn("Request failed", "endpoint", endpoint, "error", err, "attempt", attempt+1)
			}
			if attempt < maxRetries-1 {
				time.Sleep(time.Duration(retryBackoff[attempt]) * time.Second)
				continue
			}
			return nil, fmt.Errorf("request failed after %d attempts: %w", maxRetries, err)
		}

		body, readErr := io.ReadAll(resp.Body)
		resp.Body.Close()
		reqCancel()

		switch {
		case resp.StatusCode == 429:
			slog.Warn("Rate limited, waiting", "seconds", rateLimitWait)
			rateLimitUntil.Store(time.Now().Add(time.Duration(rateLimitWait) * time.Second).Unix())
			if attempt < maxRetries-1 {
				time.Sleep(time.Duration(rateLimitWait) * time.Second)
				continue
			}
			return nil, ErrRateLimited

		case resp.StatusCode == 472:
			return nil, ErrNoData

		case resp.StatusCode >= 500:
			slog.Warn("Server error", "endpoint", endpoint, "status", resp.StatusCode, "attempt", attempt+1)
			if attempt < maxRetries-1 {
				time.Sleep(time.Duration(retryBackoff[attempt]) * time.Second)
				continue
			}
			return nil, fmt.Errorf("server error %d after %d attempts", resp.StatusCode, maxRetries)

		case resp.StatusCode >= 400:
			snippet := string(body)
			if len(snippet) > 200 {
				snippet = snippet[:200]
			}
			return nil, fmt.Errorf("client error %d: %s", resp.StatusCode, snippet)

		case resp.StatusCode == 200:
			if readErr != nil {
				if attempt < maxRetries-1 {
					time.Sleep(time.Duration(retryBackoff[attempt]) * time.Second)
					continue
				}
				return nil, fmt.Errorf("read body: %w", readErr)
			}
			if len(body) == 0 {
				return nil, ErrNoData
			}
			return body, nil

		default:
			return nil, fmt.Errorf("unexpected status %d", resp.StatusCode)
		}
	}
	return nil, fmt.Errorf("max retries exceeded")
}

// VerifyConnection checks if the Theta Terminal is reachable.
func (c *ThetaClient) VerifyConnection() bool {
	params := url.Values{
		"symbol": {"SPY"},
		"format": {"csv"},
	}
	_, err := c.request("/option/list/expirations", params)
	return err == nil || !errors.Is(err, ErrNotRunning)
}

// GetExpirations returns all available expiration dates for a symbol.
func (c *ThetaClient) GetExpirations(symbol string) ([]string, error) {
	params := url.Values{
		"symbol": {symbol},
		"format": {"csv"},
	}
	body, err := c.request("/option/list/expirations", params)
	if err != nil {
		if errors.Is(err, ErrNoData) {
			return nil, nil
		}
		return nil, err
	}
	return parseExpirations(body)
}

// GetOHLCBatch fetches 1-min OHLCV for a full month.
func (c *ThetaClient) GetOHLCBatch(symbol, expiration, startDate, endDate string) ([]OHLCRow, error) {
	params := url.Values{
		"symbol":     {symbol},
		"expiration": {expiration},
		"start_date": {startDate},
		"end_date":   {endDate},
		"interval":   {"1m"},
		"strike":     {"*"},
		"right":      {"both"},
		"format":     {"csv"},
	}
	body, err := c.request("/option/history/ohlc", params)
	if err != nil {
		if errors.Is(err, ErrNoData) {
			return nil, nil
		}
		return nil, err
	}
	return parseOHLCStream(body)
}

// GetQuoteBatch fetches 1-min bid/ask quotes for a full month.
func (c *ThetaClient) GetQuoteBatch(symbol, expiration, startDate, endDate string) ([]QuoteRow, error) {
	params := url.Values{
		"symbol":     {symbol},
		"expiration": {expiration},
		"start_date": {startDate},
		"end_date":   {endDate},
		"interval":   {"1m"},
		"strike":     {"*"},
		"right":      {"both"},
		"format":     {"csv"},
	}
	body, err := c.request("/option/history/quote", params)
	if err != nil {
		if errors.Is(err, ErrNoData) {
			return nil, nil
		}
		return nil, err
	}
	return parseQuoteStream(body)
}

// GetGreeksBatch fetches 1-min first-order Greeks for a full month.
func (c *ThetaClient) GetGreeksBatch(symbol, expiration, startDate, endDate string) ([]GreeksRow, error) {
	params := url.Values{
		"symbol":     {symbol},
		"expiration": {expiration},
		"start_date": {startDate},
		"end_date":   {endDate},
		"interval":   {"1m"},
		"strike":     {"*"},
		"right":      {"both"},
		"format":     {"csv"},
	}
	body, err := c.request("/option/history/greeks/first_order", params)
	if err != nil {
		if errors.Is(err, ErrNoData) {
			return nil, nil
		}
		return nil, err
	}
	return parseGreeksStream(body)
}

// GetEODGreeksDay fetches EOD Greeks for all expirations on a single day.
func (c *ThetaClient) GetEODGreeksDay(symbol, date string) ([]EODGreeksRow, error) {
	params := url.Values{
		"symbol":     {symbol},
		"expiration": {"*"},
		"start_date": {date},
		"end_date":   {date},
		"strike":     {"*"},
		"right":      {"both"},
		"format":     {"csv"},
	}
	body, err := c.request("/option/history/greeks/eod", params)
	if err != nil {
		if errors.Is(err, ErrNoData) {
			return nil, nil
		}
		return nil, err
	}
	return parseEODGreeksStream(body)
}

// GetEODOIDay fetches EOD open interest for all expirations on a single day.
func (c *ThetaClient) GetEODOIDay(symbol, date string) ([]EODOIRow, error) {
	params := url.Values{
		"symbol":     {symbol},
		"expiration": {"*"},
		"start_date": {date},
		"end_date":   {date},
		"strike":     {"*"},
		"right":      {"both"},
		"format":     {"csv"},
	}
	body, err := c.request("/option/history/open_interest", params)
	if err != nil {
		if errors.Is(err, ErrNoData) {
			return nil, nil
		}
		return nil, err
	}
	return parseEODOIStream(body)
}

// isConnectionError checks if the error indicates the server is unreachable.
func isConnectionError(err error) bool {
	if err == nil {
		return false
	}
	s := err.Error()
	for _, substr := range []string{"connection refused", "no such host", "dial tcp"} {
		if len(s) >= len(substr) {
			for i := 0; i <= len(s)-len(substr); i++ {
				if s[i:i+len(substr)] == substr {
					return true
				}
			}
		}
	}
	return false
}
