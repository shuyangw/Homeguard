package main

import (
	"bytes"
	"encoding/csv"
	"io"
	"strconv"
	"strings"
)

// colIdx safely gets a column value by name from a row using a header index map.
func colIdx(header map[string]int, row []string, name string) string {
	idx, ok := header[name]
	if !ok || idx >= len(row) {
		return ""
	}
	return strings.TrimSpace(row[idx])
}

func parseFloat(s string) float64 {
	if s == "" {
		return 0
	}
	v, _ := strconv.ParseFloat(s, 64)
	return v
}

func parseInt32(s string) int32 {
	if s == "" {
		return 0
	}
	v, _ := strconv.ParseInt(s, 10, 32)
	return int32(v)
}

func parseInt64(s string) int64 {
	if s == "" {
		return 0
	}
	v, _ := strconv.ParseInt(s, 10, 64)
	return v
}

func floatPtr(v float64) *float64 { return &v }
func int64Ptr(v int64) *int64     { return &v }

// readCSVHeader reads the first row of a CSV stream and returns a column-name
// to index map. The csv.Reader is positioned at the first data row afterward.
func readCSVHeader(r *csv.Reader) (map[string]int, error) {
	headerRow, err := r.Read()
	if err != nil {
		return nil, err
	}
	header := make(map[string]int, len(headerRow))
	for i, col := range headerRow {
		header[strings.TrimSpace(strings.ToLower(col))] = i
	}
	return header, nil
}

// parseExpirations parses the expirations endpoint CSV from raw bytes.
// This is a small response so we keep it as bytes-based.
func parseExpirations(data []byte) ([]string, error) {
	r := csv.NewReader(bytes.NewReader(data))
	r.LazyQuotes = true

	header, err := readCSVHeader(r)
	if err != nil {
		return nil, err
	}

	var result []string
	for {
		row, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}
		exp := colIdx(header, row, "expiration")
		if exp != "" {
			exp = strings.ReplaceAll(exp, "-", "")
			result = append(result, exp)
		}
	}
	return result, nil
}

// parseOHLCStream parses the /option/history/ohlc CSV row-by-row.
func parseOHLCStream(data []byte) ([]OHLCRow, error) {
	r := csv.NewReader(bytes.NewReader(data))
	r.LazyQuotes = true
	r.ReuseRecord = true

	header, err := readCSVHeader(r)
	if err != nil {
		if err == io.EOF {
			return nil, nil
		}
		return nil, err
	}

	var result []OHLCRow
	for {
		row, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return result, err
		}
		rec := OHLCRow{
			Timestamp:  colIdx(header, row, "timestamp"),
			Expiration: colIdx(header, row, "expiration"),
			Strike:     parseFloat(colIdx(header, row, "strike")),
			Right:      colIdx(header, row, "right"),
			Open:       parseFloat(colIdx(header, row, "open")),
			High:       parseFloat(colIdx(header, row, "high")),
			Low:        parseFloat(colIdx(header, row, "low")),
			Close:      parseFloat(colIdx(header, row, "close")),
			Volume:     parseInt32(colIdx(header, row, "volume")),
			VWAP:       parseFloat(colIdx(header, row, "vwap")),
		}
		tc := colIdx(header, row, "count")
		if tc == "" {
			tc = colIdx(header, row, "trade_count")
		}
		rec.TradeCount = parseInt32(tc)
		result = append(result, rec)
	}
	return result, nil
}

// parseQuoteStream parses the /option/history/quote CSV row-by-row.
func parseQuoteStream(data []byte) ([]QuoteRow, error) {
	r := csv.NewReader(bytes.NewReader(data))
	r.LazyQuotes = true
	r.ReuseRecord = true

	header, err := readCSVHeader(r)
	if err != nil {
		if err == io.EOF {
			return nil, nil
		}
		return nil, err
	}

	var result []QuoteRow
	for {
		row, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return result, err
		}
		rec := QuoteRow{
			Timestamp:  colIdx(header, row, "timestamp"),
			Expiration: colIdx(header, row, "expiration"),
			Strike:     parseFloat(colIdx(header, row, "strike")),
			Right:      colIdx(header, row, "right"),
		}
		bid := colIdx(header, row, "bid")
		if bid == "" {
			bid = colIdx(header, row, "bid_close")
		}
		ask := colIdx(header, row, "ask")
		if ask == "" {
			ask = colIdx(header, row, "ask_close")
		}
		rec.BidClose = parseFloat(bid)
		rec.AskClose = parseFloat(ask)
		result = append(result, rec)
	}
	return result, nil
}

// parseGreeksStream parses the /option/history/greeks/first_order CSV row-by-row.
func parseGreeksStream(data []byte) ([]GreeksRow, error) {
	r := csv.NewReader(bytes.NewReader(data))
	r.LazyQuotes = true
	r.ReuseRecord = true

	header, err := readCSVHeader(r)
	if err != nil {
		if err == io.EOF {
			return nil, nil
		}
		return nil, err
	}

	var result []GreeksRow
	for {
		row, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return result, err
		}
		rec := GreeksRow{
			Timestamp:  colIdx(header, row, "timestamp"),
			Expiration: colIdx(header, row, "expiration"),
			Strike:     parseFloat(colIdx(header, row, "strike")),
			Right:      colIdx(header, row, "right"),
			ImpliedVol: parseFloat(colIdx(header, row, "implied_vol")),
			Delta:      parseFloat(colIdx(header, row, "delta")),
			Theta:      parseFloat(colIdx(header, row, "theta")),
			Vega:       parseFloat(colIdx(header, row, "vega")),
		}
		up := colIdx(header, row, "underlying_price")
		if up == "" {
			up = colIdx(header, row, "underlying_px")
		}
		rec.UnderlyingPx = parseFloat(up)
		result = append(result, rec)
	}
	return result, nil
}

// parseEODGreeksStream parses the /option/history/greeks/eod CSV row-by-row.
func parseEODGreeksStream(data []byte) ([]EODGreeksRow, error) {
	r := csv.NewReader(bytes.NewReader(data))
	r.LazyQuotes = true
	r.ReuseRecord = true

	header, err := readCSVHeader(r)
	if err != nil {
		if err == io.EOF {
			return nil, nil
		}
		return nil, err
	}

	var result []EODGreeksRow
	for {
		row, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return result, err
		}
		dateVal := colIdx(header, row, "date")
		if dateVal == "" {
			dateVal = colIdx(header, row, "timestamp")
		}
		if dateVal == "" {
			dateVal = colIdx(header, row, "trade_date")
		}
		rec := EODGreeksRow{
			Date:       normalizeDateToYYYYMMDD(dateVal),
			Expiration: colIdx(header, row, "expiration"),
			Strike:     parseFloat(colIdx(header, row, "strike")),
			Right:      colIdx(header, row, "right"),
			Gamma:      parseFloat(colIdx(header, row, "gamma")),
		}
		result = append(result, rec)
	}
	return result, nil
}

// parseEODOIStream parses the /option/history/open_interest CSV row-by-row.
func parseEODOIStream(data []byte) ([]EODOIRow, error) {
	r := csv.NewReader(bytes.NewReader(data))
	r.LazyQuotes = true
	r.ReuseRecord = true

	header, err := readCSVHeader(r)
	if err != nil {
		if err == io.EOF {
			return nil, nil
		}
		return nil, err
	}

	var result []EODOIRow
	for {
		row, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return result, err
		}
		dateVal := colIdx(header, row, "date")
		if dateVal == "" {
			dateVal = colIdx(header, row, "timestamp")
		}
		if dateVal == "" {
			dateVal = colIdx(header, row, "trade_date")
		}
		rec := EODOIRow{
			Date:         normalizeDateToYYYYMMDD(dateVal),
			Expiration:   colIdx(header, row, "expiration"),
			Strike:       parseFloat(colIdx(header, row, "strike")),
			Right:        colIdx(header, row, "right"),
			OpenInterest: parseInt64(colIdx(header, row, "open_interest")),
		}
		result = append(result, rec)
	}
	return result, nil
}

// normalizeDateToYYYYMMDD converts various date formats to YYYYMMDD.
func normalizeDateToYYYYMMDD(s string) string {
	s = strings.TrimSpace(s)
	if len(s) == 0 {
		return ""
	}
	if idx := strings.Index(s, " "); idx > 0 {
		s = s[:idx]
	}
	result := strings.ReplaceAll(s, "-", "")
	if len(result) > 8 {
		result = result[:8]
	}
	return result
}
