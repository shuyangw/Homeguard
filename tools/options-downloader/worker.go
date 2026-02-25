package main

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"sync/atomic"
	"time"

	"golang.org/x/sync/errgroup"
)

// Downloader orchestrates the download of options data.
type Downloader struct {
	client      *ThetaClient
	manifest    *ManifestManager
	outputDir   string
	symbols     []string
	startDate   string
	endDate     string
	concurrency int
	retryFailed bool

	// Expiration cache
	expCache sync.Map // map[string][]string

	// Progress counters
	completed  atomic.Int64
	failed     atomic.Int64
	totalRows  atomic.Int64
	totalItems int
	startTime  time.Time

	// Expiration-level progress
	totalExps     atomic.Int64
	completedExps atomic.Int64
	currentEntry  sync.Map // workerID -> string (current entry key)
}

// NewDownloader creates a new Downloader.
func NewDownloader(
	outputDir string,
	symbols []string,
	startDate, endDate string,
	concurrency, timeout int,
	retryFailed bool,
) (*Downloader, error) {
	manifestPath := filepath.Join(outputDir, "_manifest_combined.json")
	manifest, err := NewManifestManager(manifestPath)
	if err != nil {
		return nil, fmt.Errorf("load manifest: %w", err)
	}

	return &Downloader{
		client:      NewThetaClient(timeout),
		manifest:    manifest,
		outputDir:   outputDir,
		symbols:     symbols,
		startDate:   startDate,
		endDate:     endDate,
		concurrency: concurrency,
		retryFailed: retryFailed,
	}, nil
}

// partitionPath returns the parquet file path for a symbol-month.
func (d *Downloader) partitionPath(symbol string, year, month int) string {
	return filepath.Join(
		d.outputDir, "options_combined",
		fmt.Sprintf("root=%s", symbol),
		fmt.Sprintf("year=%d", year),
		fmt.Sprintf("month=%02d", month),
		"data.parquet",
	)
}

// partitionExists checks if a parquet file already exists and is non-empty.
func (d *Downloader) partitionExists(symbol string, year, month int) bool {
	path := d.partitionPath(symbol, year, month)
	info, err := os.Stat(path)
	return err == nil && info.Size() > 0
}

// getExpirationsCached returns expirations for a symbol, caching the result.
func (d *Downloader) getExpirationsCached(symbol string) ([]string, error) {
	if v, ok := d.expCache.Load(symbol); ok {
		return v.([]string), nil
	}
	exps, err := d.client.GetExpirations(symbol)
	if err != nil {
		return nil, err
	}
	if exps != nil {
		d.expCache.Store(symbol, exps)
	}
	return exps, nil
}

// filterExpirations returns expirations active during the given month.
func filterExpirations(expirations []string, year, month int) []string {
	monthStart := fmt.Sprintf("%d%02d01", year, month)
	var result []string
	for _, e := range expirations {
		if e >= monthStart {
			result = append(result, e)
		}
	}
	return result
}

// downloadMonth downloads and merges all data for a single symbol-month.
func (d *Downloader) downloadMonth(ctx context.Context, entry *DownloadEntry) (int, error) {
	symbol := entry.Symbol
	year := entry.Year
	month := entry.Month

	// Skip if already downloaded
	if d.partitionExists(symbol, year, month) {
		d.manifest.UpdateEntry(entry.Key(), "complete", 0, nil)
		return 0, nil
	}

	d.manifest.UpdateEntry(entry.Key(), "in_progress", 0, nil)
	d.manifest.Save()

	// Get expirations
	expirations, err := d.getExpirationsCached(symbol)
	if err != nil {
		return 0, err
	}

	validExps := filterExpirations(expirations, year, month)
	if len(validExps) == 0 {
		d.manifest.UpdateEntry(entry.Key(), "complete", 0, nil)
		d.manifest.Save()
		slog.Info("No active expirations", "entry", entry.Key())
		return 0, nil
	}

	d.totalExps.Add(int64(len(validExps)))
	slog.Info("Downloading", "entry", entry.Key(), "expirations", len(validExps))

	// Calculate month date range
	_, lastDay := monthLastDay(year, month)
	startDate := fmt.Sprintf("%d%02d01", year, month)
	endDate := fmt.Sprintf("%d%02d%02d", year, month, lastDay)

	// Download intraday data with bounded parallelism across expirations.
	// Each expiration makes 3 sequential API calls (OHLC, Quote, Greeks).
	// expParallelism controls how many expirations run concurrently per worker.
	type expResult struct {
		rows []*MergedRow
	}
	results := make([]expResult, len(validExps))

	g, gCtx := errgroup.WithContext(ctx)
	g.SetLimit(expParallelism)

	for i, exp := range validExps {
		i, exp := i, exp
		g.Go(func() error {
			if gCtx.Err() != nil {
				return gCtx.Err()
			}
			slog.Info("Fetching expiration", "entry", entry.Key(), "exp", exp, "idx", fmt.Sprintf("%d/%d", i+1, len(validExps)))

			// 1. OHLC (required base)
			ohlcRows, err := d.client.GetOHLCBatch(symbol, exp, startDate, endDate)
			if err != nil {
				if errors.Is(err, ErrRateLimited) {
					return err
				}
				slog.Debug("OHLC fetch failed", "exp", exp, "error", err)
				d.completedExps.Add(1)
				return nil
			}
			if len(ohlcRows) == 0 {
				d.completedExps.Add(1)
				return nil
			}

			// 2. Quote
			quoteRows, err := d.client.GetQuoteBatch(symbol, exp, startDate, endDate)
			if err != nil && errors.Is(err, ErrRateLimited) {
				return err
			}

			// 3. Greeks
			greeksRows, err := d.client.GetGreeksBatch(symbol, exp, startDate, endDate)
			if err != nil && errors.Is(err, ErrRateLimited) {
				return err
			}

			results[i].rows = mergeIntraday(ohlcRows, quoteRows, greeksRows)
			d.completedExps.Add(1)
			return nil
		})
	}

	if err := g.Wait(); err != nil {
		return 0, err
	}

	// Collect all rows in order
	var allRows []*MergedRow
	for _, r := range results {
		allRows = append(allRows, r.rows...)
	}

	slog.Info("Intraday complete", "entry", entry.Key(), "rows", len(allRows))

	if len(allRows) == 0 {
		errMsg := "No data returned"
		d.manifest.UpdateEntry(entry.Key(), "failed", 0, &errMsg)
		d.manifest.Save()
		slog.Warn("No data", "entry", entry.Key())
		return 0, nil
	}

	// Download EOD data: expiration=* requires day-at-a-time requests.
	// Parallelize across days with bounded concurrency, and run greeks+OI
	// concurrently within each day (2 independent calls).
	tradingDates := extractTradingDates(allRows)
	slog.Info("EOD fetch", "entry", entry.Key(), "trading_days", len(tradingDates))

	type eodDayResult struct {
		greeks []EODGreeksRow
		oi     []EODOIRow
	}
	eodResults := make([]eodDayResult, len(tradingDates))

	eodG, eodCtx := errgroup.WithContext(ctx)
	eodG.SetLimit(expParallelism)

	for di, dateStr := range tradingDates {
		di, dateStr := di, dateStr
		eodG.Go(func() error {
			if eodCtx.Err() != nil {
				return eodCtx.Err()
			}

			// Run greeks and OI concurrently for this day
			var dayGreeks []EODGreeksRow
			var dayOI []EODOIRow
			var gErr, oErr error

			var dayWg sync.WaitGroup
			dayWg.Add(2)
			go func() {
				defer dayWg.Done()
				dayGreeks, gErr = d.client.GetEODGreeksDay(symbol, dateStr)
			}()
			go func() {
				defer dayWg.Done()
				dayOI, oErr = d.client.GetEODOIDay(symbol, dateStr)
			}()
			dayWg.Wait()

			if gErr != nil {
				if errors.Is(gErr, ErrRateLimited) {
					return gErr
				}
				slog.Debug("EOD Greeks failed", "date", dateStr, "error", gErr)
			} else {
				eodResults[di].greeks = dayGreeks
			}

			if oErr != nil {
				if errors.Is(oErr, ErrRateLimited) {
					return oErr
				}
				slog.Debug("EOD OI failed", "date", dateStr, "error", oErr)
			} else {
				eodResults[di].oi = dayOI
			}
			return nil
		})
	}

	if err := eodG.Wait(); err != nil {
		return 0, err
	}

	var eodGreeks []EODGreeksRow
	var eodOI []EODOIRow
	for _, r := range eodResults {
		eodGreeks = append(eodGreeks, r.greeks...)
		eodOI = append(eodOI, r.oi...)
	}

	// Merge EOD data onto intraday
	mergeEOD(allRows, eodGreeks, eodOI)

	// Filter out non-trading days (weekends + US market holidays)
	allRows = filterTradingDays(allRows)

	if len(allRows) == 0 {
		errMsg := "No trading-day data"
		d.manifest.UpdateEntry(entry.Key(), "failed", 0, &errMsg)
		d.manifest.Save()
		slog.Warn("No trading-day data after filter", "entry", entry.Key())
		return 0, nil
	}

	// Write parquet
	path := d.partitionPath(symbol, year, month)
	if err := writeParquet(path, allRows); err != nil {
		errMsg := err.Error()
		d.manifest.UpdateEntry(entry.Key(), "failed", 0, &errMsg)
		d.manifest.Save()
		return 0, fmt.Errorf("write parquet: %w", err)
	}

	totalRows := len(allRows)
	d.manifest.UpdateEntry(entry.Key(), "complete", totalRows, nil)
	d.manifest.Save()

	slog.Info("[+] Complete",
		"entry", entry.Key(),
		"rows", totalRows,
		"expirations", len(validExps),
	)

	return totalRows, nil
}

// Run starts the download process.
func (d *Downloader) Run(ctx context.Context) error {
	slog.Info("============================================================")
	slog.Info("THETADATA GO OPTIONS DOWNLOADER")
	slog.Info("============================================================")
	slog.Info("Config",
		"symbols", len(d.symbols),
		"start", d.startDate,
		"end", d.endDate,
		"concurrency", d.concurrency,
		"output", d.outputDir,
	)

	if !d.client.VerifyConnection() {
		return fmt.Errorf("cannot connect to Theta Terminal at %s", thetaBaseURL)
	}
	slog.Info("[+] Connected to Theta Terminal")

	// Initialize entries
	newEntries := d.manifest.InitializeEntries(d.symbols, d.startDate, d.endDate)
	slog.Info("Manifest initialized", "new_entries", newEntries)

	pending := d.manifest.GetPending(d.retryFailed)
	stats := d.manifest.GetStats()
	slog.Info("Manifest stats", "stats", stats, "pending", len(pending))

	// Pre-filter: skip partitions that already exist on disk
	var filtered []*DownloadEntry
	skipped := 0
	for _, entry := range pending {
		if d.partitionExists(entry.Symbol, entry.Year, entry.Month) {
			d.manifest.UpdateEntry(entry.Key(), "complete", 0, nil)
			skipped++
		} else {
			filtered = append(filtered, entry)
		}
	}
	if skipped > 0 {
		d.manifest.Save()
		slog.Info("Skipped existing partitions", "count", skipped)
	}

	pending = filtered
	slog.Info("To process", "count", len(pending))

	if len(pending) == 0 {
		slog.Info("Nothing to download!")
		return nil
	}

	// Interleave symbols for balanced load (newest first within each symbol)
	interleaved := interleaveEntries(pending)
	d.totalItems = len(interleaved)
	d.startTime = time.Now()

	slog.Info("Starting workers",
		"entries", d.totalItems,
		"concurrency", d.concurrency,
	)

	// Create work channel
	workCh := make(chan *DownloadEntry, len(interleaved))
	for _, entry := range interleaved {
		workCh <- entry
	}
	close(workCh)

	// Start progress monitor
	progressCtx, progressCancel := context.WithCancel(ctx)
	defer progressCancel()
	go d.progressMonitor(progressCtx)

	// Launch worker goroutines
	var wg sync.WaitGroup
	for i := 0; i < d.concurrency; i++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			for entry := range workCh {
				if ctx.Err() != nil {
					return
				}
				rows, err := d.downloadMonth(ctx, entry)
				if err != nil {
					if errors.Is(err, ErrRateLimited) {
						slog.Warn("Rate limited, waiting",
							"entry", entry.Key(),
							"seconds", rateLimitWait,
						)
						time.Sleep(time.Duration(rateLimitWait) * time.Second)
						// Retry once
						rows, err = d.downloadMonth(ctx, entry)
					}
					if err != nil {
						errMsg := err.Error()
						d.manifest.UpdateEntry(entry.Key(), "failed", 0, &errMsg)
						d.manifest.Save()
						d.failed.Add(1)
						slog.Error("Failed", "entry", entry.Key(), "error", err)
						continue
					}
				}
				d.completed.Add(1)
				d.totalRows.Add(int64(rows))
			}
		}(i)
	}

	wg.Wait()
	progressCancel()
	d.manifest.Save()

	elapsed := time.Since(d.startTime)
	slog.Info("============================================================")
	slog.Info("DOWNLOAD COMPLETE",
		"completed", d.completed.Load(),
		"failed", d.failed.Load(),
		"total_rows", d.totalRows.Load(),
		"elapsed", elapsed.Round(time.Second),
	)
	slog.Info("============================================================")

	return nil
}

// progressMonitor logs progress every 10 seconds.
func (d *Downloader) progressMonitor(ctx context.Context) {
	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			completed := d.completed.Load()
			failed := d.failed.Load()
			processed := completed + failed
			elapsed := time.Since(d.startTime).Seconds()

			// Expiration-level progress for ETA (more granular than month-level)
			doneExps := d.completedExps.Load()
			totalExps := d.totalExps.Load()

			// Use expiration-level rate for ETA when available (more accurate)
			var etaStr string
			if doneExps > 0 && totalExps > 0 && doneExps < totalExps {
				expRate := float64(doneExps) / elapsed
				remainingExps := float64(totalExps) - float64(doneExps)
				etaStr = formatDuration(remainingExps / expRate)
			} else if processed > 0 && int(processed) < d.totalItems {
				rate := float64(processed) / elapsed
				remaining := float64(d.totalItems) - float64(processed)
				etaStr = formatDuration(remaining / rate)
			} else {
				continue
			}

			pct := float64(processed) / float64(d.totalItems) * 100

			slog.Info("[PROGRESS]",
				"months", fmt.Sprintf("%d/%d (%.1f%%)", processed, d.totalItems, pct),
				"expirations", fmt.Sprintf("%d/%d", doneExps, totalExps),
				"rows", d.totalRows.Load(),
				"failed", failed,
				"elapsed", formatDuration(elapsed),
				"eta", etaStr,
			)
		}
	}
}

// interleaveEntries round-robins entries across symbols, newest first.
func interleaveEntries(entries []*DownloadEntry) []*DownloadEntry {
	bySymbol := make(map[string][]*DownloadEntry)
	for _, e := range entries {
		bySymbol[e.Symbol] = append(bySymbol[e.Symbol], e)
	}

	// Sort each symbol's entries newest first
	for _, es := range bySymbol {
		sort.Slice(es, func(i, j int) bool {
			if es[i].Year != es[j].Year {
				return es[i].Year > es[j].Year
			}
			return es[i].Month > es[j].Month
		})
	}

	// Get sorted symbol list for deterministic ordering
	symbols := make([]string, 0, len(bySymbol))
	for s := range bySymbol {
		symbols = append(symbols, s)
	}
	sort.Strings(symbols)

	// Round-robin
	var result []*DownloadEntry
	iterators := make(map[string]int)
	for _, s := range symbols {
		iterators[s] = 0
	}

	for len(iterators) > 0 {
		for _, s := range symbols {
			idx, ok := iterators[s]
			if !ok {
				continue
			}
			es := bySymbol[s]
			if idx >= len(es) {
				delete(iterators, s)
				continue
			}
			result = append(result, es[idx])
			iterators[s] = idx + 1
		}
	}

	return result
}

// monthLastDay returns the last day of a given month.
func monthLastDay(year, month int) (int, int) {
	t := time.Date(year, time.Month(month)+1, 0, 0, 0, 0, 0, time.UTC)
	return year, t.Day()
}

// extractTradingDates returns sorted unique YYYYMMDD dates from merged rows.
// This tells us which days actually have intraday data (i.e. trading days).
func extractTradingDates(rows []*MergedRow) []string {
	seen := make(map[string]bool)
	for _, r := range rows {
		d := extractDateFromTimestamp(r.Timestamp)
		if d != "" && !seen[d] {
			seen[d] = true
		}
	}
	dates := make([]string, 0, len(seen))
	for d := range seen {
		dates = append(dates, d)
	}
	sort.Strings(dates)
	return dates
}

// filterTradingDays removes rows whose timestamp falls on a weekend or US market holiday.
func filterTradingDays(rows []*MergedRow) []*MergedRow {
	result := make([]*MergedRow, 0, len(rows))
	for _, r := range rows {
		dateStr := extractDateFromTimestamp(r.Timestamp)
		if dateStr == "" || len(dateStr) != 8 {
			continue
		}
		t, err := time.Parse("20060102", dateStr)
		if err != nil {
			continue
		}
		wd := t.Weekday()
		if wd == time.Saturday || wd == time.Sunday {
			continue
		}
		if isUSMarketHoliday(t) {
			continue
		}
		result = append(result, r)
	}
	return result
}

// isUSMarketHoliday returns true if the date is a US stock market holiday.
// Covers NYSE/NASDAQ observed holidays from 2012-2030.
func isUSMarketHoliday(t time.Time) bool {
	key := t.Format("20060102")
	_, ok := usMarketHolidays[key]
	return ok
}

// usMarketHolidays is a set of US stock market holidays (YYYYMMDD).
// Generated from NYSE holiday calendar. Covers 2012-2030.
var usMarketHolidays = map[string]bool{
	// 2012
	"20120102": true, "20120116": true, "20120220": true, "20120406": true,
	"20120528": true, "20120704": true, "20120903": true, "20121122": true,
	"20121225": true,
	// 2013
	"20130101": true, "20130121": true, "20130218": true, "20130329": true,
	"20130527": true, "20130704": true, "20130902": true, "20131128": true,
	"20131225": true,
	// 2014
	"20140101": true, "20140120": true, "20140217": true, "20140418": true,
	"20140526": true, "20140704": true, "20140901": true, "20141127": true,
	"20141225": true,
	// 2015
	"20150101": true, "20150119": true, "20150216": true, "20150403": true,
	"20150525": true, "20150703": true, "20150907": true, "20151126": true,
	"20151225": true,
	// 2016
	"20160101": true, "20160118": true, "20160215": true, "20160325": true,
	"20160530": true, "20160704": true, "20160905": true, "20161124": true,
	"20161226": true,
	// 2017
	"20170102": true, "20170116": true, "20170220": true, "20170414": true,
	"20170529": true, "20170704": true, "20170904": true, "20171123": true,
	"20171225": true,
	// 2018
	"20180101": true, "20180115": true, "20180219": true, "20180330": true,
	"20180528": true, "20180704": true, "20180903": true, "20181122": true,
	"20181225": true,
	// 2019
	"20190101": true, "20190121": true, "20190218": true, "20190419": true,
	"20190527": true, "20190704": true, "20190902": true, "20191128": true,
	"20191225": true,
	// 2020
	"20200101": true, "20200120": true, "20200217": true, "20200410": true,
	"20200525": true, "20200703": true, "20200907": true, "20201126": true,
	"20201225": true,
	// 2021
	"20210101": true, "20210118": true, "20210215": true, "20210402": true,
	"20210531": true, "20210705": true, "20210906": true, "20211125": true,
	"20211224": true,
	// 2022
	"20220117": true, "20220221": true, "20220415": true, "20220530": true,
	"20220620": true, "20220704": true, "20220905": true, "20221124": true,
	"20221226": true,
	// 2023
	"20230102": true, "20230116": true, "20230220": true, "20230407": true,
	"20230529": true, "20230619": true, "20230704": true, "20230904": true,
	"20231123": true, "20231225": true,
	// 2024
	"20240101": true, "20240115": true, "20240219": true, "20240329": true,
	"20240527": true, "20240619": true, "20240704": true, "20240902": true,
	"20241128": true, "20241225": true,
	// 2025
	"20250101": true, "20250120": true, "20250217": true, "20250418": true,
	"20250526": true, "20250619": true, "20250704": true, "20250901": true,
	"20251127": true, "20251225": true,
	// 2026
	"20260101": true, "20260119": true, "20260216": true, "20260403": true,
	"20260525": true, "20260619": true, "20260703": true, "20260907": true,
	"20261126": true, "20261225": true,
	// 2027
	"20270101": true, "20270118": true, "20270215": true, "20270326": true,
	"20270531": true, "20270618": true, "20270705": true, "20270906": true,
	"20271125": true, "20271224": true,
	// 2028
	"20280117": true, "20280221": true, "20280414": true, "20280529": true,
	"20280619": true, "20280704": true, "20280904": true, "20281123": true,
	"20281225": true,
	// 2029
	"20290101": true, "20290115": true, "20290219": true, "20290330": true,
	"20290528": true, "20290619": true, "20290704": true, "20290903": true,
	"20291122": true, "20291225": true,
	// 2030
	"20300101": true, "20300121": true, "20300218": true, "20300419": true,
	"20300527": true, "20300619": true, "20300704": true, "20300902": true,
	"20301128": true, "20301225": true,
}

// formatDuration formats seconds into a human-readable string.
func formatDuration(seconds float64) string {
	if seconds < 60 {
		return fmt.Sprintf("%.0fs", seconds)
	} else if seconds < 3600 {
		return fmt.Sprintf("%.1fm", seconds/60)
	}
	return fmt.Sprintf("%.1fh", seconds/3600)
}
