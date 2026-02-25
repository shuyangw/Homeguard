package main

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"time"

	"github.com/parquet-go/parquet-go"
	"github.com/parquet-go/parquet-go/compress/zstd"
)

// ParquetRow is the struct that maps directly to the parquet schema.
// Field names and tags control column names and types in the output.
type ParquetRow struct {
	Timestamp       string   `parquet:"timestamp"`
	Expiration      string   `parquet:"expiration"`
	Strike          float64  `parquet:"strike"`
	Right           string   `parquet:"right"`
	Open            float64  `parquet:"open"`
	High            float64  `parquet:"high"`
	Low             float64  `parquet:"low"`
	Close           float64  `parquet:"close"`
	Volume          int32    `parquet:"volume"`
	TradeCount      int32    `parquet:"trade_count"`
	VWAP            float64  `parquet:"vwap"`
	BidClose        float64  `parquet:"bid_close"`
	AskClose        float64  `parquet:"ask_close"`
	ImpliedVol      float64  `parquet:"implied_vol"`
	Delta           float64  `parquet:"delta"`
	Theta           float64  `parquet:"theta"`
	Vega            float64  `parquet:"vega"`
	UnderlyingPx    float64  `parquet:"underlying_px"`
	GammaEOD        float64  `parquet:"gamma_eod"`
	OpenInterestEOD *int64   `parquet:"open_interest_eod,optional"`
}

// writeParquet writes merged rows to a parquet file with zstd compression.
// Uses atomic write: writes to .tmp then renames.
func writeParquet(path string, rows []*MergedRow) error {
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("mkdir %s: %w", dir, err)
	}

	tmpPath := path + ".tmp"

	f, err := os.Create(tmpPath)
	if err != nil {
		return fmt.Errorf("create %s: %w", tmpPath, err)
	}

	writer := parquet.NewGenericWriter[ParquetRow](f,
		parquet.Compression(&zstd.Codec{}),
	)

	// Convert MergedRows to ParquetRows in batches
	const batchSize = 10000
	batch := make([]ParquetRow, 0, batchSize)

	for _, r := range rows {
		pr := ParquetRow{
			Timestamp:  r.Timestamp,
			Expiration: r.Expiration,
			Strike:     r.Strike,
			Right:      r.Right,
			Open:       r.Open,
			High:       r.High,
			Low:        r.Low,
			Close:      r.Close,
			Volume:     r.Volume,
			TradeCount: r.TradeCount,
			VWAP:       r.VWAP,
		}
		// Use NaN for nullable float64 fields when nil
		if r.BidClose != nil {
			pr.BidClose = *r.BidClose
		} else {
			pr.BidClose = math.NaN()
		}
		if r.AskClose != nil {
			pr.AskClose = *r.AskClose
		} else {
			pr.AskClose = math.NaN()
		}
		if r.ImpliedVol != nil {
			pr.ImpliedVol = *r.ImpliedVol
		} else {
			pr.ImpliedVol = math.NaN()
		}
		if r.Delta != nil {
			pr.Delta = *r.Delta
		} else {
			pr.Delta = math.NaN()
		}
		if r.Theta != nil {
			pr.Theta = *r.Theta
		} else {
			pr.Theta = math.NaN()
		}
		if r.Vega != nil {
			pr.Vega = *r.Vega
		} else {
			pr.Vega = math.NaN()
		}
		if r.UnderlyingPx != nil {
			pr.UnderlyingPx = *r.UnderlyingPx
		} else {
			pr.UnderlyingPx = math.NaN()
		}
		if r.GammaEOD != nil {
			pr.GammaEOD = *r.GammaEOD
		} else {
			pr.GammaEOD = math.NaN()
		}
		if r.OpenInterestEOD != nil {
			pr.OpenInterestEOD = r.OpenInterestEOD
		} else {
			pr.OpenInterestEOD = nil
		}

		batch = append(batch, pr)
		if len(batch) >= batchSize {
			if _, err := writer.Write(batch); err != nil {
				f.Close()
				os.Remove(tmpPath)
				return fmt.Errorf("write batch: %w", err)
			}
			batch = batch[:0]
		}
	}

	// Write remaining
	if len(batch) > 0 {
		if _, err := writer.Write(batch); err != nil {
			f.Close()
			os.Remove(tmpPath)
			return fmt.Errorf("write final batch: %w", err)
		}
	}

	if err := writer.Close(); err != nil {
		f.Close()
		os.Remove(tmpPath)
		return fmt.Errorf("close writer: %w", err)
	}
	if err := f.Close(); err != nil {
		os.Remove(tmpPath)
		return fmt.Errorf("close file: %w", err)
	}

	// Atomic rename
	if err := os.Rename(tmpPath, path); err != nil {
		os.Remove(tmpPath)
		return fmt.Errorf("rename %s -> %s: %w", tmpPath, path, err)
	}

	return nil
}

// epochMsToTime converts epoch milliseconds to time.Time.
func epochMsToTime(ms int64) time.Time {
	return time.UnixMilli(ms)
}
