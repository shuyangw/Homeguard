package main

import "strings"

// mergeIntraday takes OHLC as the base and left-joins Quote and Greeks data.
// Returns merged rows keyed by (timestamp, expiration, strike, right).
func mergeIntraday(ohlc []OHLCRow, quotes []QuoteRow, greeks []GreeksRow) []*MergedRow {
	if len(ohlc) == 0 {
		return nil
	}

	// Build base map from OHLC
	rows := make([]*MergedRow, 0, len(ohlc))
	index := make(map[JoinKey]*MergedRow, len(ohlc))

	for i := range ohlc {
		o := &ohlc[i]
		r := &MergedRow{
			Timestamp:  o.Timestamp,
			Expiration: o.Expiration,
			Strike:     o.Strike,
			Right:      o.Right,
			Open:       o.Open,
			High:       o.High,
			Low:        o.Low,
			Close:      o.Close,
			Volume:     o.Volume,
			TradeCount: o.TradeCount,
			VWAP:       o.VWAP,
		}
		rows = append(rows, r)
		key := JoinKey{
			Timestamp:  o.Timestamp,
			Expiration: o.Expiration,
			Strike:     o.Strike,
			Right:      o.Right,
		}
		index[key] = r
	}

	// Left-join Quote data
	for i := range quotes {
		q := &quotes[i]
		key := JoinKey{
			Timestamp:  q.Timestamp,
			Expiration: q.Expiration,
			Strike:     q.Strike,
			Right:      q.Right,
		}
		if r, ok := index[key]; ok {
			r.BidClose = floatPtr(q.BidClose)
			r.AskClose = floatPtr(q.AskClose)
		}
	}

	// Left-join Greeks data
	for i := range greeks {
		g := &greeks[i]
		key := JoinKey{
			Timestamp:  g.Timestamp,
			Expiration: g.Expiration,
			Strike:     g.Strike,
			Right:      g.Right,
		}
		if r, ok := index[key]; ok {
			r.ImpliedVol = floatPtr(g.ImpliedVol)
			r.Delta = floatPtr(g.Delta)
			r.Theta = floatPtr(g.Theta)
			r.Vega = floatPtr(g.Vega)
			r.UnderlyingPx = floatPtr(g.UnderlyingPx)
		}
	}

	return rows
}

// mergeEOD enriches merged rows with EOD gamma and open interest.
// Joins on (date extracted from timestamp, expiration, strike, right).
func mergeEOD(rows []*MergedRow, eodGreeks []EODGreeksRow, eodOI []EODOIRow) {
	// Build EOD gamma lookup
	gammaMap := make(map[EODKey]float64, len(eodGreeks))
	for i := range eodGreeks {
		g := &eodGreeks[i]
		key := EODKey{
			Date:       g.Date,
			Expiration: g.Expiration,
			Strike:     g.Strike,
			Right:      g.Right,
		}
		gammaMap[key] = g.Gamma
	}

	// Build EOD OI lookup
	oiMap := make(map[EODKey]int64, len(eodOI))
	for i := range eodOI {
		o := &eodOI[i]
		key := EODKey{
			Date:       o.Date,
			Expiration: o.Expiration,
			Strike:     o.Strike,
			Right:      o.Right,
		}
		oiMap[key] = o.OpenInterest
	}

	// Enrich each row
	for _, r := range rows {
		dateStr := extractDateFromTimestamp(r.Timestamp)
		key := EODKey{
			Date:       dateStr,
			Expiration: r.Expiration,
			Strike:     r.Strike,
			Right:      r.Right,
		}
		if gamma, ok := gammaMap[key]; ok {
			r.GammaEOD = floatPtr(gamma)
		}
		if oi, ok := oiMap[key]; ok {
			r.OpenInterestEOD = int64Ptr(oi)
		}
	}
}

// extractDateFromTimestamp extracts YYYYMMDD from a timestamp string.
// Handles ISO format "2024-01-15 09:30:00" or "2024-01-15T09:30:00"
// and epoch milliseconds.
func extractDateFromTimestamp(ts string) string {
	ts = strings.TrimSpace(ts)
	if len(ts) == 0 {
		return ""
	}
	// If it looks like a date string (starts with digit and contains -)
	if len(ts) >= 10 && (ts[4] == '-') {
		return strings.ReplaceAll(ts[:10], "-", "")
	}
	// If it's a pure numeric string (epoch ms), convert
	if len(ts) >= 13 {
		allDigits := true
		for _, c := range ts {
			if c < '0' || c > '9' {
				allDigits = false
				break
			}
		}
		if allDigits {
			ms := parseInt64(ts)
			if ms > 0 {
				t := epochMsToTime(ms)
				return t.Format("20060102")
			}
		}
	}
	// Fallback: strip hyphens from first 10 chars
	if len(ts) >= 8 {
		return strings.ReplaceAll(ts[:min(10, len(ts))], "-", "")
	}
	return ts
}
