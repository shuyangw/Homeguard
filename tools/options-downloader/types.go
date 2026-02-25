package main

// OHLCRow is a parsed row from the /option/history/ohlc endpoint.
type OHLCRow struct {
	Timestamp  string
	Expiration string
	Strike     float64
	Right      string
	Open       float64
	High       float64
	Low        float64
	Close      float64
	Volume     int32
	TradeCount int32
	VWAP       float64
}

// QuoteRow is a parsed row from the /option/history/quote endpoint.
type QuoteRow struct {
	Timestamp  string
	Expiration string
	Strike     float64
	Right      string
	BidClose   float64
	AskClose   float64
}

// GreeksRow is a parsed row from the /option/history/greeks/first_order endpoint.
type GreeksRow struct {
	Timestamp    string
	Expiration   string
	Strike       float64
	Right        string
	ImpliedVol   float64
	Delta        float64
	Theta        float64
	Vega         float64
	UnderlyingPx float64
}

// EODGreeksRow is a parsed row from the /option/history/greeks/eod endpoint.
type EODGreeksRow struct {
	Date       string
	Expiration string
	Strike     float64
	Right      string
	Gamma      float64
}

// EODOIRow is a parsed row from the /option/history/open_interest endpoint.
type EODOIRow struct {
	Date         string
	Expiration   string
	Strike       float64
	Right        string
	OpenInterest int64
}

// MergedRow is the final 20-column output row written to parquet.
type MergedRow struct {
	Timestamp       string
	Expiration      string
	Strike          float64
	Right           string
	Open            float64
	High            float64
	Low             float64
	Close           float64
	Volume          int32
	TradeCount      int32
	VWAP            float64
	BidClose        *float64
	AskClose        *float64
	ImpliedVol      *float64
	Delta           *float64
	Theta           *float64
	Vega            *float64
	UnderlyingPx    *float64
	GammaEOD        *float64
	OpenInterestEOD *int64
}

// JoinKey is the composite key for intraday merges.
type JoinKey struct {
	Timestamp  string
	Expiration string
	Strike     float64
	Right      string
}

// EODKey is the composite key for EOD merges.
type EODKey struct {
	Date       string // YYYYMMDD
	Expiration string
	Strike     float64
	Right      string
}

// DownloadEntry represents one symbol-month download task in the manifest.
type DownloadEntry struct {
	Symbol       string  `json:"symbol"`
	Year         int     `json:"year"`
	Month        int     `json:"month"`
	State        string  `json:"state"`
	RowsCount    int     `json:"rows_count"`
	ErrorMessage *string `json:"error_message"`
	CompletedAt  *string `json:"completed_at"`
}

// Key returns the manifest key for this entry (e.g. "SPY_2024_01").
func (e *DownloadEntry) Key() string {
	return entryKey(e.Symbol, e.Year, e.Month)
}

// Manifest is the top-level download tracking structure.
type Manifest struct {
	CreatedAt string                    `json:"created_at"`
	UpdatedAt string                    `json:"updated_at"`
	Entries   map[string]*DownloadEntry `json:"entries"`
}
