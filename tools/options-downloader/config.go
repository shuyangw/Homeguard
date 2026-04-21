package main

const (
	thetaBaseURL       = "http://localhost:25503/v3"
	defaultStartDate   = "2012-06-01"
	defaultOutputDir   = `H:\Stock_Data\options`
	defaultConcurrency = 8
	defaultTimeout     = 300 // seconds
	maxRetries         = 3
	rateLimitWait      = 60 // seconds
	expParallelism     = 3  // concurrent expirations per worker
)

var retryBackoff = [3]int{2, 4, 8} // seconds

var liquidUniverse = []string{
	// Index ETFs
	"SPY", "QQQ", "IWM", "DIA",
	// Index Options
	"SPX",
	// Tech Mega-caps
	"NVDA", "TSLA", "AAPL", "MSFT", "AMD", "AMZN", "META", "GOOGL", "AVGO", "PLTR",
	// Sector ETFs
	"XLF", "XLK", "XLV", "XLI", "XLE", "SMH",
	// Bonds
	"TLT",
	// Commodities
	"GLD", "SLV",
	// International
	"EEM", "FXI",
	// Crypto-related
	"IBIT", "MSTR", "COIN",
	// Volatility
	"VIX",
}

// symbolStartDates maps symbols to their earliest data availability date.
// Don't request data before these dates (pre-IPO or pre-listing).
var symbolStartDates = map[string]string{
	"IBIT": "2024-01-01",
	"COIN": "2021-04-01",
	"PLTR": "2020-10-01",
	"META": "2021-11-01",
	"FB":   "2012-06-01",
	"MSTR": "2000-01-01",
	"AVGO": "2012-06-01",
	"SMH":  "2012-06-01",
}
