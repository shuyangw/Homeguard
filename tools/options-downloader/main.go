package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"log/slog"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"time"
)

func main() {
	symbolsFlag := flag.String("symbols", "", "Comma-separated symbols (default: all LIQUID_UNIVERSE)")
	startFlag := flag.String("start", defaultStartDate, "Start date YYYY-MM-DD")
	endFlag := flag.String("end", "", "End date YYYY-MM-DD (default: today)")
	concurrencyFlag := flag.Int("concurrency", defaultConcurrency, "Worker goroutine count")
	timeoutFlag := flag.Int("timeout", defaultTimeout, "HTTP timeout in seconds")
	outputFlag := flag.String("output", defaultOutputDir, "Output directory")
	retryFailedFlag := flag.Bool("retry-failed", false, "Retry previously failed entries")
	showManifestFlag := flag.Bool("show-manifest", false, "Show manifest stats and exit")
	resetFlag := flag.Bool("reset", false, "Reset manifest and start fresh")
	flag.Parse()

	// Determine symbols
	symbols := liquidUniverse
	if *symbolsFlag != "" {
		symbols = nil
		for _, s := range strings.Split(*symbolsFlag, ",") {
			s = strings.TrimSpace(strings.ToUpper(s))
			if s != "" {
				symbols = append(symbols, s)
			}
		}
	}

	// Determine end date
	endDate := *endFlag
	if endDate == "" {
		endDate = time.Now().Format("2006-01-02")
	}

	outputDir := *outputFlag

	// Setup logging
	setupLogging(outputDir)

	// Handle --reset
	if *resetFlag {
		manifestPath := filepath.Join(outputDir, "_manifest_combined.json")
		if err := os.Remove(manifestPath); err == nil {
			slog.Info("Manifest reset")
		}
	}

	// Handle --show-manifest
	if *showManifestFlag {
		manifestPath := filepath.Join(outputDir, "_manifest_combined.json")
		m, err := NewManifestManager(manifestPath)
		if err != nil {
			fmt.Printf("Error: %v\n", err)
			os.Exit(1)
		}
		stats := m.GetStats()
		fmt.Printf("Manifest: %s\n", manifestPath)
		fmt.Printf("Stats: %v\n", stats)
		return
	}

	// Create downloader
	downloader, err := NewDownloader(
		outputDir,
		symbols,
		*startFlag,
		endDate,
		*concurrencyFlag,
		*timeoutFlag,
		*retryFailedFlag,
	)
	if err != nil {
		slog.Error("Failed to create downloader", "error", err)
		os.Exit(1)
	}

	// Setup signal handling for graceful shutdown
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	if err := downloader.Run(ctx); err != nil {
		slog.Error("Download failed", "error", err)
		os.Exit(1)
	}
}

// setupLogging configures slog with both console and file handlers.
func setupLogging(outputDir string) {
	logDir := filepath.Join(outputDir, "_logs")
	os.MkdirAll(logDir, 0755)

	today := time.Now().Format("20060102")
	logFile, err := os.OpenFile(
		filepath.Join(logDir, fmt.Sprintf("download_go_%s.log", today)),
		os.O_CREATE|os.O_APPEND|os.O_WRONLY,
		0644,
	)

	var handler slog.Handler
	if err != nil {
		// Fall back to console only
		handler = slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{
			Level: slog.LevelInfo,
		})
	} else {
		// Multi-writer: console + file
		multi := io.MultiWriter(os.Stdout, logFile)
		handler = slog.NewTextHandler(multi, &slog.HandlerOptions{
			Level: slog.LevelInfo,
		})
	}

	slog.SetDefault(slog.New(handler))
}
