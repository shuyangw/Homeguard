package main

import (
	"encoding/json"
	"fmt"
	"os"
	"sync"
	"time"
)

// ManifestManager provides thread-safe access to the download manifest.
type ManifestManager struct {
	path     string
	manifest *Manifest
	mu       sync.Mutex
}

// NewManifestManager loads or creates a manifest at the given path.
func NewManifestManager(path string) (*ManifestManager, error) {
	m := &ManifestManager{
		path: path,
	}
	if err := m.loadOrCreate(); err != nil {
		return nil, err
	}
	return m, nil
}

func (m *ManifestManager) loadOrCreate() error {
	data, err := os.ReadFile(m.path)
	if err != nil {
		// File doesn't exist, create new manifest
		m.manifest = &Manifest{
			CreatedAt: time.Now().Format(time.RFC3339),
			Entries:   make(map[string]*DownloadEntry),
		}
		return nil
	}

	var manifest Manifest
	if err := json.Unmarshal(data, &manifest); err != nil {
		// Corrupt file, create new
		m.manifest = &Manifest{
			CreatedAt: time.Now().Format(time.RFC3339),
			Entries:   make(map[string]*DownloadEntry),
		}
		return nil
	}
	if manifest.Entries == nil {
		manifest.Entries = make(map[string]*DownloadEntry)
	}
	m.manifest = &manifest
	return nil
}

// Save writes the manifest to disk atomically.
func (m *ManifestManager) Save() error {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.manifest.UpdatedAt = time.Now().Format(time.RFC3339)

	data, err := json.MarshalIndent(m.manifest, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal manifest: %w", err)
	}

	tmpPath := m.path + ".tmp"
	if err := os.WriteFile(tmpPath, data, 0644); err != nil {
		return fmt.Errorf("write manifest: %w", err)
	}

	if err := os.Rename(tmpPath, m.path); err != nil {
		os.Remove(tmpPath)
		return fmt.Errorf("rename manifest: %w", err)
	}

	return nil
}

// InitializeEntries creates entries for all symbol-month combinations
// in the date range. Returns the number of new entries added.
func (m *ManifestManager) InitializeEntries(symbols []string, startDate, endDate string) int {
	m.mu.Lock()
	defer m.mu.Unlock()

	start, _ := time.Parse("2006-01-02", startDate)
	end, _ := time.Parse("2006-01-02", endDate)
	newCount := 0

	for _, symbol := range symbols {
		symbolStart := start
		if sd, ok := symbolStartDates[symbol]; ok {
			ipoDate, _ := time.Parse("2006-01-02", sd)
			if ipoDate.After(start) {
				symbolStart = ipoDate
			}
		}

		// Start from first of month
		current := time.Date(symbolStart.Year(), symbolStart.Month(), 1, 0, 0, 0, 0, time.UTC)
		for !current.After(end) {
			key := entryKey(symbol, current.Year(), int(current.Month()))
			if _, exists := m.manifest.Entries[key]; !exists {
				m.manifest.Entries[key] = &DownloadEntry{
					Symbol: symbol,
					Year:   current.Year(),
					Month:  int(current.Month()),
					State:  "pending",
				}
				newCount++
			}
			current = current.AddDate(0, 1, 0)
		}
	}

	return newCount
}

// GetPending returns entries that need processing.
func (m *ManifestManager) GetPending(includeFailed bool) []*DownloadEntry {
	m.mu.Lock()
	defer m.mu.Unlock()

	var result []*DownloadEntry
	for _, entry := range m.manifest.Entries {
		switch entry.State {
		case "pending", "in_progress":
			result = append(result, entry)
		case "failed":
			if includeFailed {
				result = append(result, entry)
			}
		}
	}
	return result
}

// UpdateEntry updates an entry's state and saves.
func (m *ManifestManager) UpdateEntry(key, state string, rowsCount int, errMsg *string) {
	m.mu.Lock()
	defer m.mu.Unlock()

	entry, ok := m.manifest.Entries[key]
	if !ok {
		return
	}
	entry.State = state
	entry.RowsCount = rowsCount
	entry.ErrorMessage = errMsg
	if state == "complete" || state == "failed" {
		now := time.Now().Format(time.RFC3339)
		entry.CompletedAt = &now
	}
}

// GetStats returns a count of entries by state.
func (m *ManifestManager) GetStats() map[string]int {
	m.mu.Lock()
	defer m.mu.Unlock()

	stats := make(map[string]int)
	for _, entry := range m.manifest.Entries {
		stats[entry.State]++
	}
	return stats
}

// entryKey returns the manifest key for a symbol-year-month combination.
func entryKey(symbol string, year, month int) string {
	return fmt.Sprintf("%s_%d_%02d", symbol, year, month)
}
