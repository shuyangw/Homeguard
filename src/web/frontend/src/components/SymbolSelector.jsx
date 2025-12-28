import React, { useEffect, useState } from 'react';
import { List, Keyboard, Loader2, FileText, ChevronDown } from 'lucide-react';

const SymbolSelector = ({ value, onChange }) => {
  const [mode, setMode] = useState('list');
  const [lists, setLists] = useState([]);
  const [selectedList, setSelectedList] = useState('');
  const [previewSymbols, setPreviewSymbols] = useState([]);
  const [loading, setLoading] = useState(false);
  const [manualInput, setManualInput] = useState('');

  useEffect(() => {
    fetch('http://localhost:8000/api/symbols/lists')
      .then(res => {
        if (!res.ok) {
          throw new Error(`HTTP ${res.status}: ${res.statusText}`);
        }
        return res.json();
      })
      .then(data => {
        console.log("Loaded symbol lists:", data);
        setLists(data);
      })
      .catch(err => {
        console.error("Failed to load symbol lists:", err);
      });
  }, []);

  const handleListSelect = async (filename) => {
    if (!filename) {
      setSelectedList('');
      setPreviewSymbols([]);
      onChange([]);
      return;
    }

    setSelectedList(filename);
    setLoading(true);
    try {
      const res = await fetch(`http://localhost:8000/api/symbols/lists/${filename}`);
      const data = await res.json();
      setPreviewSymbols(data);
      onChange(data);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleManualChange = (txt) => {
    setManualInput(txt);
    const syms = txt.split(',').map(s => s.trim().toUpperCase()).filter(s => s);
    onChange(syms);
  };

  return (
    <div className="space-y-4">
      {/* Mode Tabs */}
      <div className="tabs">
        <button
          type="button"
          className={`tab ${mode === 'list' ? 'active' : ''}`}
          onClick={() => setMode('list')}
        >
          <List size={14} />
          <span>From List</span>
        </button>
        <button
          type="button"
          className={`tab ${mode === 'manual' ? 'active' : ''}`}
          onClick={() => setMode('manual')}
        >
          <Keyboard size={14} />
          <span>Manual</span>
        </button>
      </div>

      {mode === 'manual' ? (
        <div className="space-y-2">
          <textarea
            className="form-textarea h-28"
            placeholder="AAPL, MSFT, GOOGL, TSLA..."
            value={manualInput}
            onChange={(e) => handleManualChange(e.target.value)}
          />
          <div className="flex items-center justify-between text-xs text-[var(--text-tertiary)]">
            <span>Separate symbols with commas</span>
            {value.length > 0 && (
              <span className="text-[var(--accent-cyan)]">{value.length} symbols</span>
            )}
          </div>
        </div>
      ) : (
        <div className="space-y-4">
          <div className="form-select-wrapper">
            <select
              className="form-select"
              value={selectedList}
              onChange={(e) => handleListSelect(e.target.value)}
            >
              <option value="">Select a symbol list...</option>
              {lists.map(f => (
                <option key={f} value={f}>{f}</option>
              ))}
            </select>
            <ChevronDown size={16} className="select-arrow" />
          </div>

          {loading && (
            <div className="flex items-center gap-2 text-sm text-[var(--accent-cyan)]">
              <Loader2 size={14} className="animate-spin" />
              <span>Loading symbols...</span>
            </div>
          )}

          {previewSymbols.length > 0 && !loading && (
            <div className="mt-3 bg-[var(--bg-primary)] rounded-lg border border-[var(--glass-border)] overflow-hidden">
              <div className="px-3 py-2 border-b border-[var(--glass-border)] bg-[var(--bg-void)]/30 flex items-center justify-between">
                <div className="flex items-center gap-2 text-[11px] text-[var(--text-tertiary)]">
                  <FileText size={12} />
                  <span className="font-mono">{selectedList}</span>
                </div>
                <span className="px-2 py-0.5 text-[10px] font-semibold bg-[var(--accent-cyan)]/15 text-[var(--accent-cyan)] rounded">
                  {previewSymbols.length} symbols
                </span>
              </div>
              <div className="p-3 max-h-28 overflow-y-auto">
                <div className="flex flex-wrap gap-1">
                  {previewSymbols.slice(0, 40).map(s => (
                    <span key={s} className="px-1.5 py-0.5 text-[10px] font-mono bg-[var(--accent-cyan)]/10 text-[var(--accent-cyan)] rounded border border-[var(--accent-cyan)]/20">
                      {s}
                    </span>
                  ))}
                  {previewSymbols.length > 40 && (
                    <span className="px-2 py-0.5 text-[10px] text-[var(--text-tertiary)]">
                      +{previewSymbols.length - 40} more
                    </span>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default SymbolSelector;
