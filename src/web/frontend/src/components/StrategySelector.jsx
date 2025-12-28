import React, { useEffect, useState } from 'react';
import { Briefcase, FlaskConical, Loader2, ChevronDown } from 'lucide-react';

const StrategySelector = ({ value, onChange }) => {
  const [strategies, setStrategies] = useState({ Production: [], Research: [] });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch('http://localhost:8000/api/strategies')
      .then(res => {
        if (!res.ok) {
          throw new Error(`HTTP ${res.status}: ${res.statusText}`);
        }
        return res.json();
      })
      .then(data => {
        console.log("Loaded strategies:", data);
        setStrategies(data);
        setLoading(false);
      })
      .catch(err => {
        console.error("Failed to load strategies:", err);
        setError(`Failed to load strategies: ${err.message}`);
        setLoading(false);
      });
  }, []);

  if (loading) {
    return (
      <div className="flex items-center gap-3 py-4 text-[var(--text-tertiary)]">
        <Loader2 size={16} className="animate-spin" />
        <span className="text-sm">Loading strategies...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="py-4 text-[var(--accent-rose)] text-sm">
        {error}
      </div>
    );
  }

  const selectedStrategy = [...(strategies.Production || []), ...(strategies.Research || [])]
    .find(s => s.class_name === value);

  const isProduction = strategies.Production?.some(s => s.class_name === value);

  return (
    <div className="space-y-3">
      <div className="form-select-wrapper">
        <select
          className="form-select"
          value={value}
          onChange={(e) => onChange(e.target.value)}
        >
          <option value="">Select a strategy...</option>

          {strategies.Production && strategies.Production.length > 0 && (
            <optgroup label="Production">
              {strategies.Production.map(s => (
                <option key={s.class_name} value={s.class_name}>
                  {s.class_name}
                </option>
              ))}
            </optgroup>
          )}

          {strategies.Research && strategies.Research.length > 0 && (
            <optgroup label="Research">
              {strategies.Research.map(s => (
                <option key={s.class_name} value={s.class_name}>
                  {s.class_name}
                </option>
              ))}
            </optgroup>
          )}
        </select>
        <ChevronDown size={16} className="select-arrow" />
      </div>

      {/* Selected strategy indicator */}
      {value && (
        <div className="flex items-center justify-between mt-3 px-4 py-3 bg-[var(--bg-primary)] rounded-lg border border-[var(--glass-border)]">
          <div>
            <div className="text-sm font-medium text-[var(--text-primary)] font-mono">
              {value}
            </div>
            <div className="text-[11px] text-[var(--text-tertiary)]">
              {isProduction ? 'Production Strategy' : 'Research Strategy'}
            </div>
          </div>
          <span className={`
            px-2.5 py-1 text-[10px] font-semibold rounded
            ${isProduction
              ? 'bg-[var(--accent-emerald)]/15 text-[var(--accent-emerald)]'
              : 'bg-[var(--accent-violet)]/15 text-[var(--accent-violet)]'}
          `}>
            {isProduction ? 'PROD' : 'DEV'}
          </span>
        </div>
      )}
    </div>
  );
};

export default StrategySelector;
