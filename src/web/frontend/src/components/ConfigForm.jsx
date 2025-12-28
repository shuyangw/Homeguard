import React, { useState, useEffect } from 'react';
import StrategySelector from './StrategySelector';
import SymbolSelector from './SymbolSelector';
import {
  Calendar,
  Shield,
  GitBranch,
  RotateCcw,
  Play,
  History,
  ChevronDown,
  DollarSign,
  Target,
  Layers,
  Clock,
  Save
} from 'lucide-react';

const STORAGE_KEY = 'homeguard_last_backtest_config';

const defaultConfig = {
  strategy_name: '',
  symbols: [],
  start_date: '2024-01-01',
  end_date: '2024-12-31',
  initial_capital: 100000,
  risk_config: {
    use_stop_loss: true,
    stop_loss_pct: 0.02,
    take_profit_pct: 0.0,
    stop_loss_type: 'percentage',
    max_positions: 10,
    max_single_position_pct: 0.25
  },
  walk_forward: {
    enabled: false,
    train_months: 12,
    test_months: 6,
    step_months: 6
  },
  rolling_window: {
    enabled: false,
    window_days: 60,
    step_days: 30
  },
  strategy_params: {}
};

const ConfigForm = ({ onRun, history, isRunning }) => {
  const [formData, setFormData] = useState(() => {
    // Load from localStorage on initial render
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      if (saved) {
        const parsed = JSON.parse(saved);
        console.log('Loaded cached config:', parsed);
        return { ...defaultConfig, ...parsed };
      }
    } catch (e) {
      console.error('Failed to load cached config:', e);
    }
    return defaultConfig;
  });

  const [showHistory, setShowHistory] = useState(false);
  const [lastSaved, setLastSaved] = useState(null);

  // Auto-save to localStorage when config changes
  useEffect(() => {
    // Only save if there's meaningful data
    if (formData.strategy_name || formData.symbols.length > 0) {
      try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(formData));
        setLastSaved(new Date());
      } catch (e) {
        console.error('Failed to save config:', e);
      }
    }
  }, [formData]);

  const loadConfig = (config) => {
    setFormData({
      ...formData,
      strategy_name: config.strategy_name,
      symbols: config.symbols,
      start_date: config.start_date,
      end_date: config.end_date,
      initial_capital: config.initial_capital,
      risk_config: config.risk_config || formData.risk_config,
      walk_forward: config.walk_forward || formData.walk_forward,
      rolling_window: config.rolling_window || formData.rolling_window,
      strategy_params: config.strategy_params
    });
    setShowHistory(false);
  };

  const handleChange = (field, value) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  const handleRiskChange = (field, value) => {
    setFormData(prev => ({
      ...prev,
      risk_config: { ...prev.risk_config, [field]: value }
    }));
  };

  const handleWFChange = (field, value) => {
    setFormData(prev => ({
      ...prev,
      walk_forward: { ...prev.walk_forward, [field]: value },
      rolling_window: value === true && field === 'enabled'
        ? { ...prev.rolling_window, enabled: false }
        : prev.rolling_window
    }));
  };

  const handleRollingChange = (field, value) => {
    setFormData(prev => ({
      ...prev,
      rolling_window: { ...prev.rolling_window, [field]: value },
      walk_forward: value === true && field === 'enabled'
        ? { ...prev.walk_forward, enabled: false }
        : prev.walk_forward
    }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    onRun(formData);
  };

  const isValid = formData.strategy_name && formData.symbols.length > 0;

  return (
    <form onSubmit={handleSubmit}>
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <h2 className="text-2xl font-bold text-[var(--text-primary)] tracking-tight">
            Configure Backtest
          </h2>
          <div className="flex items-center gap-3 mt-1">
            <p className="text-sm text-[var(--text-tertiary)]">
              Set up your strategy parameters and run analysis
            </p>
            {lastSaved && (
              <span className="flex items-center gap-1.5 text-[10px] text-[var(--accent-emerald)]">
                <Save size={10} />
                <span>Auto-saved</span>
              </span>
            )}
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex items-center gap-2">
          {/* Reset Button */}
          {(formData.strategy_name || formData.symbols.length > 0) && (
            <button
              type="button"
              onClick={() => {
                localStorage.removeItem(STORAGE_KEY);
                setFormData(defaultConfig);
                setLastSaved(null);
              }}
              className="btn btn-ghost text-sm text-[var(--text-tertiary)] hover:text-[var(--accent-rose)]"
            >
              <RotateCcw size={14} />
              <span>Reset</span>
            </button>
          )}

          {/* History Dropdown */}
          {history && history.length > 0 && (
            <div className="relative">
              <button
                type="button"
                onClick={() => setShowHistory(!showHistory)}
                className="btn btn-secondary text-sm"
              >
                <History size={16} />
                <span>History</span>
                <ChevronDown size={14} className={`transition-transform ${showHistory ? 'rotate-180' : ''}`} />
              </button>

              {showHistory && (
                <div className="absolute right-0 top-full mt-2 w-72 bg-[var(--bg-secondary)] border border-[var(--glass-border)] rounded-lg shadow-xl z-50 overflow-hidden">
                  <div className="p-3 border-b border-[var(--glass-border)] bg-[var(--bg-primary)]">
                    <span className="text-xs font-semibold text-[var(--text-tertiary)] uppercase tracking-wider">
                      Recent Configurations
                    </span>
                  </div>
                  <div className="max-h-64 overflow-y-auto">
                    {history.map((h, i) => (
                      <button
                        key={i}
                        type="button"
                        className="w-full p-3 text-left hover:bg-[var(--bg-tertiary)] transition-colors border-b border-[var(--glass-border)] last:border-0"
                        onClick={() => loadConfig(h)}
                      >
                        <div className="font-medium text-sm text-[var(--accent-cyan)]">
                          {h.strategy_name}
                        </div>
                        <div className="flex items-center gap-2 mt-1 text-xs text-[var(--text-tertiary)]">
                          <Clock size={10} />
                          <span>{h.saved_at?.split('T')[0]}</span>
                          <span className="text-[var(--text-tertiary)]">|</span>
                          <span>{h.symbols?.length || 0} symbols</span>
                        </div>
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Main Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">

        {/* Left Column - Strategy & Symbols */}
        <div className="lg:col-span-7 space-y-6">

          {/* Strategy Card */}
          <div className="panel">
            <div className="panel-header">
              <h3><Layers size={14} /> Strategy Selection</h3>
            </div>
            <div className="panel-body">
              <StrategySelector
                value={formData.strategy_name}
                onChange={(v) => handleChange('strategy_name', v)}
              />
            </div>
          </div>

          {/* Symbols Card */}
          <div className="panel">
            <div className="panel-header">
              <h3><Target size={14} /> Trading Universe</h3>
              {formData.symbols.length > 0 && (
                <span className="badge badge-info">{formData.symbols.length} symbols</span>
              )}
            </div>
            <div className="panel-body">
              <SymbolSelector
                value={formData.symbols}
                onChange={(v) => handleChange('symbols', v)}
              />
            </div>
          </div>

          {/* Date Range Card */}
          <div className="panel">
            <div className="panel-header">
              <h3><Calendar size={14} /> Backtest Period</h3>
            </div>
            <div className="panel-body">
              <div className="grid grid-cols-2 gap-4">
                <div className="form-group mb-0">
                  <label className="form-label">Start Date</label>
                  <input
                    type="date"
                    className="form-input"
                    value={formData.start_date}
                    onChange={(e) => handleChange('start_date', e.target.value)}
                  />
                </div>
                <div className="form-group mb-0">
                  <label className="form-label">End Date</label>
                  <input
                    type="date"
                    className="form-input"
                    value={formData.end_date}
                    onChange={(e) => handleChange('end_date', e.target.value)}
                  />
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Right Column - Settings */}
        <div className="lg:col-span-5 space-y-6">

          {/* Capital */}
          <div className="panel">
            <div className="panel-header">
              <h3><DollarSign size={14} /> Initial Capital</h3>
            </div>
            <div className="panel-body">
              <div className="relative">
                <span className="absolute left-3 top-1/2 -translate-y-1/2 text-[var(--text-tertiary)]">$</span>
                <input
                  type="number"
                  className="form-input pl-7 text-lg font-mono"
                  value={formData.initial_capital}
                  onChange={(e) => handleChange('initial_capital', parseInt(e.target.value))}
                />
              </div>
            </div>
          </div>

          {/* Risk Management */}
          <div className="panel">
            <div className="panel-header">
              <h3><Shield size={14} /> Risk Management</h3>
            </div>
            <div className="panel-body">
              <div className="grid grid-cols-2 gap-4">
                <PercentField
                  label="Stop Loss"
                  value={formData.risk_config.stop_loss_pct}
                  onChange={(v) => handleRiskChange('stop_loss_pct', v)}
                  hint="Exit if position drops by this %"
                />
                <PercentField
                  label="Take Profit"
                  value={formData.risk_config.take_profit_pct || 0}
                  onChange={(v) => handleRiskChange('take_profit_pct', v)}
                  hint="Exit if position gains this %"
                />
                <InputField
                  label="Max Positions"
                  type="number"
                  value={formData.risk_config.max_positions}
                  onChange={(v) => handleRiskChange('max_positions', parseInt(v))}
                />
                <PercentField
                  label="Max Allocation"
                  value={formData.risk_config.max_single_position_pct}
                  onChange={(v) => handleRiskChange('max_single_position_pct', v)}
                  hint="Max % of capital per position"
                />
              </div>
            </div>
          </div>

          {/* Walk Forward */}
          <TogglePanel
            icon={<GitBranch size={14} />}
            title="Walk-Forward Validation"
            enabled={formData.walk_forward.enabled}
            onToggle={(v) => handleWFChange('enabled', v)}
            accentColor="emerald"
          >
            <div className="grid grid-cols-3 gap-3">
              <InputField
                label="Train (Mo)"
                type="number"
                value={formData.walk_forward.train_months}
                onChange={(v) => handleWFChange('train_months', parseInt(v))}
                compact
              />
              <InputField
                label="Test (Mo)"
                type="number"
                value={formData.walk_forward.test_months}
                onChange={(v) => handleWFChange('test_months', parseInt(v))}
                compact
              />
              <InputField
                label="Step (Mo)"
                type="number"
                value={formData.walk_forward.step_months}
                onChange={(v) => handleWFChange('step_months', parseInt(v))}
                compact
              />
            </div>
          </TogglePanel>

          {/* Rolling Window */}
          <TogglePanel
            icon={<RotateCcw size={14} />}
            title="Rolling Window"
            enabled={formData.rolling_window?.enabled || false}
            onToggle={(v) => handleRollingChange('enabled', v)}
            accentColor="cyan"
          >
            <div className="grid grid-cols-2 gap-3">
              <InputField
                label="Window (Days)"
                type="number"
                value={formData.rolling_window?.window_days || 60}
                onChange={(v) => handleRollingChange('window_days', parseInt(v))}
                compact
              />
              <InputField
                label="Step (Days)"
                type="number"
                value={formData.rolling_window?.step_days || 30}
                onChange={(v) => handleRollingChange('step_days', parseInt(v))}
                compact
              />
            </div>
          </TogglePanel>
        </div>
      </div>

      {/* Submit Button */}
      <div className="mt-8 flex justify-end">
        <button
          type="submit"
          disabled={!isValid || isRunning}
          className={`
            btn btn-primary px-8 py-3 text-base
            ${(!isValid || isRunning) ? 'opacity-50 cursor-not-allowed' : ''}
          `}
        >
          {isRunning ? (
            <>
              <span className="w-4 h-4 border-2 border-[var(--bg-void)] border-t-transparent rounded-full animate-spin" />
              <span>Running...</span>
            </>
          ) : (
            <>
              <Play size={18} />
              <span>Run Backtest</span>
            </>
          )}
        </button>
      </div>
    </form>
  );
};

// Reusable Input Field
const InputField = ({ label, type, value, onChange, step, compact }) => (
  <div className={compact ? '' : 'form-group mb-0'}>
    <label className="form-label text-[10px]">{label}</label>
    <input
      type={type}
      step={step}
      className={`form-input ${compact ? 'form-input-sm' : ''}`}
      value={value}
      onChange={(e) => onChange(e.target.value)}
    />
  </div>
);

// Percentage Field - displays as whole number (25) but stores as decimal (0.25)
const PercentField = ({ label, value, onChange, hint }) => (
  <div className="form-group mb-0">
    <label className="form-label text-[10px]">{label}</label>
    <div className="relative">
      <input
        type="number"
        step="1"
        min="0"
        max="100"
        className="form-input pr-8"
        value={Math.round(value * 100)}
        onChange={(e) => onChange(parseFloat(e.target.value) / 100)}
      />
      <span className="absolute right-3 top-1/2 -translate-y-1/2 text-[var(--text-tertiary)] text-sm">%</span>
    </div>
    {hint && <p className="text-[9px] text-[var(--text-tertiary)] mt-1">{hint}</p>}
  </div>
);

// Toggle Panel Component
const TogglePanel = ({ icon, title, enabled, onToggle, accentColor, children }) => (
  <div
    className={`
      panel transition-all duration-200
      ${enabled ? `border-[var(--accent-${accentColor})]/30` : ''}
    `}
    style={enabled ? { borderColor: `var(--accent-${accentColor})`, borderWidth: '1px' } : {}}
  >
    <div className="panel-header">
      <h3 className={enabled ? `text-[var(--accent-${accentColor})]` : ''}>
        {icon} {title}
      </h3>
      <label className="toggle">
        <input
          type="checkbox"
          checked={enabled}
          onChange={(e) => onToggle(e.target.checked)}
        />
        <div className="toggle-track">
          <div className="toggle-thumb" />
        </div>
      </label>
    </div>
    {enabled && (
      <div className="panel-body fade-in">
        {children}
      </div>
    )}
  </div>
);

export default ConfigForm;
