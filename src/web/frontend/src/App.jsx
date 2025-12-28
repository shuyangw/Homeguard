import React, { useState, useEffect } from 'react';
import ConfigForm from './components/ConfigForm';
import ResultsDashboard from './components/ResultsDashboard';
import ErrorBoundary from './components/ErrorBoundary';
import { Activity, Settings2, TrendingUp, Zap } from 'lucide-react';

function App() {
  const [view, setView] = useState('config');
  const [currentRunId, setCurrentRunId] = useState(null);
  const [history, setHistory] = useState([]);
  const [isRunning, setIsRunning] = useState(false);

  useEffect(() => {
    fetchHistory();
  }, []);

  const fetchHistory = () => {
    fetch('http://localhost:8000/api/backtest/history')
      .then(res => res.json())
      .then(setHistory)
      .catch(console.error);
  };

  const handleRunBacktest = async (config) => {
    try {
      setIsRunning(true);
      const res = await fetch('http://localhost:8000/api/backtest', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config),
      });

      if (!res.ok) throw new Error('Failed to start backtest');

      const data = await res.json();
      setCurrentRunId(data.run_id);
      setView('results');
      fetchHistory();
    } catch (err) {
      alert("Error starting backtest: " + err.message);
    } finally {
      setIsRunning(false);
    }
  };

  return (
    <div className="min-h-screen">
      {/* Header */}
      <header className="border-b border-[var(--glass-border)] bg-[var(--bg-secondary)]/80 sticky top-0 z-50">
        <div className="container py-3">
          <div className="flex items-center justify-between">
            {/* Logo & Title */}
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-[var(--accent-emerald)] to-[var(--accent-cyan)] flex items-center justify-center flex-shrink-0">
                <TrendingUp size={20} className="text-[var(--bg-void)]" />
              </div>
              <div>
                <h1 className="text-xl font-bold tracking-tight text-[var(--text-primary)]">
                  HOMEGUARD
                </h1>
                <div className="flex items-center gap-2 mt-0.5">
                  <span className="text-[10px] font-medium tracking-widest text-[var(--text-tertiary)] uppercase">
                    Quantitative Research
                  </span>
                  <span className="px-1.5 py-0.5 text-[9px] font-semibold bg-[var(--accent-emerald)]/15 text-[var(--accent-emerald)] rounded">
                    LIVE
                  </span>
                </div>
              </div>
            </div>

            {/* Navigation */}
            <nav className="flex items-center gap-1 p-1 bg-[var(--bg-primary)] rounded-lg border border-[var(--glass-border)]">
              <NavButton
                active={view === 'config'}
                onClick={() => setView('config')}
                icon={<Settings2 size={16} />}
                label="Configure"
              />
              <NavButton
                active={view === 'results'}
                onClick={() => setView('results')}
                disabled={!currentRunId}
                icon={<Activity size={16} />}
                label="Results"
                pulse={isRunning}
              />
            </nav>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container py-8">
        <div className="fade-in">
          <ErrorBoundary>
            {view === 'config' ? (
              <ConfigForm onRun={handleRunBacktest} history={history} isRunning={isRunning} />
            ) : (
              <ResultsDashboard runId={currentRunId} />
            )}
          </ErrorBoundary>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-[var(--glass-border)] py-6 mt-auto">
        <div className="container">
          <div className="flex items-center justify-between text-xs text-[var(--text-tertiary)]">
            <span className="font-mono">v1.0.0</span>
            <span>Homeguard Backtest Engine</span>
          </div>
        </div>
      </footer>
    </div>
  );
}

// Navigation Button Component
const NavButton = ({ active, onClick, disabled, icon, label, pulse }) => (
  <button
    onClick={onClick}
    disabled={disabled}
    className={`
      relative flex items-center gap-2 px-4 py-2 rounded-md text-sm font-medium
      transition-all duration-200 ease-out
      ${active
        ? 'bg-[var(--bg-tertiary)] text-[var(--accent-emerald)]'
        : 'text-[var(--text-tertiary)] hover:text-[var(--text-secondary)] hover:bg-[var(--bg-secondary)]'
      }
      ${disabled ? 'opacity-40 cursor-not-allowed' : 'cursor-pointer'}
    `}
  >
    {icon}
    <span>{label}</span>
    {pulse && (
      <span className="absolute top-1.5 right-1.5 w-2 h-2 bg-[var(--accent-emerald)] rounded-full animate-pulse" />
    )}
  </button>
);

export default App;
