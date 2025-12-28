import React, { useEffect, useState, useRef } from 'react';
import {
  Activity,
  Terminal,
  ExternalLink,
  TrendingUp,
  TrendingDown,
  BarChart3,
  Target,
  Clock,
  CheckCircle2,
  XCircle,
  Loader2,
  Download
} from 'lucide-react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  ReferenceLine,
  Area,
  AreaChart
} from 'recharts';

const ResultsDashboard = ({ runId }) => {
  const [status, setStatus] = useState('initializing');
  const [progress, setProgress] = useState(0);
  const [logs, setLogs] = useState([]);
  const [result, setResult] = useState(null);
  const logsEndRef = useRef(null);
  const wsRef = useRef(null);

  useEffect(() => {
    if (!runId) return;

    const ws = new WebSocket(`ws://localhost:8000/ws/progress/${runId}`);
    wsRef.current = ws;

    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      if (msg.type === 'log') {
        setLogs(prev => [...prev, msg.data]);
      } else if (msg.type === 'progress') {
        setProgress(msg.data * 100);
      } else if (msg.type === 'status') {
        setStatus(msg.data);
        if (msg.result) setResult(msg.result);
        if (msg.error) {
          setLogs(prev => [...prev, { level: 'ERROR', message: msg.error }]);
        }
      }
    };

    ws.onclose = () => {
      console.log("WebSocket Disconnected");
    };

    return () => {
      if (ws.readyState === 1) ws.close();
    };
  }, [runId]);

  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  // Mock equity curve for display until real data is hooked up
  const mockEquity = Array.from({ length: 30 }, (_, i) => ({
    date: `Day ${i + 1}`,
    value: 100000 * (1 + (i * 0.008 + Math.sin(i * 0.5) * 0.015))
  }));

  const metrics = result?.metrics || {};

  const getStatusConfig = () => {
    switch (status) {
      case 'completed':
        return {
          icon: <CheckCircle2 size={18} />,
          color: 'var(--accent-emerald)',
          label: 'Completed',
          bgClass: 'bg-[var(--accent-emerald)]/10'
        };
      case 'failed':
        return {
          icon: <XCircle size={18} />,
          color: 'var(--accent-rose)',
          label: 'Failed',
          bgClass: 'bg-[var(--accent-rose)]/10'
        };
      case 'running':
        return {
          icon: <Loader2 size={18} className="animate-spin" />,
          color: 'var(--accent-cyan)',
          label: 'Running',
          bgClass: 'bg-[var(--accent-cyan)]/10'
        };
      default:
        return {
          icon: <Clock size={18} />,
          color: 'var(--accent-amber)',
          label: 'Initializing',
          bgClass: 'bg-[var(--accent-amber)]/10'
        };
    }
  };

  const statusConfig = getStatusConfig();

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-[var(--text-primary)] tracking-tight">
            Backtest Results
          </h2>
          <p className="text-sm text-[var(--text-tertiary)] mt-1">
            Run ID: <span className="font-mono text-[var(--accent-cyan)]">{runId}</span>
          </p>
        </div>

        {status === 'completed' && result?.report_path && (
          <a
            href={`http://localhost:8000/api/backtest/${runId}/report`}
            target="_blank"
            rel="noopener noreferrer"
            className="btn btn-primary"
          >
            <Download size={16} />
            <span>Full Report</span>
            <ExternalLink size={14} />
          </a>
        )}
      </div>

      {/* Progress Section */}
      <div className="panel">
        <div className="panel-header">
          <h3><Activity size={14} /> Execution Status</h3>
          <div
            className={`flex items-center gap-2 px-3 py-1 rounded-full text-xs font-medium ${statusConfig.bgClass}`}
            style={{ color: statusConfig.color }}
          >
            {statusConfig.icon}
            <span>{statusConfig.label}</span>
          </div>
        </div>
        <div className="panel-body">
          <div className="flex items-center justify-between mb-3">
            <span className="text-sm text-[var(--text-secondary)]">Progress</span>
            <span className="text-sm font-mono font-medium text-[var(--text-primary)]">
              {Math.round(progress)}%
            </span>
          </div>
          <div className="progress-bar">
            <div
              className={`progress-fill ${status === 'failed' ? 'failed' : ''}`}
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>
      </div>

      {/* Results Section (Show only when completed) */}
      {status === 'completed' && result && (
        <div className="fade-in space-y-6">

          {/* KPI Cards */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <MetricCard
              label="Total Return"
              value={metrics['Total Return [%]']}
              suffix="%"
              icon={<TrendingUp size={18} />}
              accentColor="emerald"
              positive={parseFloat(metrics['Total Return [%]']) >= 0}
            />
            <MetricCard
              label="Sharpe Ratio"
              value={metrics['Sharpe Ratio']}
              icon={<BarChart3 size={18} />}
              accentColor="cyan"
            />
            <MetricCard
              label="Max Drawdown"
              value={metrics['Max Drawdown [%]']}
              suffix="%"
              icon={<TrendingDown size={18} />}
              accentColor="rose"
              invertColor
            />
            <MetricCard
              label="Win Rate"
              value={metrics['Win Rate [%]']}
              suffix="%"
              icon={<Target size={18} />}
              accentColor="amber"
            />
          </div>

          {/* Chart */}
          <div className="panel">
            <div className="panel-header">
              <h3><Activity size={14} /> Equity Curve</h3>
              <span className="badge badge-info text-[10px]">
                {mockEquity.length} data points
              </span>
            </div>
            <div className="panel-body p-0">
              <div className="h-80 p-4">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={mockEquity}>
                    <defs>
                      <linearGradient id="equityGradient" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="var(--accent-emerald)" stopOpacity={0.3} />
                        <stop offset="100%" stopColor="var(--accent-emerald)" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid
                      strokeDasharray="3 3"
                      stroke="var(--glass-border)"
                      vertical={false}
                    />
                    <XAxis
                      dataKey="date"
                      axisLine={false}
                      tickLine={false}
                      tick={{ fill: 'var(--text-tertiary)', fontSize: 11 }}
                      dy={10}
                    />
                    <YAxis
                      domain={['auto', 'auto']}
                      axisLine={false}
                      tickLine={false}
                      tick={{ fill: 'var(--text-tertiary)', fontSize: 11 }}
                      tickFormatter={(v) => `$${(v / 1000).toFixed(0)}k`}
                      dx={-10}
                    />
                    <ReferenceLine
                      y={100000}
                      stroke="var(--text-tertiary)"
                      strokeDasharray="5 5"
                      strokeOpacity={0.5}
                    />
                    <Tooltip content={<CustomTooltip />} />
                    <Area
                      type="monotone"
                      dataKey="value"
                      stroke="var(--accent-emerald)"
                      strokeWidth={2}
                      fill="url(#equityGradient)"
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Logs Console */}
      <div className="console">
        <div className="console-header">
          <div className="console-dots">
            <span className="dot dot-red" />
            <span className="dot dot-yellow" />
            <span className="dot dot-green" />
          </div>
          <div className="flex items-center gap-2 text-xs text-[var(--text-tertiary)]">
            <Terminal size={12} />
            <span>Console Output</span>
          </div>
          <span className="text-[10px] font-mono text-[var(--text-tertiary)]">
            {logs.length} lines
          </span>
        </div>
        <div className="console-body">
          {logs.length === 0 && (
            <div className="flex items-center gap-2 text-[var(--text-tertiary)] italic">
              <Loader2 size={12} className="animate-spin" />
              <span>Waiting for logs...</span>
            </div>
          )}
          {logs.map((log, i) => (
            <LogLine key={i} log={log} />
          ))}
          <div ref={logsEndRef} />
        </div>
      </div>
    </div>
  );
};

// Custom Tooltip for Chart
const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload || !payload.length) return null;

  const value = payload[0].value;
  const initialValue = 100000;
  const returnPct = ((value - initialValue) / initialValue * 100).toFixed(2);
  const isPositive = value >= initialValue;

  return (
    <div className="bg-[var(--bg-secondary)] border border-[var(--glass-border)] rounded-lg p-3 shadow-xl">
      <p className="text-xs text-[var(--text-tertiary)] mb-1">{label}</p>
      <p className="text-sm font-mono font-medium text-[var(--text-primary)]">
        ${value.toLocaleString()}
      </p>
      <p className={`text-xs font-mono ${isPositive ? 'text-[var(--accent-emerald)]' : 'text-[var(--accent-rose)]'}`}>
        {isPositive ? '+' : ''}{returnPct}%
      </p>
    </div>
  );
};

// Metric Card Component
const MetricCard = ({ label, value, suffix = '', icon, accentColor, positive = true, invertColor = false }) => {
  const numValue = parseFloat(value) || 0;
  const displayValue = numValue.toFixed(2);

  const colorVar = `var(--accent-${accentColor})`;
  const shouldShowNegative = invertColor ? numValue > 0 : numValue < 0;
  const textColor = shouldShowNegative ? 'var(--accent-rose)' : colorVar;

  return (
    <div className="metric-card group">
      <div className="metric-accent" style={{ backgroundColor: colorVar }} />
      <div className="flex items-start justify-between">
        <div>
          <div className="text-[10px] font-semibold uppercase tracking-wider text-[var(--text-tertiary)] mb-2">
            {label}
          </div>
          <div
            className="text-2xl font-bold font-mono tracking-tight"
            style={{ color: textColor }}
          >
            {displayValue}{suffix}
          </div>
        </div>
        <div
          className="p-2 rounded-lg transition-colors"
          style={{
            backgroundColor: `color-mix(in srgb, ${colorVar} 10%, transparent)`,
            color: colorVar
          }}
        >
          {icon}
        </div>
      </div>
    </div>
  );
};

// Log Line Component
const LogLine = ({ log }) => {
  const getLogStyle = (level) => {
    switch (level) {
      case 'ERROR':
        return { color: 'var(--accent-rose)', prefix: '[-]' };
      case 'SUCCESS':
        return { color: 'var(--accent-emerald)', prefix: '[+]' };
      case 'WARNING':
        return { color: 'var(--accent-amber)', prefix: '[!]' };
      case 'HEADER':
        return { color: 'var(--accent-violet)', prefix: '[*]', bold: true };
      default:
        return { color: 'var(--text-secondary)', prefix: '   ' };
    }
  };

  const style = getLogStyle(log.level);
  const timestamp = log.timestamp?.split('T')[1]?.split('.')[0] || '00:00:00';

  return (
    <div className="console-line group">
      <span className="text-[var(--text-tertiary)] opacity-50 group-hover:opacity-100 transition-opacity">
        {timestamp}
      </span>
      <span style={{ color: style.color }}>{style.prefix}</span>
      <span
        className={`break-all ${style.bold ? 'font-semibold' : ''}`}
        style={{ color: style.color }}
      >
        {log.message}
      </span>
    </div>
  );
};

export default ResultsDashboard;
