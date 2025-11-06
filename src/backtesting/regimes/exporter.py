"""
Regime analysis export module.

Exports regime analysis results to various formats:
- CSV: Performance tables
- HTML: Formatted report with styling
- JSON: Machine-readable data structure
"""

from pathlib import Path
from typing import Dict, Optional
import pandas as pd
import json
from dataclasses import asdict

from backtesting.regimes.analyzer import RegimeAnalysisResults, RegimePerformance, RegimeLabel


class RegimeExporter:
    """Export regime analysis results to various formats."""

    def export_csv(self, results: RegimeAnalysisResults, output_path: Path):
        """
        Export regime performance tables to CSV.

        Creates three CSV files:
        - {base}_trend.csv: Trend regime performance
        - {base}_volatility.csv: Volatility regime performance
        - {base}_drawdown.csv: Drawdown regime performance

        Args:
            results: RegimeAnalysisResults object
            output_path: Output file path (base name)
        """
        base_path = Path(output_path).with_suffix('')

        # Export trend regimes
        if results.trend_performance:
            trend_df = self._performance_dict_to_dataframe(results.trend_performance)
            trend_path = Path(str(base_path) + '_trend.csv')
            trend_df.to_csv(trend_path, index=False)

        # Export volatility regimes
        if results.volatility_performance:
            vol_df = self._performance_dict_to_dataframe(results.volatility_performance)
            vol_path = Path(str(base_path) + '_volatility.csv')
            vol_df.to_csv(vol_path, index=False)

        # Export drawdown regimes
        if results.drawdown_performance:
            dd_df = self._performance_dict_to_dataframe(results.drawdown_performance)
            dd_path = Path(str(base_path) + '_drawdown.csv')
            dd_df.to_csv(dd_path, index=False)

        # Export summary
        summary_df = pd.DataFrame({
            'Metric': ['Overall Sharpe', 'Overall Return %', 'Robustness Score', 'Best Regime', 'Worst Regime'],
            'Value': [
                f"{results.overall_sharpe:.2f}",
                f"{results.overall_return:.2f}",
                f"{results.robustness_score:.1f}",
                results.best_regime,
                results.worst_regime
            ]
        })
        summary_path = Path(str(base_path) + '_summary.csv')
        summary_df.to_csv(summary_path, index=False)

    def export_html(
        self,
        results: RegimeAnalysisResults,
        output_path: Path,
        strategy_name: str = "Strategy",
        symbol: str = "Portfolio"
    ):
        """
        Export formatted HTML report.

        Args:
            results: RegimeAnalysisResults object
            output_path: Output HTML file path
            strategy_name: Strategy name for title
            symbol: Symbol/portfolio name for title
        """
        html_content = self._generate_html(results, strategy_name, symbol)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

    def export_json(self, results: RegimeAnalysisResults, output_path: Path):
        """
        Export to JSON for programmatic access.

        Args:
            results: RegimeAnalysisResults object
            output_path: Output JSON file path
        """
        # Convert RegimeAnalysisResults to dict
        data = {
            'overall_sharpe': results.overall_sharpe,
            'overall_return': results.overall_return,
            'robustness_score': results.robustness_score,
            'best_regime': results.best_regime,
            'worst_regime': results.worst_regime,
            'trend_performance': self._performance_dict_to_dict(results.trend_performance),
            'volatility_performance': self._performance_dict_to_dict(results.volatility_performance),
            'drawdown_performance': self._performance_dict_to_dict(results.drawdown_performance)
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    def _performance_dict_to_dataframe(
        self,
        perf_dict: Dict[RegimeLabel, RegimePerformance]
    ) -> pd.DataFrame:
        """Convert performance dictionary to DataFrame."""
        rows = []
        for regime_label, perf in perf_dict.items():
            rows.append({
                'Regime': regime_label.value if hasattr(regime_label, 'value') else str(regime_label),
                'Sharpe Ratio': round(perf.sharpe_ratio, 2),
                'Total Return %': round(perf.total_return, 2),
                'Max Drawdown %': round(perf.max_drawdown, 2),
                'Win Rate %': round(perf.win_rate, 2),
                'Num Trades': perf.num_trades,
                'Num Periods': perf.num_periods
            })
        return pd.DataFrame(rows)

    def _performance_dict_to_dict(
        self,
        perf_dict: Dict[RegimeLabel, RegimePerformance]
    ) -> Dict[str, Dict]:
        """Convert performance dictionary to plain dict for JSON."""
        result = {}
        for regime_label, perf in perf_dict.items():
            regime_name = regime_label.value if hasattr(regime_label, 'value') else str(regime_label)
            result[regime_name] = {
                'sharpe_ratio': perf.sharpe_ratio,
                'total_return': perf.total_return,
                'max_drawdown': perf.max_drawdown,
                'win_rate': perf.win_rate,
                'num_trades': perf.num_trades,
                'num_periods': perf.num_periods
            }
        return result

    def _generate_html(
        self,
        results: RegimeAnalysisResults,
        strategy_name: str,
        symbol: str
    ) -> str:
        """Generate HTML report content."""
        # Robustness interpretation
        if results.robustness_score >= 70:
            robustness_label = "Excellent"
            robustness_color = "#10b981"  # Green
        elif results.robustness_score >= 50:
            robustness_label = "Good"
            robustness_color = "#3b82f6"  # Blue
        else:
            robustness_label = "Needs Improvement"
            robustness_color = "#ef4444"  # Red

        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Regime Analysis - {strategy_name} - {symbol}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #0f172a;
            color: #e2e8f0;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        h1 {{
            color: #a78bfa;
            border-bottom: 2px solid #6366f1;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #60a5fa;
            margin-top: 30px;
        }}
        .summary-card {{
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
        }}
        .metric {{
            display: inline-block;
            margin: 10px 20px 10px 0;
        }}
        .metric-label {{
            font-size: 0.9em;
            color: #94a3b8;
        }}
        .metric-value {{
            font-size: 1.8em;
            font-weight: bold;
        }}
        .robustness-score {{
            color: {robustness_color};
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: #1e293b;
        }}
        th {{
            background: #334155;
            color: #e2e8f0;
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }}
        td {{
            padding: 10px 12px;
            border-bottom: 1px solid #334155;
        }}
        tr:hover {{
            background: #334155;
        }}
        .positive {{
            color: #10b981;
        }}
        .negative {{
            color: #ef4444;
        }}
        .neutral {{
            color: #94a3b8;
        }}
        .badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.85em;
            font-weight: 600;
        }}
        .badge-best {{
            background: #10b981;
            color: #000;
        }}
        .badge-worst {{
            background: #ef4444;
            color: #fff;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Regime-Based Performance Analysis</h1>
        <p><strong>Strategy:</strong> {strategy_name} | <strong>Symbol:</strong> {symbol}</p>

        <div class="summary-card">
            <h2>Overall Performance</h2>
            <div class="metric">
                <div class="metric-label">Sharpe Ratio</div>
                <div class="metric-value {'positive' if results.overall_sharpe > 0 else 'negative'}">
                    {results.overall_sharpe:.2f}
                </div>
            </div>
            <div class="metric">
                <div class="metric-label">Total Return</div>
                <div class="metric-value {'positive' if results.overall_return > 0 else 'negative'}">
                    {results.overall_return:.1f}%
                </div>
            </div>
            <div class="metric">
                <div class="metric-label">Robustness Score</div>
                <div class="metric-value robustness-score">
                    {results.robustness_score:.0f}/100
                </div>
            </div>
            <p><strong>Robustness:</strong> <span style="color: {robustness_color}">{robustness_label}</span></p>
            <p>
                <span class="badge badge-best">Best: {results.best_regime}</span>
                <span class="badge badge-worst">Worst: {results.worst_regime}</span>
            </p>
        </div>

        <h2>Trend Regime Performance</h2>
        {self._performance_table_html(results.trend_performance, "Trend")}

        <h2>Volatility Regime Performance</h2>
        {self._performance_table_html(results.volatility_performance, "Volatility")}

        <h2>Drawdown Regime Performance</h2>
        {self._performance_table_html(results.drawdown_performance, "Drawdown")}

        <div style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #334155; color: #64748b; font-size: 0.9em;">
            <p>Generated with Homeguard Backtesting Framework | Regime-Based Testing (Level 4)</p>
        </div>
    </div>
</body>
</html>"""
        return html

    def _performance_table_html(
        self,
        perf_dict: Dict[RegimeLabel, RegimePerformance],
        regime_type: str
    ) -> str:
        """Generate HTML table for performance data."""
        if not perf_dict:
            return '<p class="neutral">No data available</p>'

        rows = []
        for regime_label, perf in perf_dict.items():
            regime_name = regime_label.value if hasattr(regime_label, 'value') else str(regime_label)
            sharpe_class = 'positive' if perf.sharpe_ratio > 0 else 'negative'
            return_class = 'positive' if perf.total_return > 0 else 'negative'

            rows.append(f"""
                <tr>
                    <td><strong>{regime_name}</strong></td>
                    <td class="{sharpe_class}">{perf.sharpe_ratio:.2f}</td>
                    <td class="{return_class}">{perf.total_return:.1f}%</td>
                    <td class="negative">{perf.max_drawdown:.1f}%</td>
                    <td>{perf.win_rate:.1f}%</td>
                    <td>{perf.num_trades}</td>
                    <td>{perf.num_periods}</td>
                </tr>
            """)

        table = f"""
        <table>
            <thead>
                <tr>
                    <th>Regime</th>
                    <th>Sharpe</th>
                    <th>Return %</th>
                    <th>Drawdown %</th>
                    <th>Win Rate %</th>
                    <th>Trades</th>
                    <th>Periods</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows)}
            </tbody>
        </table>
        """
        return table
