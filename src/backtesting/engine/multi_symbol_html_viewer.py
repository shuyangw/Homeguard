"""
HTML viewer generator for multi-symbol portfolio metrics and charts.

Creates a standalone HTML file that visualizes portfolio metrics and charts.
"""

import json
from pathlib import Path
from typing import Dict, Any


class MultiSymbolHTMLViewer:
    """Generate standalone HTML viewer for multi-symbol portfolio analytics."""

    @staticmethod
    def generate_html(
        metrics: Dict[str, Any],
        charts: Dict[str, Any],
        output_path: Path,
        title: str = "Multi-Symbol Portfolio Analysis"
    ) -> None:
        """
        Generate standalone HTML file with embedded metrics and chart visualizations.

        Args:
            metrics: Portfolio metrics dictionary
            charts: Chart data dictionary
            output_path: Path to save HTML file
            title: Page title
        """
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        :root {{
            --bg-primary: #1a1a1a;
            --bg-secondary: #2d2d2d;
            --bg-tertiary: #3a3a3a;
            --text-primary: #e0e0e0;
            --text-secondary: #b0b0b0;
            --accent-blue: #4a9eff;
            --accent-green: #4caf50;
            --accent-red: #f44336;
            --accent-yellow: #ffc107;
            --border-color: #444;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            padding: 20px;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}

        header {{
            background: var(--bg-secondary);
            padding: 30px;
            border-radius: 8px;
            margin-bottom: 30px;
            border-left: 4px solid var(--accent-blue);
        }}

        h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            color: var(--accent-blue);
        }}

        h2 {{
            font-size: 1.8em;
            margin: 30px 0 20px 0;
            padding-bottom: 10px;
            border-bottom: 2px solid var(--border-color);
            color: var(--text-primary);
        }}

        h3 {{
            font-size: 1.3em;
            margin: 20px 0 15px 0;
            color: var(--text-secondary);
        }}

        .nav {{
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
            margin-bottom: 30px;
        }}

        .nav-button {{
            background: var(--bg-secondary);
            color: var(--text-primary);
            border: 1px solid var(--border-color);
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            transition: all 0.3s ease;
            text-decoration: none;
            display: inline-block;
        }}

        .nav-button:hover {{
            background: var(--accent-blue);
            border-color: var(--accent-blue);
        }}

        .section {{
            background: var(--bg-secondary);
            padding: 25px;
            border-radius: 8px;
            margin-bottom: 30px;
        }}

        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}

        .metric-card {{
            background: var(--bg-tertiary);
            padding: 20px;
            border-radius: 6px;
            border-left: 3px solid var(--accent-blue);
        }}

        .metric-card.positive {{
            border-left-color: var(--accent-green);
        }}

        .metric-card.negative {{
            border-left-color: var(--accent-red);
        }}

        .metric-label {{
            font-size: 0.9em;
            color: var(--text-secondary);
            margin-bottom: 5px;
        }}

        .metric-value {{
            font-size: 1.8em;
            font-weight: bold;
            color: var(--text-primary);
        }}

        .metric-value.positive {{
            color: var(--accent-green);
        }}

        .metric-value.negative {{
            color: var(--accent-red);
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: var(--bg-tertiary);
            border-radius: 6px;
            overflow: hidden;
        }}

        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }}

        th {{
            background: var(--bg-secondary);
            color: var(--accent-blue);
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.85em;
        }}

        tr:hover {{
            background: rgba(74, 158, 255, 0.1);
        }}

        .positive-cell {{
            color: var(--accent-green);
            font-weight: 600;
        }}

        .negative-cell {{
            color: var(--accent-red);
            font-weight: 600;
        }}

        .chart-container {{
            position: relative;
            height: 400px;
            margin: 20px 0;
            background: var(--bg-tertiary);
            padding: 20px;
            border-radius: 6px;
        }}

        .chart-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}

        .chart-wrapper {{
            background: var(--bg-tertiary);
            padding: 20px;
            border-radius: 6px;
        }}

        .chart-wrapper h3 {{
            margin-top: 0;
            color: var(--text-primary);
        }}

        .chart-canvas {{
            position: relative;
            height: 350px;
        }}

        .summary-stats {{
            display: flex;
            justify-content: space-around;
            flex-wrap: wrap;
            gap: 20px;
            padding: 20px;
            background: var(--bg-tertiary);
            border-radius: 6px;
            margin-bottom: 20px;
        }}

        .summary-stat {{
            text-align: center;
            min-width: 150px;
        }}

        .summary-stat-label {{
            font-size: 0.85em;
            color: var(--text-secondary);
            margin-bottom: 5px;
        }}

        .summary-stat-value {{
            font-size: 2em;
            font-weight: bold;
            color: var(--accent-blue);
        }}

        .correlation-matrix {{
            overflow-x: auto;
        }}

        .heatmap-cell {{
            padding: 10px;
            text-align: center;
            font-weight: 600;
        }}

        @media (max-width: 768px) {{
            .chart-grid {{
                grid-template-columns: 1fr;
            }}

            .metrics-grid {{
                grid-template-columns: 1fr;
            }}
        }}

        .timestamp {{
            color: var(--text-secondary);
            font-size: 0.9em;
        }}

        .footer {{
            text-align: center;
            padding: 20px;
            color: var(--text-secondary);
            margin-top: 40px;
            border-top: 1px solid var(--border-color);
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>{title}</h1>
            <p class="timestamp">Generated: <span id="generated-time"></span></p>
        </header>

        <nav class="nav">
            <a href="#overview" class="nav-button">Overview</a>
            <a href="#composition" class="nav-button">Composition</a>
            <a href="#attribution" class="nav-button">Attribution</a>
            <a href="#charts" class="nav-button">Charts</a>
            <a href="#diversification" class="nav-button">Diversification</a>
        </nav>

        <section id="overview" class="section">
            <h2>Portfolio Overview</h2>
            <div id="overview-stats" class="summary-stats"></div>
        </section>

        <section id="composition" class="section">
            <h2>Portfolio Composition</h2>
            <div id="composition-metrics" class="metrics-grid"></div>
        </section>

        <section id="attribution" class="section">
            <h2>Symbol Attribution</h2>
            <div id="attribution-summary" class="metrics-grid"></div>
            <h3>Per-Symbol Performance</h3>
            <div id="attribution-table"></div>
        </section>

        <section id="charts" class="section">
            <h2>Performance Visualizations</h2>

            <div class="chart-container">
                <canvas id="equityChart"></canvas>
            </div>

            <div class="chart-grid">
                <div class="chart-wrapper">
                    <h3>Portfolio Composition</h3>
                    <div class="chart-canvas">
                        <canvas id="compositionChart"></canvas>
                    </div>
                </div>

                <div class="chart-wrapper">
                    <h3>P&L Contribution</h3>
                    <div class="chart-canvas">
                        <canvas id="pnlPieChart"></canvas>
                    </div>
                </div>

                <div class="chart-wrapper">
                    <h3>Drawdown Timeline</h3>
                    <div class="chart-canvas">
                        <canvas id="drawdownChart"></canvas>
                    </div>
                </div>

                <div class="chart-wrapper">
                    <h3>Position Count</h3>
                    <div class="chart-canvas">
                        <canvas id="positionCountChart"></canvas>
                    </div>
                </div>

                <div class="chart-wrapper">
                    <h3>Rolling Sharpe Ratio</h3>
                    <div class="chart-canvas">
                        <canvas id="rollingSharpeChart"></canvas>
                    </div>
                </div>

                <div class="chart-wrapper">
                    <h3>Monthly Returns</h3>
                    <div class="chart-canvas">
                        <canvas id="monthlyReturnsChart"></canvas>
                    </div>
                </div>
            </div>
        </section>

        <section id="diversification" class="section">
            <h2>Diversification Analysis</h2>
            <div id="diversification-metrics" class="metrics-grid"></div>
            <h3>Symbol Correlation Matrix</h3>
            <div id="correlation-table" class="correlation-matrix"></div>
        </section>

        <footer class="footer">
            <p>Generated with Claude Code - Multi-Symbol Portfolio Analytics</p>
        </footer>
    </div>

    <script>
        // Embedded data
        const metricsData = {json.dumps(metrics, indent=2)};
        const chartsData = {json.dumps(charts, indent=2)};

        // Set timestamp
        document.getElementById('generated-time').textContent = new Date().toLocaleString();

        // Render overview stats
        function renderOverviewStats() {{
            const overview = document.getElementById('overview-stats');
            const composition = metricsData.composition || {{}};
            const attribution = metricsData.attribution || {{}};
            const diversification = metricsData.diversification || {{}};

            const stats = [
                {{label: 'Total P&L', value: formatCurrency(attribution.total_pnl || 0), positive: (attribution.total_pnl || 0) > 0}},
                {{label: 'Best Symbol', value: attribution.best_symbol || 'N/A', positive: true}},
                {{label: 'Worst Symbol', value: attribution.worst_symbol || 'N/A', positive: false}},
                {{label: 'Avg Position Count', value: (composition['Avg Position Count'] || 0).toFixed(2), positive: true}},
                {{label: 'Avg Correlation', value: (diversification['Average Correlation'] || 0).toFixed(3), positive: (diversification['Average Correlation'] || 0) < 0.7}}
            ];

            overview.innerHTML = stats.map(stat => `
                <div class="summary-stat">
                    <div class="summary-stat-label">${{stat.label}}</div>
                    <div class="summary-stat-value ${{stat.positive ? 'positive' : 'negative'}}">${{stat.value}}</div>
                </div>
            `).join('');
        }}

        // Render composition metrics
        function renderCompositionMetrics() {{
            const container = document.getElementById('composition-metrics');
            const composition = metricsData.composition || {{}};

            const metrics = Object.entries(composition).map(([key, value]) => {{
                const isPercentage = key.includes('[%]') || key.includes('Utilization');
                const formattedValue = isPercentage ? value.toFixed(2) + '%' : value.toFixed(2);

                return `
                    <div class="metric-card">
                        <div class="metric-label">${{key}}</div>
                        <div class="metric-value">${{formattedValue}}</div>
                    </div>
                `;
            }});

            container.innerHTML = metrics.join('');
        }}

        // Render attribution summary
        function renderAttributionSummary() {{
            const container = document.getElementById('attribution-summary');
            const attribution = metricsData.attribution || {{}};
            const rebalancing = metricsData.rebalancing || {{}};

            const metrics = [
                {{label: 'Total P&L', value: formatCurrency(attribution.total_pnl || 0), positive: (attribution.total_pnl || 0) > 0}},
                {{label: 'Rebalancing Events', value: rebalancing['Rebalancing Event Count'] || 0, positive: true}},
                {{label: 'Position Turnover', value: (rebalancing['Position Turnover [trades/month]'] || 0).toFixed(2) + ' trades/mo', positive: true}}
            ];

            container.innerHTML = metrics.map(m => `
                <div class="metric-card ${{m.positive ? 'positive' : 'negative'}}">
                    <div class="metric-label">${{m.label}}</div>
                    <div class="metric-value ${{m.positive ? 'positive' : 'negative'}}">${{m.value}}</div>
                </div>
            `).join('');
        }}

        // Render attribution table
        function renderAttributionTable() {{
            const container = document.getElementById('attribution-table');
            const perSymbol = metricsData.attribution?.per_symbol || {{}};

            if (Object.keys(perSymbol).length === 0) {{
                container.innerHTML = '<p>No attribution data available</p>';
                return;
            }}

            const headers = ['Symbol', 'Total P&L', 'Contribution %', 'Total Trades', 'Win Rate %', 'Sharpe Ratio', 'Avg Hold Days'];
            const rows = Object.entries(perSymbol).map(([symbol, stats]) => [
                symbol,
                formatCurrency(stats['Total P&L']),
                stats['Contribution [%]'].toFixed(2) + '%',
                stats['Total Trades'],
                stats['Win Rate [%]'].toFixed(2) + '%',
                stats['Sharpe Ratio'].toFixed(3),
                stats['Avg Hold Duration [days]'].toFixed(1)
            ]);

            container.innerHTML = `
                <table>
                    <thead>
                        <tr>${{headers.map(h => `<th>${{h}}</th>`).join('')}}</tr>
                    </thead>
                    <tbody>
                        ${{rows.map(row => `
                            <tr>
                                <td><strong>${{row[0]}}</strong></td>
                                <td class="${{parseFloat(row[1].replace(/[$,]/g, '')) >= 0 ? 'positive-cell' : 'negative-cell'}}">${{row[1]}}</td>
                                <td>${{row[2]}}</td>
                                <td>${{row[3]}}</td>
                                <td>${{row[4]}}</td>
                                <td>${{row[5]}}</td>
                                <td>${{row[6]}}</td>
                            </tr>
                        `).join('')}}
                    </tbody>
                </table>
            `;
        }}

        // Render diversification metrics
        function renderDiversificationMetrics() {{
            const container = document.getElementById('diversification-metrics');
            const diversification = metricsData.diversification || {{}};

            const metrics = Object.entries(diversification)
                .filter(([key]) => key !== 'correlation_matrix')
                .map(([key, value]) => `
                    <div class="metric-card">
                        <div class="metric-label">${{key}}</div>
                        <div class="metric-value">${{typeof value === 'number' ? value.toFixed(3) : value}}</div>
                    </div>
                `);

            container.innerHTML = metrics.join('');
        }}

        // Render charts
        function renderCharts() {{
            Chart.defaults.color = '#e0e0e0';
            Chart.defaults.borderColor = '#444';

            // Per-symbol equity chart
            if (chartsData.per_symbol_equity?.labels) {{
                new Chart(document.getElementById('equityChart'), {{
                    type: 'line',
                    data: chartsData.per_symbol_equity,
                    options: {{
                        ...chartsData.per_symbol_equity.options,
                        maintainAspectRatio: false
                    }}
                }});
            }}

            // Portfolio composition
            if (chartsData.portfolio_composition?.labels) {{
                new Chart(document.getElementById('compositionChart'), {{
                    type: 'line',
                    data: chartsData.portfolio_composition,
                    options: {{
                        ...chartsData.portfolio_composition.options,
                        maintainAspectRatio: false
                    }}
                }});
            }}

            // P&L contribution pie
            if (chartsData.pnl_contribution_pie?.labels) {{
                new Chart(document.getElementById('pnlPieChart'), {{
                    type: 'pie',
                    data: chartsData.pnl_contribution_pie,
                    options: {{
                        ...chartsData.pnl_contribution_pie.options,
                        maintainAspectRatio: false
                    }}
                }});
            }}

            // Drawdown timeline
            if (chartsData.drawdown_timeline?.labels) {{
                new Chart(document.getElementById('drawdownChart'), {{
                    type: 'line',
                    data: chartsData.drawdown_timeline,
                    options: {{
                        ...chartsData.drawdown_timeline.options,
                        maintainAspectRatio: false
                    }}
                }});
            }}

            // Position count
            if (chartsData.position_count_timeline?.labels) {{
                new Chart(document.getElementById('positionCountChart'), {{
                    type: 'line',
                    data: chartsData.position_count_timeline,
                    options: {{
                        ...chartsData.position_count_timeline.options,
                        maintainAspectRatio: false
                    }}
                }});
            }}

            // Rolling Sharpe
            if (chartsData.rolling_sharpe?.labels) {{
                new Chart(document.getElementById('rollingSharpeChart'), {{
                    type: 'line',
                    data: chartsData.rolling_sharpe,
                    options: {{
                        ...chartsData.rolling_sharpe.options,
                        maintainAspectRatio: false
                    }}
                }});
            }}

            // Monthly returns bar chart
            const monthlyData = chartsData.monthly_returns_heatmap?.data || [];
            if (monthlyData.length > 0) {{
                // Format data for bar chart
                const labels = monthlyData.map(d => `${{d.month}} ${{d.year}}`);
                const values = monthlyData.map(d => d.value);

                // Color bars based on positive/negative
                const backgroundColors = values.map(v => v >= 0 ? 'rgba(76, 175, 80, 0.7)' : 'rgba(244, 67, 54, 0.7)');
                const borderColors = values.map(v => v >= 0 ? 'rgba(76, 175, 80, 1)' : 'rgba(244, 67, 54, 1)');

                new Chart(document.getElementById('monthlyReturnsChart'), {{
                    type: 'bar',
                    data: {{
                        labels: labels,
                        datasets: [{{
                            label: 'Monthly Return %',
                            data: values,
                            backgroundColor: backgroundColors,
                            borderColor: borderColors,
                            borderWidth: 1
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {{
                            y: {{
                                beginAtZero: true,
                                grid: {{
                                    color: '#444'
                                }},
                                ticks: {{
                                    callback: function(value) {{
                                        return value.toFixed(1) + '%';
                                    }}
                                }}
                            }},
                            x: {{
                                grid: {{
                                    display: false
                                }}
                            }}
                        }},
                        plugins: {{
                            title: {{
                                display: false
                            }},
                            legend: {{
                                display: false
                            }},
                            tooltip: {{
                                callbacks: {{
                                    label: function(context) {{
                                        return 'Return: ' + context.parsed.y.toFixed(2) + '%';
                                    }}
                                }}
                            }}
                        }}
                    }}
                }});
            }}
        }}

        // Helper functions
        function formatCurrency(value) {{
            return new Intl.NumberFormat('en-US', {{
                style: 'currency',
                currency: 'USD',
                minimumFractionDigits: 2,
                maximumFractionDigits: 2
            }}).format(value);
        }}

        // Initialize
        document.addEventListener('DOMContentLoaded', () => {{
            renderOverviewStats();
            renderCompositionMetrics();
            renderAttributionSummary();
            renderAttributionTable();
            renderDiversificationMetrics();
            renderCharts();
        }});
    </script>
</body>
</html>"""

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
