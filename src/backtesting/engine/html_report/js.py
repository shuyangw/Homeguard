"""
JavaScript Templates for HTML Report Generation.

This module contains the JavaScript code used in backtest HTML reports.
Includes theme toggle, chart configurations, and interactive functions.
"""

import json
from typing import Dict, List, Any, Optional


# Base JavaScript template - theme toggle and utility functions
JS_BASE_TEMPLATE = """
        // Dark mode toggle
        function toggleTheme() {{
            const html = document.documentElement;
            const currentTheme = html.getAttribute('data-theme');
            const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
            html.setAttribute('data-theme', newTheme);
            localStorage.setItem('theme', newTheme);

            const icon = document.querySelector('.theme-toggle i');
            icon.className = newTheme === 'dark' ? 'fas fa-sun' : 'fas fa-moon';

            updateChartsTheme(newTheme);
        }}

        // Load theme from localStorage
        const savedTheme = localStorage.getItem('theme') || 'light';
        document.documentElement.setAttribute('data-theme', savedTheme);
        if (savedTheme === 'dark') {{
            document.querySelector('.theme-toggle i').className = 'fas fa-sun';
        }}

        // Get theme colors
        function getThemeColors() {{
            const theme = document.documentElement.getAttribute('data-theme');
            const isDark = theme === 'dark';

            return {{
                textColor: isDark ? '#e0e0e0' : '#212529',
                gridColor: isDark ? '#404040' : '#dee2e6',
                success: '#10b981',
                danger: '#ef4444',
                warning: '#f59e0b',
                info: '#3b82f6',
                primary: '#6366f1'
            }};
        }}

        // Chart options
        function getChartOptions(title) {{
            const colors = getThemeColors();
            return {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{
                        display: false
                    }},
                    title: {{
                        display: false
                    }}
                }},
                scales: {{
                    y: {{
                        ticks: {{
                            color: colors.textColor
                        }},
                        grid: {{
                            color: colors.gridColor
                        }}
                    }},
                    x: {{
                        ticks: {{
                            color: colors.textColor
                        }},
                        grid: {{
                            color: colors.gridColor
                        }}
                    }}
                }}
            }};
        }}

        // Utility function to create histogram bins
        function createHistogramBins(data, numBins) {{
            const min = Math.min(...data);
            const max = Math.max(...data);
            const binSize = (max - min) / numBins;
            const bins = new Array(numBins).fill(0);
            const labels = [];

            for (let i = 0; i < numBins; i++) {{
                const binStart = min + i * binSize;
                const binEnd = min + (i + 1) * binSize;
                labels.push(`${{binStart.toFixed(1)}} to ${{binEnd.toFixed(1)}}`);
            }}

            data.forEach(val => {{
                let binIndex = Math.floor((val - min) / binSize);
                if (binIndex >= numBins) binIndex = numBins - 1;
                if (binIndex < 0) binIndex = 0;
                bins[binIndex]++;
            }});

            return {{ labels, data: bins }};
        }}

        // Download CSV function
        function downloadCSV() {{
            const blob = new Blob([csvData], {{ type: 'text/csv' }});
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'backtest_results.csv';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);
        }}

        // Toggle row expansion
        function toggleRow(rowId) {{
            const row = document.getElementById(rowId);
            const icon = document.getElementById('icon_' + rowId);
            if (row.classList.contains('show')) {{
                row.classList.remove('show');
                icon.classList.remove('rotated');
            }} else {{
                row.classList.add('show');
                icon.classList.add('rotated');
            }}
        }}
"""


# Chart data template - requires format with data variables
JS_CHART_DATA_TEMPLATE = """
        // Chart data
        const symbols = {symbols_list};
        const returns = {returns_list};
        const sharpe = {sharpe_list};
        const drawdown = {drawdown_list};
        const winRates = {win_rate_list};

        // CSV data for download
        const csvData = `{csv_data}`;
"""


# Returns chart template
JS_RETURNS_CHART = """
        // Returns Chart
        const returnsCtx = document.getElementById('returnsChart').getContext('2d');
        const returnsChart = new Chart(returnsCtx, {{
            type: 'bar',
            data: {{
                labels: symbols,
                datasets: [{{
                    label: 'Total Return (%)',
                    data: returns,
                    backgroundColor: returns.map(r => r >= 0 ? 'rgba(16, 185, 129, 0.6)' : 'rgba(239, 68, 68, 0.6)'),
                    borderColor: returns.map(r => r >= 0 ? '#10b981' : '#ef4444'),
                    borderWidth: 2
                }}]
            }},
            options: getChartOptions('Returns by Symbol')
        }});
"""


# Sharpe chart template
JS_SHARPE_CHART = """
        // Sharpe Chart
        const sharpeCtx = document.getElementById('sharpeChart').getContext('2d');
        const sharpeChart = new Chart(sharpeCtx, {{
            type: 'bar',
            data: {{
                labels: symbols,
                datasets: [{{
                    label: 'Sharpe Ratio',
                    data: sharpe,
                    backgroundColor: sharpe.map(s => s >= 1.0 ? 'rgba(16, 185, 129, 0.6)' : s >= 0.5 ? 'rgba(245, 158, 11, 0.6)' : 'rgba(239, 68, 68, 0.6)'),
                    borderColor: sharpe.map(s => s >= 1.0 ? '#10b981' : s >= 0.5 ? '#f59e0b' : '#ef4444'),
                    borderWidth: 2
                }}]
            }},
            options: getChartOptions('Sharpe Ratio by Symbol')
        }});
"""


# Drawdown chart template
JS_DRAWDOWN_CHART = """
        // Drawdown Chart
        const drawdownCtx = document.getElementById('drawdownChart').getContext('2d');
        const drawdownChart = new Chart(drawdownCtx, {{
            type: 'bar',
            data: {{
                labels: symbols,
                datasets: [{{
                    label: 'Max Drawdown (%)',
                    data: drawdown,
                    backgroundColor: drawdown.map(d => d > -10 ? 'rgba(16, 185, 129, 0.6)' : d > -20 ? 'rgba(245, 158, 11, 0.6)' : 'rgba(239, 68, 68, 0.6)'),
                    borderColor: drawdown.map(d => d > -10 ? '#10b981' : d > -20 ? '#f59e0b' : '#ef4444'),
                    borderWidth: 2
                }}]
            }},
            options: getChartOptions('Drawdown Distribution')
        }});
"""


# Radar chart template - requires format with summary values
JS_RADAR_CHART_TEMPLATE = """
        // Radar Chart for overall metrics
        const radarCtx = document.getElementById('metricsRadar').getContext('2d');
        const radarChart = new Chart(radarCtx, {{
            type: 'radar',
            data: {{
                labels: ['Win Rate', 'Median Return', 'Median Sharpe', 'Low Drawdown', 'Consistency'],
                datasets: [{{
                    label: 'Strategy Performance',
                    data: [
                        {win_rate_symbols},
                        Math.max(0, {total_return_median} * 2),
                        Math.max(0, {sharpe_median} * 20),
                        Math.max(0, 100 + {max_drawdown_mean}),
                        {consistency_score} * 10
                    ],
                    backgroundColor: 'rgba(99, 102, 241, 0.2)',
                    borderColor: '#6366f1',
                    borderWidth: 2,
                    pointBackgroundColor: '#6366f1',
                    pointBorderColor: '#fff',
                    pointHoverBackgroundColor: '#fff',
                    pointHoverBorderColor: '#6366f1'
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    r: {{
                        beginAtZero: true,
                        ticks: {{
                            color: getThemeColors().textColor
                        }},
                        grid: {{
                            color: getThemeColors().gridColor
                        }},
                        pointLabels: {{
                            color: getThemeColors().textColor
                        }}
                    }}
                }},
                plugins: {{
                    legend: {{
                        display: false
                    }}
                }}
            }}
        }});
"""


# Returns distribution chart
JS_RETURNS_DIST_CHART = """
        // Returns Distribution Histogram
        const returnsDistCtx = document.getElementById('returnsDistChart').getContext('2d');
        const returnsBins = createHistogramBins(returns, 8);
        const returnsDistChart = new Chart(returnsDistCtx, {{
            type: 'bar',
            data: {{
                labels: returnsBins.labels,
                datasets: [{{
                    label: 'Frequency',
                    data: returnsBins.data,
                    backgroundColor: 'rgba(99, 102, 241, 0.6)',
                    borderColor: '#6366f1',
                    borderWidth: 2
                }}]
            }},
            options: getChartOptions('Returns Distribution')
        }});
"""


# Risk-return scatter chart
JS_RISK_RETURN_CHART = """
        // Risk-Return Scatter Plot
        const riskReturnCtx = document.getElementById('riskReturnScatter').getContext('2d');
        const scatterData = symbols.map((sym, i) => ({{
            x: Math.abs(drawdown[i]),
            y: returns[i],
            label: sym
        }}));
        const riskReturnChart = new Chart(riskReturnCtx, {{
            type: 'scatter',
            data: {{
                datasets: [{{
                    label: 'Symbols',
                    data: scatterData,
                    backgroundColor: returns.map(r => r >= 0 ? 'rgba(16, 185, 129, 0.6)' : 'rgba(239, 68, 68, 0.6)'),
                    borderColor: returns.map(r => r >= 0 ? '#10b981' : '#ef4444'),
                    borderWidth: 2,
                    pointRadius: 8,
                    pointHoverRadius: 10
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{
                        display: false
                    }},
                    tooltip: {{
                        callbacks: {{
                            label: function(context) {{
                                const point = context.raw;
                                return point.label + ': Return=' + point.y.toFixed(2) + '%, Risk=' + point.x.toFixed(2) + '%';
                            }}
                        }}
                    }}
                }},
                scales: {{
                    y: {{
                        title: {{
                            display: true,
                            text: 'Return (%)',
                            color: getThemeColors().textColor
                        }},
                        ticks: {{
                            color: getThemeColors().textColor
                        }},
                        grid: {{
                            color: getThemeColors().gridColor
                        }}
                    }},
                    x: {{
                        title: {{
                            display: true,
                            text: 'Max Drawdown (abs %)',
                            color: getThemeColors().textColor
                        }},
                        ticks: {{
                            color: getThemeColors().textColor
                        }},
                        grid: {{
                            color: getThemeColors().gridColor
                        }}
                    }}
                }}
            }}
        }});
"""


# Win rate chart
JS_WIN_RATE_CHART = """
        // Win Rate Distribution Chart
        const winRateCtx = document.getElementById('winRateChart').getContext('2d');
        const winRateChart = new Chart(winRateCtx, {{
            type: 'bar',
            data: {{
                labels: symbols,
                datasets: [{{
                    label: 'Win Rate (%)',
                    data: winRates,
                    backgroundColor: winRates.map(w => w >= 60 ? 'rgba(16, 185, 129, 0.6)' : w >= 50 ? 'rgba(245, 158, 11, 0.6)' : 'rgba(239, 68, 68, 0.6)'),
                    borderColor: winRates.map(w => w >= 60 ? '#10b981' : w >= 50 ? '#f59e0b' : '#ef4444'),
                    borderWidth: 2
                }}]
            }},
            options: getChartOptions('Win Rate Distribution')
        }});
"""


# Combined equity chart template - requires format with equity_chart_data
JS_EQUITY_CHART_TEMPLATE = """
        // Combined Portfolio Equity Curve Chart
        let combinedEquityChart = null;
        if ({has_equity_data}) {{
            const equityChartData = {equity_chart_data};
            const equityCanvasElement = document.getElementById('combinedEquityChart');

            if (equityCanvasElement) {{
                const equityCtx = equityCanvasElement.getContext('2d');
                combinedEquityChart = new Chart(equityCtx, {{
                    type: 'line',
                    data: equityChartData,
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        interaction: {{
                            mode: 'index',
                            intersect: false
                        }},
                        plugins: {{
                            legend: {{
                                position: 'top',
                                labels: {{
                                    color: getThemeColors().textColor,
                                    padding: 15,
                                    font: {{
                                        size: 12
                                    }}
                                }}
                            }},
                            tooltip: {{
                                mode: 'index',
                                intersect: false,
                                backgroundColor: 'rgba(0, 0, 0, 0.8)',
                                titleColor: '#fff',
                                bodyColor: '#fff',
                                borderColor: '#6366f1',
                                borderWidth: 1,
                                callbacks: {{
                                    label: function(context) {{
                                        let label = context.dataset.label || '';
                                        if (label) {{
                                            label += ': ';
                                        }}
                                        label += '$' + context.parsed.y.toLocaleString('en-US', {{minimumFractionDigits: 2, maximumFractionDigits: 2}});
                                        return label;
                                    }}
                                }}
                            }}
                        }},
                        scales: {{
                            x: {{
                                ticks: {{
                                    color: getThemeColors().textColor,
                                    maxRotation: 45,
                                    minRotation: 45,
                                    font: {{
                                        size: 12
                                    }}
                                }},
                                grid: {{
                                    color: getThemeColors().gridColor,
                                    display: false
                                }}
                            }},
                            y: {{
                                title: {{
                                    display: true,
                                    text: 'Portfolio Value ($)',
                                    color: getThemeColors().textColor,
                                    font: {{
                                        size: 15,
                                        weight: 'bold'
                                    }}
                                }},
                                ticks: {{
                                    color: getThemeColors().textColor,
                                    font: {{
                                        size: 13,
                                        weight: '500'
                                    }},
                                    callback: function(value) {{
                                        return '$' + value.toLocaleString();
                                    }}
                                }},
                                grid: {{
                                    color: getThemeColors().gridColor
                                }}
                            }}
                        }}
                    }}
                }});
            }}
        }}
"""


# Benchmark chart template
JS_BENCHMARK_CHART_TEMPLATE = """
        // Benchmark Comparison Chart
        let benchmarkChart = null;
        const benchmarkOutperformers = {outperformers};
        const benchmarkUnderperformers = {underperformers};

        if ({has_benchmark_data}) {{
            const benchmarkChartData = {benchmark_chart_data};
            const benchmarkCanvasElement = document.getElementById('benchmarkComparisonChart');

            if (benchmarkCanvasElement && benchmarkChartData.datasets) {{
                const benchmarkCtx = benchmarkCanvasElement.getContext('2d');
                benchmarkChart = new Chart(benchmarkCtx, {{
                    type: 'line',
                    data: benchmarkChartData,
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        interaction: {{
                            mode: 'index',
                            intersect: false
                        }},
                        plugins: {{
                            legend: {{
                                position: 'top',
                                labels: {{
                                    color: getThemeColors().textColor,
                                    padding: 12,
                                    font: {{
                                        size: 11
                                    }},
                                    boxWidth: 30,
                                    usePointStyle: false
                                }},
                                onClick: (e, legendItem, legend) => {{
                                    const index = legendItem.datasetIndex;
                                    const chart = legend.chart;
                                    const meta = chart.getDatasetMeta(index);
                                    meta.hidden = !meta.hidden;
                                    chart.update();
                                }}
                            }},
                            tooltip: {{
                                mode: 'index',
                                intersect: false,
                                backgroundColor: 'rgba(0, 0, 0, 0.85)',
                                titleColor: '#fff',
                                bodyColor: '#fff',
                                borderColor: '#6366f1',
                                borderWidth: 1,
                                callbacks: {{
                                    label: function(context) {{
                                        let label = context.dataset.label || '';
                                        if (label) {{
                                            label += ': ';
                                        }}
                                        label += '$' + context.parsed.y.toLocaleString('en-US', {{minimumFractionDigits: 2, maximumFractionDigits: 2}});
                                        return label;
                                    }}
                                }}
                            }}
                        }},
                        scales: {{
                            x: {{
                                ticks: {{
                                    color: getThemeColors().textColor,
                                    maxRotation: 45,
                                    minRotation: 45,
                                    font: {{
                                        size: 11
                                    }}
                                }},
                                grid: {{
                                    color: getThemeColors().gridColor,
                                    display: false
                                }}
                            }},
                            y: {{
                                title: {{
                                    display: true,
                                    text: 'Portfolio Value ($)',
                                    color: getThemeColors().textColor,
                                    font: {{
                                        size: 14,
                                        weight: 'bold'
                                    }}
                                }},
                                ticks: {{
                                    color: getThemeColors().textColor,
                                    font: {{
                                        size: 12
                                    }},
                                    callback: function(value) {{
                                        return '$' + value.toLocaleString();
                                    }}
                                }},
                                grid: {{
                                    color: getThemeColors().gridColor
                                }}
                            }}
                        }}
                    }}
                }});
            }}
        }}
"""


# SPY comparison chart template
JS_SPY_CHART_TEMPLATE = """
        // SPY Comparison Chart
        let spyChart = null;
        if ({has_spy_data}) {{
            const spyChartData = {spy_chart_data};
            const spyCanvasElement = document.getElementById('spyComparisonChart');

            if (spyCanvasElement && spyChartData.datasets) {{
                const spyCtx = spyCanvasElement.getContext('2d');
                spyChart = new Chart(spyCtx, {{
                    type: 'line',
                    data: spyChartData,
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        interaction: {{
                            mode: 'index',
                            intersect: false
                        }},
                        plugins: {{
                            legend: {{
                                position: 'top',
                                labels: {{
                                    color: getThemeColors().textColor,
                                    padding: 15,
                                    font: {{
                                        size: 13
                                    }}
                                }}
                            }},
                            tooltip: {{
                                mode: 'index',
                                intersect: false,
                                backgroundColor: 'rgba(0, 0, 0, 0.85)',
                                titleColor: '#fff',
                                bodyColor: '#fff',
                                borderColor: '#6366f1',
                                borderWidth: 1,
                                callbacks: {{
                                    label: function(context) {{
                                        let label = context.dataset.label || '';
                                        if (label) {{
                                            label += ': ';
                                        }}
                                        label += '$' + context.parsed.y.toLocaleString('en-US', {{minimumFractionDigits: 2, maximumFractionDigits: 2}});
                                        return label;
                                    }}
                                }}
                            }}
                        }},
                        scales: {{
                            x: {{
                                ticks: {{
                                    color: getThemeColors().textColor,
                                    maxRotation: 45,
                                    minRotation: 45
                                }},
                                grid: {{
                                    color: getThemeColors().gridColor,
                                    display: false
                                }}
                            }},
                            y: {{
                                title: {{
                                    display: true,
                                    text: 'Portfolio Value ($)',
                                    color: getThemeColors().textColor,
                                    font: {{
                                        size: 15,
                                        weight: 'bold'
                                    }}
                                }},
                                ticks: {{
                                    color: getThemeColors().textColor,
                                    callback: function(value) {{
                                        return '$' + value.toLocaleString();
                                    }}
                                }},
                                grid: {{
                                    color: getThemeColors().gridColor
                                }}
                            }}
                        }}
                    }}
                }});
            }}
        }}
"""


# Benchmark toggle functions
JS_BENCHMARK_TOGGLES = """
        // Toggle Functions for Benchmark Chart
        function toggleAllSymbols(show) {{
            if (!benchmarkChart) return;
            benchmarkChart.data.datasets.forEach(ds => {{
                ds.hidden = !show;
            }});
            benchmarkChart.update();
        }}

        function toggleBenchmarks() {{
            if (!benchmarkChart) return;
            const showBenchmarks = document.getElementById('showBenchmarks').checked;
            benchmarkChart.data.datasets.forEach(ds => {{
                if (ds.type === 'benchmark') {{
                    ds.hidden = !showBenchmarks;
                }}
            }});
            benchmarkChart.update();
        }}

        function toggleSymbol(symbol) {{
            if (!benchmarkChart) return;
            const checkbox = document.querySelector(`input[data-symbol="${{symbol}}"]`);
            const isChecked = checkbox ? checkbox.checked : false;

            benchmarkChart.data.datasets.forEach(ds => {{
                if (ds.symbol === symbol) {{
                    ds.hidden = !isChecked;
                }}
            }});
            benchmarkChart.update();
        }}

        function showOnlyOutperformers() {{
            if (!benchmarkChart) return;
            benchmarkChart.data.datasets.forEach(ds => {{
                if (ds.symbol) {{
                    ds.hidden = !benchmarkOutperformers.includes(ds.symbol);
                }}
            }});

            // Update checkboxes
            document.querySelectorAll('.symbol-toggle').forEach(checkbox => {{
                const symbol = checkbox.getAttribute('data-symbol');
                checkbox.checked = benchmarkOutperformers.includes(symbol);
            }});

            benchmarkChart.update();
        }}
"""


# Theme update function
JS_THEME_UPDATE = """
        // Update charts when theme changes
        function updateChartsTheme(theme) {{
            const colors = getThemeColors();

            [returnsChart, sharpeChart, drawdownChart, returnsDistChart, winRateChart].forEach(chart => {{
                chart.options.scales.x.ticks.color = colors.textColor;
                chart.options.scales.y.ticks.color = colors.textColor;
                chart.options.scales.x.grid.color = colors.gridColor;
                chart.options.scales.y.grid.color = colors.gridColor;
                chart.update();
            }});

            radarChart.options.scales.r.ticks.color = colors.textColor;
            radarChart.options.scales.r.grid.color = colors.gridColor;
            radarChart.options.scales.r.pointLabels.color = colors.textColor;
            radarChart.update();

            riskReturnChart.options.scales.x.ticks.color = colors.textColor;
            riskReturnChart.options.scales.y.ticks.color = colors.textColor;
            riskReturnChart.options.scales.x.grid.color = colors.gridColor;
            riskReturnChart.options.scales.y.grid.color = colors.gridColor;
            riskReturnChart.options.scales.x.title.color = colors.textColor;
            riskReturnChart.options.scales.y.title.color = colors.textColor;
            riskReturnChart.update();

            // Update combined equity chart if it exists
            if (combinedEquityChart) {{
                combinedEquityChart.options.plugins.legend.labels.color = colors.textColor;
                combinedEquityChart.options.scales.x.ticks.color = colors.textColor;
                combinedEquityChart.options.scales.x.grid.color = colors.gridColor;
                combinedEquityChart.options.scales.y.ticks.color = colors.textColor;
                combinedEquityChart.options.scales.y.grid.color = colors.gridColor;
                combinedEquityChart.options.scales.y.title.color = colors.textColor;
                combinedEquityChart.update();
            }}

            // Update benchmark chart if it exists
            if (benchmarkChart) {{
                benchmarkChart.options.plugins.legend.labels.color = colors.textColor;
                benchmarkChart.options.scales.x.ticks.color = colors.textColor;
                benchmarkChart.options.scales.x.grid.color = colors.gridColor;
                benchmarkChart.options.scales.y.ticks.color = colors.textColor;
                benchmarkChart.options.scales.y.grid.color = colors.gridColor;
                benchmarkChart.options.scales.y.title.color = colors.textColor;
                benchmarkChart.update();
            }}

            // Update SPY chart if it exists
            if (spyChart) {{
                spyChart.options.plugins.legend.labels.color = colors.textColor;
                spyChart.options.scales.x.ticks.color = colors.textColor;
                spyChart.options.scales.x.grid.color = colors.gridColor;
                spyChart.options.scales.y.ticks.color = colors.textColor;
                spyChart.options.scales.y.grid.color = colors.gridColor;
                spyChart.options.scales.y.title.color = colors.textColor;
                spyChart.update();
            }}
        }}

        // Initialize charts with correct theme
        updateChartsTheme(savedTheme);
"""


def get_base_js() -> str:
    """
    Get the base JavaScript template with theme toggle and utility functions.

    Returns:
        JavaScript string ready to be embedded in <script> tags.
        Note: Uses {{ and }} for literal JS braces (for f-string compatibility).
    """
    return JS_BASE_TEMPLATE


def get_chart_js(
    symbols_list: List[str],
    returns_list: List[float],
    sharpe_list: List[float],
    drawdown_list: List[float],
    win_rate_list: List[float],
    csv_data: str,
    summary: Dict[str, Any],
    equity_chart_data: Optional[Dict] = None,
    benchmark_data: Optional[Dict] = None,
    benchmark_chart_data: Optional[Dict] = None,
    spy_chart_data: Optional[Dict] = None
) -> str:
    """
    Generate the complete JavaScript for chart initialization.

    Args:
        symbols_list: List of symbol names
        returns_list: List of return percentages
        sharpe_list: List of Sharpe ratios
        drawdown_list: List of max drawdown percentages
        win_rate_list: List of win rates
        csv_data: CSV string for download functionality
        summary: Summary statistics dictionary
        equity_chart_data: Optional equity curve chart data
        benchmark_data: Optional benchmark comparison data
        benchmark_chart_data: Optional benchmark chart data
        spy_chart_data: Optional SPY comparison chart data

    Returns:
        Complete JavaScript string for all charts
    """
    # Escape backticks in CSV data
    csv_data_escaped = csv_data.replace('`', '\\`')

    # Build chart data section
    chart_data = JS_CHART_DATA_TEMPLATE.format(
        symbols_list=json.dumps(symbols_list),
        returns_list=json.dumps(returns_list),
        sharpe_list=json.dumps(sharpe_list),
        drawdown_list=json.dumps(drawdown_list),
        win_rate_list=json.dumps(win_rate_list),
        csv_data=csv_data_escaped
    )

    # Build radar chart with summary values
    radar_chart = JS_RADAR_CHART_TEMPLATE.format(
        win_rate_symbols=summary.get('Win Rate (Symbols)', 0),
        total_return_median=summary.get('Total Return [%] - Median', 0),
        sharpe_median=summary.get('Sharpe Ratio - Median', 0),
        max_drawdown_mean=summary.get('Max Drawdown [%] - Mean', 0),
        consistency_score=summary.get('Consistency Score', 0)
    )

    # Build equity chart section
    equity_chart = JS_EQUITY_CHART_TEMPLATE.format(
        has_equity_data=str(bool(equity_chart_data)).lower(),
        equity_chart_data=json.dumps(equity_chart_data) if equity_chart_data else '{}'
    )

    # Build benchmark chart section
    benchmark_data = benchmark_data or {}
    benchmark_chart = JS_BENCHMARK_CHART_TEMPLATE.format(
        outperformers=json.dumps(benchmark_data.get('outperformers', [])),
        underperformers=json.dumps(benchmark_data.get('underperformers', [])),
        has_benchmark_data=str(bool(benchmark_chart_data)).lower(),
        benchmark_chart_data=json.dumps(benchmark_chart_data) if benchmark_chart_data else '{}'
    )

    # Build SPY chart section
    spy_chart = JS_SPY_CHART_TEMPLATE.format(
        has_spy_data=str(bool(spy_chart_data)).lower(),
        spy_chart_data=json.dumps(spy_chart_data) if spy_chart_data else '{}'
    )

    # Combine all JavaScript sections
    js_parts = [
        JS_BASE_TEMPLATE,
        chart_data,
        JS_RETURNS_CHART,
        JS_SHARPE_CHART,
        JS_DRAWDOWN_CHART,
        radar_chart,
        JS_RETURNS_DIST_CHART,
        JS_RISK_RETURN_CHART,
        JS_WIN_RATE_CHART,
        equity_chart,
        benchmark_chart,
        spy_chart,
        JS_BENCHMARK_TOGGLES,
        JS_THEME_UPDATE
    ]

    return '\n'.join(js_parts)
