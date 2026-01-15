"""
CSS Templates for HTML Report Generation.

This module contains the CSS styles used in backtest HTML reports.
Supports both light and dark themes via CSS variables.
"""

# Main CSS template - supports light/dark themes via CSS variables
CSS_TEMPLATE = """
:root {
    --bg-primary: #ffffff;
    --bg-secondary: #f8f9fa;
    --bg-tertiary: #e9ecef;
    --text-primary: #212529;
    --text-secondary: #6c757d;
    --border-color: #dee2e6;
    --success: #10b981;
    --danger: #ef4444;
    --warning: #f59e0b;
    --info: #3b82f6;
    --primary: #6366f1;
    --card-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
}

[data-theme="dark"] {
    --bg-primary: #1e1e1e;
    --bg-secondary: #2d2d2d;
    --bg-tertiary: #3d3d3d;
    --text-primary: #e0e0e0;
    --text-secondary: #a0a0a0;
    --border-color: #404040;
    --card-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.4);
}

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
    background: var(--bg-secondary);
    color: var(--text-primary);
    line-height: 1.6;
    transition: background-color 0.3s ease, color 0.3s ease;
}

.container {
    max-width: 1400px;
    margin: 0 auto;
    padding: 20px;
}

.header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 30px;
    padding: 20px;
    background: var(--bg-primary);
    border-radius: 12px;
    box-shadow: var(--card-shadow);
}

h1 {
    font-size: 2em;
    font-weight: 700;
    color: var(--text-primary);
    margin: 0;
}

.theme-toggle {
    background: var(--bg-tertiary);
    border: none;
    border-radius: 50px;
    padding: 8px 16px;
    cursor: pointer;
    font-size: 1.2em;
    color: var(--text-primary);
    transition: all 0.3s ease;
}

.theme-toggle:hover {
    transform: scale(1.05);
    background: var(--primary);
    color: white;
}

h2 {
    font-size: 1.5em;
    font-weight: 600;
    color: var(--text-primary);
    margin: 30px 0 20px 0;
    padding-bottom: 10px;
    border-bottom: 2px solid var(--border-color);
}

.metrics-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    gap: 20px;
    margin-bottom: 30px;
}

.metric-card {
    background: var(--bg-primary);
    padding: 20px;
    border-radius: 12px;
    box-shadow: var(--card-shadow);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.metric-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
}

.metric-label {
    font-size: 0.85em;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 6px;
}

.metric-value {
    font-size: 2em;
    font-weight: 700;
    color: var(--text-primary);
}

.metric-value.positive {
    color: var(--success);
}

.metric-value.negative {
    color: var(--danger);
}

.metric-value.warning {
    color: var(--warning);
}

.metric-value.info {
    color: var(--info);
}

.chart-container {
    background: var(--bg-primary);
    padding: 24px;
    border-radius: 12px;
    box-shadow: var(--card-shadow);
    margin-bottom: 30px;
}

.chart-wrapper {
    position: relative;
    height: 400px;
}

.charts-row {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
    gap: 20px;
    margin-bottom: 30px;
}

table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    background: var(--bg-primary);
    box-shadow: var(--card-shadow);
    border-radius: 12px;
    overflow: hidden;
}

thead {
    background: var(--primary);
    color: white;
}

th {
    padding: 16px;
    text-align: left;
    font-weight: 600;
    font-size: 0.9em;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    position: sticky;
    top: 0;
    z-index: 10;
}

td {
    padding: 14px 16px;
    border-bottom: 1px solid var(--border-color);
}

tbody tr {
    transition: background-color 0.2s ease;
}

tbody tr:hover {
    background: var(--bg-secondary);
}

tbody tr:last-child td {
    border-bottom: none;
}

.positive-value {
    color: var(--success);
    font-weight: 600;
}

.negative-value {
    color: var(--danger);
    font-weight: 600;
}

.neutral-value {
    color: var(--text-secondary);
}

.badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 12px;
    font-size: 0.85em;
    font-weight: 600;
}

.badge-success {
    background: rgba(16, 185, 129, 0.1);
    color: var(--success);
}

.badge-danger {
    background: rgba(239, 68, 68, 0.1);
    color: var(--danger);
}

.badge-warning {
    background: rgba(245, 158, 11, 0.1);
    color: var(--warning);
}

.badge-info {
    background: rgba(59, 130, 246, 0.1);
    color: var(--info);
}

.footer {
    text-align: center;
    color: var(--text-secondary);
    font-size: 0.9em;
    margin-top: 40px;
    padding: 20px;
}

/* Executive Summary Styles */
.executive-summary {
    background: linear-gradient(135deg, var(--primary) 0%, var(--info) 100%);
    padding: 30px;
    border-radius: 12px;
    margin-bottom: 30px;
    color: white;
    box-shadow: var(--card-shadow);
}

.summary-badges {
    display: flex;
    gap: 20px;
    margin-top: 20px;
    flex-wrap: wrap;
}

.performance-badge {
    background: white;
    color: var(--text-primary);
    padding: 15px 25px;
    border-radius: 12px;
    font-weight: 600;
    display: flex;
    flex-direction: column;
    gap: 8px;
    flex: 1;
    min-width: 200px;
}

.badge-label {
    font-size: 0.85em;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--text-secondary);
}

.badge-value {
    font-size: 1.5em;
    font-weight: 700;
}

/* Advanced Metrics Table */
.advanced-metrics {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 15px;
    margin: 20px 0;
}

.metric-item {
    background: var(--bg-tertiary);
    padding: 12px;
    border-radius: 8px;
}

.metric-item-label {
    font-size: 0.85em;
    color: var(--text-secondary);
    margin-bottom: 4px;
}

.metric-item-value {
    font-size: 1.2em;
    font-weight: 600;
    color: var(--text-primary);
}

/* Download Buttons */
.download-section {
    display: flex;
    gap: 15px;
    margin: 20px 0;
    flex-wrap: wrap;
}

.download-btn {
    background: var(--primary);
    color: white;
    padding: 12px 24px;
    border-radius: 8px;
    text-decoration: none;
    font-weight: 600;
    display: inline-flex;
    align-items: center;
    gap: 8px;
    transition: all 0.3s ease;
    border: none;
    cursor: pointer;
}

.download-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
}

/* Expandable Sections */
.expandable {
    cursor: pointer;
    user-select: none;
}

.expandable:hover {
    background: var(--bg-tertiary);
}

.expanded-content {
    display: none;
    padding: 20px;
    background: var(--bg-secondary);
}

.expanded-content.show {
    display: block;
}

.expand-icon {
    transition: transform 0.3s ease;
}

.expand-icon.rotated {
    transform: rotate(90deg);
}

h3 {
    color: var(--text-primary);
    margin-bottom: 16px;
}

@media (max-width: 768px) {
    .header {
        flex-direction: column;
        gap: 16px;
    }

    .metrics-grid {
        grid-template-columns: 1fr;
    }

    .charts-row {
        grid-template-columns: 1fr;
    }

    .chart-wrapper {
        height: 300px;
    }

    .summary-badges {
        flex-direction: column;
    }

    .download-section {
        flex-direction: column;
    }
}
"""


def get_css() -> str:
    """
    Get the CSS template for HTML reports.

    Returns:
        CSS string ready to be embedded in <style> tags
    """
    return CSS_TEMPLATE


def get_css_escaped() -> str:
    """
    Get the CSS template with braces escaped for f-string usage.

    Returns:
        CSS string with {{ and }} for use in f-strings
    """
    return CSS_TEMPLATE.replace('{', '{{').replace('}', '}}')
