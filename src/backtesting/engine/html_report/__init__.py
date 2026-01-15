"""
HTML Report Generation Module.

This module provides components for generating HTML backtest reports:
- CSS templates for styling (dark/light themes)
- JavaScript for interactivity (charts, theme toggle)
- Section builders for HTML content generation

Usage:
    from src.backtesting.engine.html_report import get_css, get_base_js, get_chart_js

    css = get_css()
    js = get_chart_js(
        symbols_list=['AAPL', 'GOOGL'],
        returns_list=[10.5, -2.3],
        sharpe_list=[1.2, 0.8],
        drawdown_list=[-5.0, -8.0],
        win_rate_list=[55.0, 48.0],
        csv_data='symbol,return\\nAAPL,10.5\\nGOOGL,-2.3',
        summary={'Win Rate (Symbols)': 50, ...}
    )
"""

from .css import get_css, get_css_escaped, CSS_TEMPLATE
from .js import (
    get_base_js,
    get_chart_js,
    JS_BASE_TEMPLATE,
    JS_RETURNS_CHART,
    JS_SHARPE_CHART,
    JS_DRAWDOWN_CHART,
    JS_RISK_RETURN_CHART,
    JS_WIN_RATE_CHART,
    JS_THEME_UPDATE,
)

__all__ = [
    # CSS exports
    'get_css',
    'get_css_escaped',
    'CSS_TEMPLATE',
    # JS exports
    'get_base_js',
    'get_chart_js',
    'JS_BASE_TEMPLATE',
    'JS_RETURNS_CHART',
    'JS_SHARPE_CHART',
    'JS_DRAWDOWN_CHART',
    'JS_RISK_RETURN_CHART',
    'JS_WIN_RATE_CHART',
    'JS_THEME_UPDATE',
]
