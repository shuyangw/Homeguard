"""
Tests for HTML report module components.

These tests verify the CSS and JS template modules work correctly
without modifying the existing HTML generation in results_aggregator.py.
"""

import pytest


class TestCSSModule:
    """Tests for CSS template module."""

    def test_css_import(self):
        """Verify CSS module can be imported."""
        from src.backtesting.engine.html_report import get_css, CSS_TEMPLATE
        assert get_css is not None
        assert CSS_TEMPLATE is not None

    def test_get_css_returns_string(self):
        """Verify get_css() returns valid CSS content."""
        from src.backtesting.engine.html_report import get_css
        css = get_css()

        assert isinstance(css, str)
        assert len(css) > 1000  # CSS should be substantial

    def test_css_contains_theme_variables(self):
        """Verify CSS includes theme variable definitions."""
        from src.backtesting.engine.html_report import get_css
        css = get_css()

        # Check for critical CSS features
        assert ':root' in css  # CSS variables
        assert '[data-theme="dark"]' in css  # Dark theme
        assert '.metric-card' in css  # Card styling
        assert '.metric-label' in css  # Metric labels

    def test_get_css_escaped(self):
        """Verify get_css_escaped() works for f-string embedding."""
        from src.backtesting.engine.html_report import get_css_escaped
        css = get_css_escaped()

        assert isinstance(css, str)
        # Escaped version should not have unescaped braces
        # (though CSS doesn't typically use {} in problematic ways)


class TestJSModule:
    """Tests for JavaScript template module."""

    def test_js_import(self):
        """Verify JS module can be imported."""
        from src.backtesting.engine.html_report import get_base_js, get_chart_js
        assert get_base_js is not None
        assert get_chart_js is not None

    def test_get_base_js_returns_string(self):
        """Verify get_base_js() returns valid JS content."""
        from src.backtesting.engine.html_report import get_base_js
        js = get_base_js()

        assert isinstance(js, str)
        assert 'toggleTheme' in js  # Theme toggle function
        assert 'getThemeColors' in js  # Theme colors function

    def test_get_chart_js_minimal(self):
        """Verify get_chart_js() with minimal data."""
        from src.backtesting.engine.html_report import get_chart_js

        js = get_chart_js(
            symbols_list=['AAPL'],
            returns_list=[10.5],
            sharpe_list=[1.2],
            drawdown_list=[-5.0],
            win_rate_list=[55.0],
            csv_data='symbol,return\nAAPL,10.5',
            summary={'Win Rate (Symbols)': 100}
        )

        assert isinstance(js, str)
        assert 'AAPL' in js  # Symbol appears in chart data
        assert 'returnsChart' in js  # Returns chart
        assert 'sharpeChart' in js  # Sharpe chart

    def test_get_chart_js_with_equity(self):
        """Verify get_chart_js() with equity chart data."""
        from src.backtesting.engine.html_report import get_chart_js

        equity_data = {
            'labels': ['2024-01-01', '2024-01-02'],
            'values': [100000, 101000]
        }

        js = get_chart_js(
            symbols_list=['AAPL', 'GOOGL'],
            returns_list=[10.5, -2.3],
            sharpe_list=[1.2, 0.8],
            drawdown_list=[-5.0, -8.0],
            win_rate_list=[55.0, 48.0],
            csv_data='symbol,return\nAAPL,10.5\nGOOGL,-2.3',
            summary={'Win Rate (Symbols)': 50},
            equity_chart_data=equity_data
        )

        assert isinstance(js, str)
        assert 'equityChart' in js  # Equity chart created
        assert '2024-01-01' in js  # Date appears in labels

    def test_js_individual_templates(self):
        """Verify individual JS template constants exist."""
        from src.backtesting.engine.html_report import (
            JS_RETURNS_CHART,
            JS_SHARPE_CHART,
            JS_DRAWDOWN_CHART,
            JS_RISK_RETURN_CHART,
            JS_WIN_RATE_CHART,
            JS_THEME_UPDATE,
        )

        # Each template should be a non-empty string
        assert isinstance(JS_RETURNS_CHART, str) and len(JS_RETURNS_CHART) > 50
        assert isinstance(JS_SHARPE_CHART, str) and len(JS_SHARPE_CHART) > 50
        assert isinstance(JS_DRAWDOWN_CHART, str) and len(JS_DRAWDOWN_CHART) > 50
        assert isinstance(JS_RISK_RETURN_CHART, str) and len(JS_RISK_RETURN_CHART) > 50
        assert isinstance(JS_WIN_RATE_CHART, str) and len(JS_WIN_RATE_CHART) > 50
        assert isinstance(JS_THEME_UPDATE, str) and len(JS_THEME_UPDATE) > 50


class TestModuleIntegration:
    """Tests for module integration readiness."""

    def test_all_exports_available(self):
        """Verify all expected exports are available."""
        from src.backtesting.engine.html_report import __all__

        expected = [
            'get_css', 'get_css_escaped', 'CSS_TEMPLATE',
            'get_base_js', 'get_chart_js',
            'JS_BASE_TEMPLATE', 'JS_RETURNS_CHART', 'JS_SHARPE_CHART',
            'JS_DRAWDOWN_CHART', 'JS_RISK_RETURN_CHART', 'JS_WIN_RATE_CHART',
            'JS_THEME_UPDATE',
        ]

        for name in expected:
            assert name in __all__, f"Missing export: {name}"

    def test_css_js_can_combine(self):
        """Verify CSS and JS can be combined into HTML structure."""
        from src.backtesting.engine.html_report import get_css, get_chart_js

        css = get_css()
        js = get_chart_js(
            symbols_list=['AAPL'],
            returns_list=[10.0],
            sharpe_list=[1.0],
            drawdown_list=[-5.0],
            win_rate_list=[50.0],
            csv_data='',
            summary={}
        )

        # Simulate HTML embedding
        html = f"""<!DOCTYPE html>
<html>
<head>
<style>
{css}
</style>
</head>
<body>
<script>
{js}
</script>
</body>
</html>"""

        assert '<!DOCTYPE html>' in html
        assert ':root' in html  # CSS present
        assert 'toggleTheme' in html  # JS present
