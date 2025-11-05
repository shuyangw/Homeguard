"""
Test that the monthly returns chart is properly rendered in the HTML viewer.
"""

import sys
from pathlib import Path
import tempfile

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from backtesting.engine.multi_symbol_html_viewer import MultiSymbolHTMLViewer


def test_monthly_returns_chart():
    """Test that monthly returns chart is generated with proper data."""

    # Create mock metrics
    metrics = {
        'composition': {
            'Total Symbols': 2,
            'Avg Position Count': 1.5,
            'Max Position Count': 2,
            'Capital Utilization [%]': 75.0
        },
        'attribution': {
            'total_pnl': 1000.0,
            'best_symbol': 'AAPL',
            'worst_symbol': 'MSFT',
            'per_symbol': {
                'AAPL': {
                    'Total P&L': 750.0,
                    'Contribution [%]': 75.0,
                    'Total Trades': 10,
                    'Win Rate [%]': 60.0,
                    'Sharpe Ratio': 1.5,
                    'Avg Hold Duration [days]': 5.2
                },
                'MSFT': {
                    'Total P&L': 250.0,
                    'Contribution [%]': 25.0,
                    'Total Trades': 8,
                    'Win Rate [%]': 50.0,
                    'Sharpe Ratio': 1.2,
                    'Avg Hold Duration [days]': 4.8
                }
            }
        },
        'diversification': {
            'Average Correlation': 0.65,
            'Max Correlation': 0.85,
            'Min Correlation': 0.45
        },
        'rebalancing': {
            'Rebalancing Event Count': 5,
            'Position Turnover [trades/month]': 3.5
        }
    }

    # Create mock charts with monthly returns data
    charts = {
        'monthly_returns_heatmap': {
            'data': [
                {'year': 2023, 'month': 'Jan', 'value': 2.5},
                {'year': 2023, 'month': 'Feb', 'value': -1.2},
                {'year': 2023, 'month': 'Mar', 'value': 3.8},
                {'year': 2023, 'month': 'Apr', 'value': 1.5},
                {'year': 2023, 'month': 'May', 'value': -0.8},
                {'year': 2023, 'month': 'Jun', 'value': 4.2},
                {'year': 2023, 'month': 'Jul', 'value': 2.1},
                {'year': 2023, 'month': 'Aug', 'value': -2.5},
                {'year': 2023, 'month': 'Sep', 'value': 1.8},
                {'year': 2023, 'month': 'Oct', 'value': 3.5},
                {'year': 2023, 'month': 'Nov', 'value': -1.5},
                {'year': 2023, 'month': 'Dec', 'value': 2.8}
            ],
            'months': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        },
        'per_symbol_equity': {
            'labels': ['2023-01-01', '2023-06-01', '2023-12-31'],
            'datasets': []
        },
        'portfolio_composition': {
            'labels': ['2023-01-01', '2023-06-01', '2023-12-31'],
            'datasets': []
        }
    }

    # Generate HTML to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
        temp_path = Path(f.name)

    try:
        MultiSymbolHTMLViewer.generate_html(
            metrics=metrics,
            charts=charts,
            output_path=temp_path,
            title="Monthly Returns Chart Test"
        )

        # Read the generated HTML
        with open(temp_path, 'r', encoding='utf-8') as f:
            html_content = f.read()

        # Verify the chart rendering code is present
        assert 'monthlyReturnsChart' in html_content, "Monthly returns chart canvas missing"
        assert 'type: \'bar\'' in html_content, "Bar chart type not found"
        assert 'Monthly Return %' in html_content, "Chart label missing"

        # Verify the data is embedded
        assert 'Jan 2023' in html_content or '"Jan"' in html_content, "January data missing"
        assert '2.5' in html_content, "January value missing"

        # Check for color coding logic
        assert 'rgba(76, 175, 80' in html_content, "Green color for positive returns missing"
        assert 'rgba(244, 67, 54' in html_content, "Red color for negative returns missing"

        print(f"\n✓ Monthly returns chart HTML generated successfully!")
        print(f"✓ HTML file created at: {temp_path}")
        print(f"✓ File size: {temp_path.stat().st_size} bytes")
        print(f"✓ Bar chart type verified")
        print(f"✓ Color coding verified (green=positive, red=negative)")
        print(f"\nOpen the file in a browser to visually verify the chart renders correctly.")

    finally:
        # Don't delete - let user inspect the file
        print(f"\nGenerated HTML file kept at: {temp_path}")


if __name__ == '__main__':
    test_monthly_returns_chart()
