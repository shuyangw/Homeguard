"""
Weekly QuantStats report generator.

Pulls equity time series from VictoriaMetrics, generates a QuantStats
tearsheet, and posts it to Discord via webhook.

Run via systemd timer every Sunday 18:00 ET.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

from src.utils.logger import get_logger

logger = get_logger()

VM_URL = "http://127.0.0.1:8428/api/v1/query_range"
DISCORD_WEBHOOK_URL = os.getenv('DISCORD_MONITORING_WEBHOOK', '')


def query_vm(query: str, start: float, end: float, step: str = '1h') -> pd.Series:
    """Query VictoriaMetrics and return a pandas Series."""
    resp = requests.get(VM_URL, params={
        'query': query,
        'start': start,
        'end': end,
        'step': step,
    }, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    results = data.get('data', {}).get('result', [])
    if not results:
        return pd.Series(dtype=float)

    values = results[0]['values']
    return pd.Series(
        [float(v[1]) for v in values],
        index=[pd.Timestamp(v[0], unit='s', tz='UTC') for v in values],
    )


def generate_report():
    """Generate weekly report and post to Discord."""
    load_dotenv()

    end_ts = datetime.now().timestamp()
    start_ts = (datetime.now() - timedelta(days=7)).timestamp()

    logger.info("Fetching equity curve from VictoriaMetrics...")
    equity = query_vm('max(hg_portfolio_equity_usd)', start_ts, end_ts)

    if equity.empty or len(equity) < 10:
        logger.warning("Insufficient data for weekly report (need at least 10 data points)")
        return

    returns = equity.pct_change().dropna()

    # Generate QuantStats report
    try:
        import quantstats as qs
        report_path = '/tmp/homeguard_weekly.html'
        qs.reports.html(returns, output=report_path, title='Homeguard Weekly Report')
        logger.info(f"Report generated: {report_path}")
    except ImportError:
        logger.warning("QuantStats not installed, generating text summary instead")
        report_path = None

    # Post summary to Discord
    webhook = os.getenv('DISCORD_MONITORING_WEBHOOK', '')
    if webhook:
        summary = {
            'content': (
                f"**Homeguard Weekly Report** ({datetime.now().strftime('%Y-%m-%d')})\n"
                f"Equity: ${equity.iloc[-1]:,.2f}\n"
                f"Week Return: {returns.sum():.2%}\n"
                f"Data Points: {len(equity)}"
            )
        }
        try:
            resp = requests.post(webhook, json=summary, timeout=10)
            resp.raise_for_status()
            logger.info("Posted summary to Discord")

            # Upload HTML report if available
            if report_path and os.path.exists(report_path):
                with open(report_path, 'rb') as f:
                    resp = requests.post(
                        webhook,
                        files={'file': ('weekly_report.html', f, 'text/html')},
                        timeout=30,
                    )
                    resp.raise_for_status()
                    logger.info("Uploaded report to Discord")
        except Exception as e:
            logger.error(f"Discord post failed: {e}")
    else:
        logger.warning("DISCORD_MONITORING_WEBHOOK not set, skipping Discord post")

    logger.info("Weekly report complete")


if __name__ == '__main__':
    generate_report()
