"""Validate Grafana alert-rule PromQL against the live datasource.

Extracts every `model.expr` from the alerting provisioning files and executes it
against VictoriaMetrics, reporting HTTP status, series count, and emptiness.

Why this exists: the provisioning file is loaded by Grafana at startup and a bad
expression surfaces as a rule stuck in `error` state, not as a startup failure.
VictoriaMetrics also implements MetricsQL rather than pure PromQL, so functions
like `hour()`, `day_of_week()`, `or vector(0)`, and chained `and on()` need
confirming on the real backend rather than assumed.

An empty result is NOT a failure: most rules encode their comparison inside the
expression, so empty means "condition not met" (paired with `noDataState: OK`).
Empty is flagged so a rule that is empty for the WRONG reason -- a typo'd metric
name rather than a closed gate -- can be spotted.

Usage (on the EC2 host, where 8428 is reachable on loopback):
    python3 scripts/monitoring/validate_alert_exprs.py

Off-host, point it at the tailnet Grafana datasource proxy instead:
    python3 scripts/monitoring/validate_alert_exprs.py \
        --url https://homeguard-ec2.<tailnet>.ts.net/api/datasources/proxy/uid/victoriametrics \
        --bearer "$GRAFANA_SERVICE_ACCOUNT_TOKEN"
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Iterator

import yaml

DEFAULT_URL = 'http://127.0.0.1:8428'
ALERTING_DIR = Path(__file__).resolve().parents[2] / 'config' / 'monitoring' / 'grafana' / 'alerting'


def iter_exprs(alerting_dir: Path) -> Iterator[tuple[str, str, str]]:
    """Yield (file_name, rule_title, expr) for every Prometheus query node."""
    for path in sorted(alerting_dir.glob('*.yaml')):
        doc = yaml.safe_load(path.read_text(encoding='utf-8'))
        for group in doc.get('groups', []) or []:
            for rule in group.get('rules', []) or []:
                title = rule.get('title', '<untitled>')
                for node in rule.get('data', []) or []:
                    expr = (node.get('model') or {}).get('expr')
                    if expr:
                        yield path.name, title, expr


def query(base_url: str, expr: str, bearer: str | None, timeout: float) -> dict[str, Any]:
    """Run an instant query. Returns {'ok', 'series', 'error'}."""
    url = f"{base_url.rstrip('/')}/api/v1/query?" + urllib.parse.urlencode({'query': expr})
    request = urllib.request.Request(url)
    if bearer:
        request.add_header('Authorization', f'Bearer {bearer}')
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode('utf-8'))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode('utf-8', 'replace')[:300]
        return {'ok': False, 'series': 0, 'error': f'HTTP {exc.code}: {body}'}
    except (urllib.error.URLError, OSError) as exc:
        return {'ok': False, 'series': 0, 'error': f'unreachable: {exc}'}
    except json.JSONDecodeError as exc:
        return {'ok': False, 'series': 0, 'error': f'bad JSON: {exc}'}

    if payload.get('status') != 'success':
        detail = payload.get('error') or payload.get('errorType') or 'unknown'
        return {'ok': False, 'series': 0, 'error': str(detail)[:300]}
    return {'ok': True, 'series': len(payload.get('data', {}).get('result', [])), 'error': None}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default=DEFAULT_URL, help=f'query base URL (default {DEFAULT_URL})')
    parser.add_argument('--bearer', default=None, help='bearer token, for the Grafana proxy path')
    parser.add_argument('--alerting-dir', type=Path, default=ALERTING_DIR)
    parser.add_argument('--timeout', type=float, default=20.0)
    args = parser.parse_args()

    if not args.alerting_dir.is_dir():
        print(f'[-] No alerting dir at {args.alerting_dir}')
        return 2

    exprs = list(iter_exprs(args.alerting_dir))
    if not exprs:
        print(f'[-] No expressions found in {args.alerting_dir}')
        return 2

    print(f'[*] Validating {len(exprs)} expression(s) against {args.url}\n')
    failures = 0
    empties = 0
    for file_name, title, expr in exprs:
        result = query(args.url, expr, args.bearer, args.timeout)
        if not result['ok']:
            failures += 1
            status = '[-] FAIL'
        elif result['series'] == 0:
            empties += 1
            status = '[!] empty'
        else:
            status = f"[+] {result['series']} series"
        print(f'{status}  {file_name} :: {title}')
        print(f'    {" ".join(expr.split())}')
        if result['error']:
            print(f'    error: {result["error"]}')
        print()

    print(f'[*] {len(exprs) - failures} ok, {failures} failed, {empties} empty')
    if empties:
        print('    Empty is expected for gated rules (paired with noDataState: OK).')
        print('    Confirm each empty result is a closed gate, not a typo.')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
