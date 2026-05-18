#!/usr/bin/env python3
"""Sync `instance_off.json` -> Grafana annotations via API.

Idempotent: an annotation matching (tag, time range, text) is not re-posted.

Usage on EC2:
  python -m scripts.ops.sync_grafana_annotations \
    --source config/monitoring/grafana/annotations/instance_off.json \
    --grafana-url http://127.0.0.1:3000 \
    --api-key $GRAFANA_API_KEY \
    --dashboard-uid portfolio-overview
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict

TAG = 'instance-off'


def _iso_to_ms(iso: str) -> int:
    """Parse ISO-8601 (with Z suffix) to Unix milliseconds."""
    dt = datetime.fromisoformat(iso.replace('Z', '+00:00'))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _fetch_existing(grafana_url: str, api_key: str, dashboard_uid: str) -> List[Dict]:
    """List existing annotations with the instance-off tag on the given dashboard."""
    params = urllib.parse.urlencode({'tags': TAG, 'dashboardUID': dashboard_uid, 'limit': 1000})
    req = urllib.request.Request(
        f'{grafana_url}/api/annotations?{params}',
        headers={'Authorization': f'Bearer {api_key}'},
        method='GET',
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode())


def _matches_existing(entry: Dict, existing: List[Dict]) -> bool:
    """True iff an existing annotation has same start/end (ms) and text."""
    target_start = _iso_to_ms(entry['start'])
    target_end = _iso_to_ms(entry['end'])
    target_text = entry['text']
    for e in existing:
        if (e.get('time') == target_start
                and e.get('timeEnd') == target_end
                and e.get('text') == target_text):
            return True
    return False


def _post_annotation(grafana_url: str, api_key: str, dashboard_uid: str, entry: Dict) -> None:
    payload = {
        'dashboardUID': dashboard_uid,
        'time': _iso_to_ms(entry['start']),
        'timeEnd': _iso_to_ms(entry['end']),
        'text': entry['text'],
        'tags': [TAG],
    }
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        f'{grafana_url}/api/annotations',
        data=body,
        headers={
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
        },
        method='POST',
    )
    with urllib.request.urlopen(req) as resp:
        _ = resp.read()


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description='Sync instance_off.json into Grafana annotations.')
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--grafana-url', type=str, required=True)
    p.add_argument('--api-key', type=str, required=True)
    p.add_argument('--dashboard-uid', type=str, required=True)
    return p.parse_args(argv)


def main() -> int:
    args = _parse_args()
    spec = json.loads(args.source.read_text())
    existing = _fetch_existing(args.grafana_url, args.api_key, args.dashboard_uid)
    created = 0
    for entry in spec:
        if _matches_existing(entry, existing):
            continue
        _post_annotation(args.grafana_url, args.api_key, args.dashboard_uid, entry)
        created += 1
    print(f'sync_grafana_annotations: {created} new, {len(spec) - created} already present')
    return 0


if __name__ == '__main__':
    sys.exit(main())
