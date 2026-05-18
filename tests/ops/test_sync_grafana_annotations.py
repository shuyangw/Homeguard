"""Tests for scripts/ops/sync_grafana_annotations.py."""
import json
from unittest.mock import patch, MagicMock
import pytest


def test_sync_creates_missing_annotation(tmp_path, monkeypatch):
    """Annotation that doesn't exist yet is POSTed."""
    spec_file = tmp_path / 'instance_off.json'
    spec_file.write_text(json.dumps([
        {'start': '2026-04-16T13:00:00Z',
         'end': '2026-04-24T13:00:00Z',
         'text': 'Trading instance offline'},
    ]))
    posted = []

    def fake_urlopen(req, *args, **kw):
        url = req.full_url if hasattr(req, 'full_url') else req.get_full_url()
        resp = MagicMock()
        if 'annotations' in url and req.method == 'GET':
            # No existing annotations
            resp.read.return_value = b'[]'
        else:
            # POST response
            posted.append(json.loads(req.data.decode()))
            resp.read.return_value = b'{"id": 1}'
        resp.__enter__ = lambda s: s
        resp.__exit__ = lambda s, *a: None
        return resp

    monkeypatch.setattr('urllib.request.urlopen', fake_urlopen)
    monkeypatch.setattr('sys.argv', ['sync_grafana_annotations.py',
                                      '--source', str(spec_file),
                                      '--grafana-url', 'http://localhost:3000',
                                      '--api-key', 'fake-key',
                                      '--dashboard-uid', 'portfolio-overview'])
    from scripts.ops.sync_grafana_annotations import main
    assert main() == 0
    assert len(posted) == 1
    assert posted[0]['text'] == 'Trading instance offline'
    assert 'instance-off' in posted[0]['tags']


def test_sync_is_idempotent_when_annotation_already_present(tmp_path, monkeypatch):
    """Annotation matching tag+time range+text is NOT re-posted."""
    spec_file = tmp_path / 'instance_off.json'
    spec_file.write_text(json.dumps([
        {'start': '2026-04-16T13:00:00Z',
         'end': '2026-04-24T13:00:00Z',
         'text': 'Trading instance offline'},
    ]))
    from scripts.ops.sync_grafana_annotations import _iso_to_ms
    posted = []
    existing = [{
        'id': 99,
        'time': _iso_to_ms('2026-04-16T13:00:00Z'),
        'timeEnd': _iso_to_ms('2026-04-24T13:00:00Z'),
        'text': 'Trading instance offline',
        'tags': ['instance-off'],
    }]

    def fake_urlopen(req, *args, **kw):
        resp = MagicMock()
        if req.method == 'GET':
            resp.read.return_value = json.dumps(existing).encode()
        else:
            posted.append(json.loads(req.data.decode()))
            resp.read.return_value = b'{"id": 1}'
        resp.__enter__ = lambda s: s
        resp.__exit__ = lambda s, *a: None
        return resp

    monkeypatch.setattr('urllib.request.urlopen', fake_urlopen)
    monkeypatch.setattr('sys.argv', ['sync_grafana_annotations.py',
                                      '--source', str(spec_file),
                                      '--grafana-url', 'http://localhost:3000',
                                      '--api-key', 'fake-key',
                                      '--dashboard-uid', 'portfolio-overview'])
    from scripts.ops.sync_grafana_annotations import main
    assert main() == 0
    assert posted == []  # idempotent: nothing new posted
