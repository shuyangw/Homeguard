"""Structural tests for the Grafana alert-rule provisioning files.

Why these exist: Grafana logs and SKIPS a malformed alerting provisioning file,
then starts up healthy with zero rules loaded. That silent-skip behaviour is how
the previous rule set sat unloaded and unnoticed from 2026-04-18 to 2026-07-27.
A schema typo therefore produces no test failure, no startup failure, and no
alert -- only silence. These tests are the guard.

They also cross-check the rules against the two files they can silently drift
from: the scrape config (job labels) and the metrics registry (metric names).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
ALERTING_DIR = REPO_ROOT / 'config' / 'monitoring' / 'grafana' / 'alerting'
SCRAPE_CONFIG = REPO_ROOT / 'config' / 'monitoring' / 'victoria-metrics' / 'scrape.yaml'
REGISTRY_PY = REPO_ROOT / 'src' / 'monitoring' / 'registry.py'

UID_PATTERN = re.compile(r'^[a-zA-Z0-9_-]{1,40}$')
VALID_DATASOURCE_UIDS = {'victoriametrics', '__expr__'}


def _load_rule_files() -> list[tuple[str, dict]]:
    return [(p.name, yaml.safe_load(p.read_text(encoding='utf-8')))
            for p in sorted(ALERTING_DIR.glob('*.yaml'))]


def _all_rules() -> list[tuple[str, dict]]:
    """Yield (file_name, rule) for every provisioned rule."""
    out = []
    for name, doc in _load_rule_files():
        for group in doc.get('groups', []) or []:
            for rule in group.get('rules', []) or []:
                out.append((name, rule))
    return out


def _all_exprs() -> list[tuple[str, str]]:
    """Yield (rule_title, expr) for every Prometheus query node."""
    out = []
    for _, rule in _all_rules():
        for node in rule.get('data', []) or []:
            expr = (node.get('model') or {}).get('expr')
            if expr:
                out.append((rule.get('title', '<untitled>'), expr))
    return out


def test_alerting_dir_is_not_empty():
    assert ALERTING_DIR.is_dir(), f'missing {ALERTING_DIR}'
    assert _all_rules(), 'no provisioned alert rules found'


@pytest.mark.parametrize('file_name,doc', _load_rule_files())
def test_group_envelope(file_name, doc):
    assert doc.get('apiVersion') == 1, f'{file_name}: apiVersion must be 1'
    for group in doc.get('groups', []) or []:
        for key in ('orgId', 'name', 'folder', 'interval'):
            assert key in group, f'{file_name}: group missing {key!r}'
        # Grafana requires the evaluation interval to be a multiple of 10s.
        interval = str(group['interval'])
        assert re.fullmatch(r'\d+[smh]', interval), f'{file_name}: bad interval {interval!r}'


@pytest.mark.parametrize('file_name,rule', _all_rules())
def test_rule_required_fields(file_name, rule):
    for key in ('uid', 'title', 'condition', 'data', 'noDataState', 'execErrState', 'for'):
        assert key in rule, f'{file_name}/{rule.get("title")}: missing {key!r}'
    assert 'severity' in (rule.get('labels') or {}), \
        f'{rule["title"]}: needs a severity label for future notification routing'
    assert UID_PATTERN.match(rule['uid']), f'{rule["title"]}: bad uid {rule["uid"]!r}'
    # `for` must be a duration string; an int is silently wrong.
    assert isinstance(rule['for'], str) and re.fullmatch(r'\d+[smh]', rule['for']), \
        f'{rule["title"]}: `for` must be a duration string, got {rule["for"]!r}'


def test_uids_are_globally_unique():
    uids = [r['uid'] for _, r in _all_rules()]
    duplicates = {u for u in uids if uids.count(u) > 1}
    assert not duplicates, f'duplicate rule uids: {duplicates}'


@pytest.mark.parametrize('file_name,rule', _all_rules())
def test_condition_and_expression_refs_resolve(file_name, rule):
    """`condition` must name a node, and every expression ref an EARLIER node."""
    ref_ids = [n['refId'] for n in rule['data']]
    assert rule['condition'] in ref_ids, \
        f'{rule["title"]}: condition {rule["condition"]!r} not in {ref_ids}'
    # Being a valid refId is not enough: the condition must name the THRESHOLD
    # node. Pointing it at a raw datasource query node is accepted by the schema
    # but means the alert fires on the query's raw value rather than on the
    # intended comparison.
    condition_node = next(n for n in rule['data'] if n['refId'] == rule['condition'])
    condition_type = (condition_node.get('model') or {}).get('type')
    assert condition_type == 'threshold', (
        f'{rule["title"]}: condition names {rule["condition"]!r} of type '
        f'{condition_type!r}; it must name the threshold node'
    )
    seen = []
    for node in rule['data']:
        model = node.get('model') or {}
        assert model.get('refId') == node['refId'], \
            f'{rule["title"]}: model.refId must duplicate the outer refId ({node["refId"]})'
        ref = model.get('expression')
        if ref:
            assert ref in seen, f'{rule["title"]}: node {node["refId"]} references {ref!r} out of order'
        seen.append(node['refId'])


@pytest.mark.parametrize('file_name,rule', _all_rules())
def test_expression_nodes_declare_datasource_both_ways(file_name, rule):
    """Expression nodes need datasourceUid AND the nested model.datasource.

    Omitting the nested form is the most common provisioning failure and fails
    silently at load time.
    """
    for node in rule['data']:
        uid = node.get('datasourceUid')
        assert uid in VALID_DATASOURCE_UIDS, \
            f'{rule["title"]}/{node["refId"]}: unexpected datasourceUid {uid!r}'
        nested = (node.get('model') or {}).get('datasource') or {}
        assert nested.get('uid') == uid, \
            f'{rule["title"]}/{node["refId"]}: model.datasource.uid must mirror datasourceUid'


@pytest.mark.parametrize('title,expr', _all_exprs())
def test_job_labels_exist_in_scrape_config(title, expr):
    """Every job= literal must be a job the scrape config actually defines."""
    scrape = yaml.safe_load(SCRAPE_CONFIG.read_text(encoding='utf-8'))
    known = {c['job_name'] for c in scrape.get('scrape_configs', [])}
    referenced = set(re.findall(r'job=~?"([^"]+)"', expr))
    for ref in referenced:
        # Selectors may be regex alternations, e.g. homeguard-(ramp|omr).
        candidates = _expand_alternation(ref)
        unknown = candidates - known
        assert not unknown, f'{title}: job(s) {unknown} not in {SCRAPE_CONFIG.name} ({sorted(known)})'


def _expand_alternation(selector: str) -> set[str]:
    """Expand a simple `prefix(a|b)suffix` regex selector into literal names."""
    match = re.fullmatch(r'([^()]*)\(([^()]+)\)([^()]*)', selector)
    if not match:
        return {selector}
    prefix, body, suffix = match.groups()
    return {f'{prefix}{part}{suffix}' for part in body.split('|')}


@pytest.mark.parametrize('title,expr', _all_exprs())
def test_hg_metrics_exist_in_registry(title, expr):
    """Every hg_* metric referenced must actually be emitted by the registry.

    This is the test that would have caught hg_broker_reconnect_total, which is
    documented in docs/monitoring/METRIC_SPEC.md but exists nowhere in code.
    """
    registry_src = REGISTRY_PY.read_text(encoding='utf-8')
    for metric in set(re.findall(r'\bhg_[a-z0-9_]+', expr)):
        assert f"'{metric}'" in registry_src or f'"{metric}"' in registry_src, \
            f'{title}: {metric} is not emitted anywhere in {REGISTRY_PY.name}'


@pytest.mark.parametrize('title,expr', _all_exprs())
def test_no_positive_threshold_on_drawdown(title, expr):
    """Regression guard for the inverted-sign defect.

    hg_portfolio_drawdown_pct is negative by construction (see
    docs/monitoring/METRIC_SPEC.md), so the retired `max(...) > 7` form could
    never fire. Any future drawdown expression must not compare it against a
    positive bound.
    """
    if 'hg_portfolio_drawdown_pct' not in expr:
        pytest.skip('no drawdown reference')
    positive_compare = re.search(r'hg_portfolio_drawdown_pct[^<>]*>\s*[0-9]', expr)
    assert not positive_compare, \
        f'{title}: drawdown is negative by construction; a `> positive` bound can never fire'


def test_market_open_gate_is_never_aggregated_across_jobs():
    """Regression guard for the no-op gate defect.

    hg_market_open is unlabeled and run_cscm_live.py hardcodes it to 1.0 for
    crypto, so a bare max(hg_market_open) is permanently 1 across jobs and gates
    nothing. Any use must be scoped to a specific job.
    """
    for title, expr in _all_exprs():
        for match in re.finditer(r'(?:max|min|sum|avg)\s*\(\s*hg_market_open([^)]*)\)', expr):
            selector = match.group(1)
            assert 'job=' in selector, (
                f'{title}: hg_market_open must be scoped to a job. An unscoped '
                f'aggregate is permanently 1 because CSCM pins it to 1.0.'
            )


def test_canary_rule_is_present_and_always_true():
    """The canary is load-bearing: it is the only proof the pipeline evaluates."""
    canaries = [r for _, r in _all_rules() if r['uid'] == 'hg-alerting-canary']
    assert canaries, 'the alerting canary must not be removed'
    exprs = [(n.get('model') or {}).get('expr') for n in canaries[0]['data']]
    assert 'vector(1)' in [e for e in exprs if e], 'canary must be unconditionally true'
