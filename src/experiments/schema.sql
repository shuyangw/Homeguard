-- Experiment registry schema per docs/methodology/backtesting.md Section 9.
-- Two tables: runs (one row per backtest/optimization run) and
-- return_streams (the daily OOS return series of every passing strategy).

CREATE TABLE IF NOT EXISTS runs (
    run_id              VARCHAR PRIMARY KEY,
    timestamp_utc       TIMESTAMP,
    strategy_name       VARCHAR,
    agent_name          VARCHAR,
    phase               VARCHAR,
    parent_run_id       VARCHAR,

    -- Configuration
    params              JSON,
    universe_name       VARCHAR,
    asset_class         VARCHAR,
    data_frequency      VARCHAR,

    -- Windows
    window_start        DATE,
    window_end          DATE,
    is_start            DATE,
    is_end              DATE,
    oos_start           DATE,
    oos_end             DATE,
    n_folds             INTEGER,

    -- Metrics (computed on OOS / pooled)
    metrics             JSON,
    regime_breakdown    JSON,
    fold_metrics        JSON,

    -- Cost
    cost_tier_used      VARCHAR,
    cost_bps            DECIMAL(10, 4),
    cost_sensitivity    JSON,

    -- Multiple-testing
    combinations_in_run     INTEGER,
    combinations_project    INTEGER,

    -- Reproducibility identity (Section 8.1)
    git_sha             VARCHAR,
    config_sha          VARCHAR,
    data_snapshot_date  DATE,
    python_env_hash     VARCHAR,
    random_seeds        JSON,
    wall_clock_start    TIMESTAMP,
    wall_clock_end      TIMESTAMP,
    host                VARCHAR,

    -- Outcomes
    verdict             VARCHAR,
    verdict_reasons     JSON,
    notes               VARCHAR
);

CREATE TABLE IF NOT EXISTS return_streams (
    run_id              VARCHAR,
    date                DATE,
    return_pct          DECIMAL(18, 10),
    position_count      INTEGER,
    PRIMARY KEY (run_id, date)
);
