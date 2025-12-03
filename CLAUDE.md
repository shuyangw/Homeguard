# Claude Coding Guidelines for Homeguard

This document provides an overview of coding standards and guidelines for the Homeguard backtesting framework. For detailed information on specific topics, refer to the specialized guideline documents in [`.claude/`](.claude/).

## Quick Reference

### Environment Setup
**CRITICAL**: All Python code execution MUST use the `fintech` conda environment.
- Location: `C:\Users\qwqw1\anaconda3\envs\fintech`
- Activate: `conda activate fintech`
- Details: [`.claude/environment.md`](.claude/environment.md)

### Project Organization
Maintain clean project structure with proper separation of concerns.
- No script files in root directory
- Scripts go in `src/`, `tests/`, `scripts/`, `backtest_scripts/`
- Documentation co-located with modules
- Details: [`.claude/project_structure.md`](.claude/project_structure.md)

### Python Code Standards
Write clean, maintainable code following project conventions.
- Minimal comments - code should be self-explanatory
- Descriptive naming - functions/variables explain their purpose
- Always run unit tests before committing
- Details: [`.claude/code_standards.md`](.claude/code_standards.md)

### Backtesting Guidelines
**CRITICAL**: Avoid lookahead bias, survivorship bias, and overfitting.
- **ALWAYS use the config-driven backtesting system** - don't write ad-hoc scripts
- Run backtests via: `python -m src.backtest_runner --config config/backtesting/ma_single.yaml`
- Consult `backtest_guidelines/guidelines.md` before modifying backtest code
- Use market calendar for trading day filtering
- Apply proper risk management
- Details: [`.claude/backtesting.md`](.claude/backtesting.md)

### Existing Backtest Tools (CHECK BEFORE CREATING NEW)
**CRITICAL**: Before creating any new backtest-related script or tool, check if one already exists below. Extend existing tools rather than creating duplicates.

| Tool | Location | Purpose |
|------|----------|---------|
| **Standard Report** | `scripts/backtest/run_standard_report.py` | Monthly Sharpe/drawdown reports for any strategy |
| **Config-Driven Runner** | `python -m src.backtest_runner` | Main backtest runner with YAML configs |
| **Walk-Forward** | `config/backtesting/lgbm_walk_forward.yaml` | Out-of-sample validation |

**Standard Report Generator**:
- Module: `src/backtesting/reporting/standard_report.py`
- Usage: `python scripts/backtest/run_standard_report.py --strategy <name> --symbols <list>`
- Outputs: Console, Markdown, CSV to `settings.ini` output directory
- Symbol lists: `backtest_lists/*.csv`

**Adding New Backtest Tools**:
- Add new modules to `src/backtesting/` (not standalone scripts)
- Register in appropriate `__init__.py`
- Document in this table
- Prefer extending `StandardReportGenerator` for new report types

### Risk Management
**CRITICAL**: All backtests MUST use proper position sizing.
- Default: 10% per trade (moderate risk profile)
- Never use 99% capital per trade
- Five position sizing methods available
- Details: [`.claude/risk_management.md`](.claude/risk_management.md)

### Testing Requirements
**ALWAYS** run unit tests when modifying:
- Backtesting engine code
- Strategy implementations
- Report generation
- Details: [`.claude/testing.md`](.claude/testing.md)

### Logging Standards
**CRITICAL**: Use centralized logging module (`src/utils/logger.py`) for all output.
- Never use `print()` statements
- **ALWAYS log exceptions** - Never silently swallow errors
- Use `logger.error()` for all caught exceptions
- Color-coded output (green=success, red=error, etc.)
- Details: [`.claude/logging.md`](.claude/logging.md)

### GUI Design
**CRITICAL**: Dark theme with bright text for readability.
- Bright white text on dark backgrounds
- Semantic color coding (blue=primary, green=success, red=error)
- Details: [`.claude/gui_design.md`](.claude/gui_design.md)

### Documentation
Update docs when modifying user-facing functionality.
- README files, example scripts, API docs
- Progress tracking in `docs/progress/`
- **CRITICAL**: All reports and documentation MUST use timestamp prefixes (YYYYMMDD format)
  - Example: `20251111_MULTI_PAIR_PORTFOLIO_RESULTS.md`
  - Stored in `docs/reports/` directory
- Details: [`.claude/documentation.md`](.claude/documentation.md)

### Architecture Documentation
**CRITICAL**: Update architecture docs whenever changing system architecture.
- **ALWAYS** update `docs/architecture/` when adding/removing/moving modules
- Update `ARCHITECTURE_OVERVIEW.md` for structural changes
- Update `MODULE_REFERENCE.md` when adding/modifying modules
- Update `DATA_FLOW.md` when changing data pipelines
- Architecture docs must reflect actual codebase structure
- Details: [`.claude/documentation.md`](.claude/documentation.md)

### Infrastructure & Deployment Documentation
**CRITICAL**: Update infrastructure docs when changing deployment/cloud infrastructure.
- **ALWAYS** update `docs/INFRASTRUCTURE_OVERVIEW.md` when modifying AWS resources
- Update `terraform/README.md` when changing Terraform configuration
- Update `scripts/ec2/` documentation when adding/modifying management scripts
- Keep `docs/HEALTH_CHECK_CHEATSHEET.md` current with monitoring procedures
- Document cost changes, instance type modifications, or scheduling updates
- Infrastructure docs must reflect actual deployed resources
- Details: [`.claude/documentation.md`](.claude/documentation.md)

### Git Workflow
**CRITICAL**: Never push to remote without explicit user permission.
- Create commits for completed work
- Stage files with `git add`
- Write clear, descriptive commit messages
- **NEVER** run `git push` unless user explicitly requests it
- Ask before pushing: "Ready to push to remote?"
- Details: [`.claude/git_workflow.md`](.claude/git_workflow.md)

### Live Trading
**CRITICAL**: Watch for common live trading issues.
- **Type mismatches** - API data comes as strings; always convert explicitly
- **VIX data resilience** - Must have fallbacks for VIX fetch failures
- **Bayesian model coverage** - Model must be trained with ALL trading universe symbols
- **Market hours** - Trading only at 3:50 PM ET, exits at 9:35 AM ET
- **Timezone handling** - ALWAYS use `from src.utils.timezone import tz` and `tz.now()` instead of `datetime.now()`. EC2 instances run in UTC; the timezone utility ensures consistent Eastern Time handling.
- Details: [`.claude/live_trading.md`](.claude/live_trading.md)

### Live Trading Tools & Agents
**Available agents and tools for live trading diagnostics:**

| Tool/Agent | Location | Purpose |
|------------|----------|---------|
| **Trade Log Analyzer** | `.claude/agents/trade-log-analyzer.md` | Analyze today's trading logs, identify errors, propose fixes |
| **Backtest Optimizer** | `.claude/agents/backtest-optimizer.md` | Optimize strategy parameters and run systematic backtests |

**EC2 Management Scripts** (Windows):
- `scripts\ec2\local_start_instance.bat` - Start EC2 instance
- `scripts\ec2\local_stop_instance.bat` - Stop EC2 instance
- `scripts\ec2\check_bot.bat` - Check bot status
- `scripts\ec2\view_logs.bat` - Stream live logs
- `scripts\ec2\daily_health_check.bat` - Run 6-point health check

**EC2 Instance Aliases** (when connected via SSH):
- `bot-status` - Check systemd service status
- `bot-logs` - Stream live logs (colored)
- `bot-logs-recent` - View last 100 log lines
- `bot-update` - Pull code and restart bot
- `bot-restart` - Restart trading bot service

### Common Type Issues
Pylance/VectorBT type annotation patterns.
- DataFrame.xs() type hints
- VectorBT incomplete stubs
- SQL injection prevention
- Details: [`.claude/type_issues.md`](.claude/type_issues.md)

## File Organization

```
.claude/
├── agents/                      # Claude Code agent definitions
│   ├── backtest-optimizer.md    # Strategy optimization agent
│   └── trade-log-analyzer.md    # Live trading log analysis agent
├── backtesting.md               # Backtesting best practices
├── code_standards.md            # Python code quality standards
├── documentation.md             # Documentation requirements
├── environment.md               # Python environment setup
├── git_workflow.md              # Git commit and push guidelines
├── gui_design.md                # GUI design standards
├── live_trading.md              # Live trading issues and pitfalls
├── logging.md                   # Logging requirements
├── project_structure.md         # File organization rules
├── risk_management.md           # Position sizing and risk
├── testing.md                   # Unit testing requirements
└── type_issues.md               # Common Pylance type fixes
```

## Defensive Mindset

**CRITICAL**: Always assume something can go wrong. Be realistic, not optimistic.

### Verification Over Assumption
- **Never assume code works** - always run and verify
- **Never assume tests pass** - run the full test suite after changes
- **Never assume files exist** - check before reading/writing
- **Never assume APIs return expected data** - handle edge cases

### After Making Changes
- Run relevant tests immediately - don't batch verification
- Check for import errors by actually importing the module
- Verify file writes by reading back the content
- Test edge cases, not just the happy path

### When Reporting Status
- Don't say "fixed" until verified with tests
- Don't say "complete" until all edge cases are handled
- Report failures and partial successes honestly
- If something might break, say so explicitly

### Common Failure Points
- **Data type mismatches** - str vs int, float vs Decimal, datetime vs str timestamps
- Import cycles when adding new modules
- Missing dependencies in different environments
- Path issues between Windows/macOS/Linux
- Race conditions in parallel code
- Silent failures that return None instead of raising

### Error Handling Philosophy
- Fail fast and loud - don't hide errors
- Log all exceptions with full context
- Return explicit error states, not silent None
- Test error paths, not just success paths

## Getting Started

1. **Read this overview** - Understand the quick reference topics
2. **Consult specific guides** - Dive into detailed docs as needed
3. **Follow the standards** - Apply guidelines consistently
4. **Run tests** - Always test before committing
5. **Update docs** - Keep documentation synchronized with code

## Critical Rules (Always Follow)

1. ✅ Use `fintech` conda environment for all Python operations
2. ✅ Keep root directory clean - no script files
3. ✅ Run unit tests before committing code changes
4. ✅ Use proper risk management in backtests
5. ✅ Use centralized logger - never `print()`
6. ✅ **ALWAYS log exceptions with `logger.error()` - never silently swallow errors**
7. ✅ Bright text on dark backgrounds in GUI
8. ✅ Update documentation when modifying features
9. ✅ **Update architecture docs when changing system structure**
10. ✅ **Update infrastructure docs when modifying AWS deployment**
11. ✅ Consult `backtest_guidelines/guidelines.md` before backtesting changes
12. ✅ **Timestamp all documentation files (YYYYMMDD_filename.md format)**
13. ✅ **NEVER push to remote without explicit user permission**
14. ✅ **Use config-driven backtesting system** - don't write ad-hoc backtest scripts
15. ✅ **Verify before claiming success** - run tests, don't assume code works
16. ✅ **Check existing backtest tools first** - see "Existing Backtest Tools" table before creating new ones

## When to Consult Detailed Guides

- **Running a backtest** → Use config-driven system, see [`.claude/backtesting.md`](.claude/backtesting.md)
- **Before backtesting work** → Read [`.claude/backtesting.md`](.claude/backtesting.md)
- **Live trading issues** → Read [`.claude/live_trading.md`](.claude/live_trading.md)
- **Adding GUI components** → Read [`.claude/gui_design.md`](.claude/gui_design.md)
- **Writing tests** → Read [`.claude/testing.md`](.claude/testing.md)
- **Implementing risk features** → Read [`.claude/risk_management.md`](.claude/risk_management.md)
- **Fixing type errors** → Read [`.claude/type_issues.md`](.claude/type_issues.md)
- **Organizing files** → Read [`.claude/project_structure.md`](.claude/project_structure.md)
- **Adding logging** → Read [`.claude/logging.md`](.claude/logging.md)
- **Creating documentation** → Read [`.claude/documentation.md`](.claude/documentation.md)
- **Committing or pushing code** → Read [`.claude/git_workflow.md`](.claude/git_workflow.md)
- **Modifying AWS infrastructure** → Update `docs/INFRASTRUCTURE_OVERVIEW.md` and `terraform/README.md`

---

**Note**: This overview provides quick reference. For comprehensive details and code examples, always refer to the specific guideline files in [`.claude/`](.claude/).
