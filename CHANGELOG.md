# Changelog

All notable changes to Homeguard are documented here.

## [2025-12-06]

### Changed
- Removed hardcoded EC2 IPs/instance IDs from documentation (use `.env` placeholders)
- Consolidated documentation structure:
  - Merged `docs/plans/` and `docs/todos/` into `docs/planning/`
  - Archived legacy reports to `docs/archive/legacy-reports/`
  - Archived 2024 progress files to `docs/archive/progress-2024/`
- Added CONTRIBUTING.md and CHANGELOG.md

### Security
- All EC2 management scripts now read from `.env` file instead of hardcoded values

## [2025-11-15]

### Added
- AWS EC2 deployment infrastructure (Terraform)
- Lambda-powered automated scheduling (9 AM start, 4:30 PM stop ET Mon-Fri)
- Systemd service with auto-restart capabilities
- Discord bot integration for observability
- 10 SSH management scripts in `scripts/ec2/`
- Comprehensive health check system (6-point validation)

### Infrastructure
- EC2 t4g.small ARM64 instance (Amazon Linux 2023)
- ~$7/month total cost (46% savings vs 24/7 operation)

## [2025-11-06]

### Added
- Regime-based testing (4 levels of validation)
- Walk-forward validation framework
- GUI integration for regime analysis

## [2025-11-05]

### Changed
- Reorganized documentation into topic folders
- Created Architecture Overview and Module Reference
- Completed all 50 backtest accuracy tests

## [2025-11-03]

### Added
- Parallel chart generation
- Portfolio mode GUI improvements
- Benchmark comparison feature

## [2025-10-31]

### Added
- Initial strategy implementation framework
- Risk management integration
- Symbol auto-download feature
