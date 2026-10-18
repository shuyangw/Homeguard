# Documentation and Examples

## User-Facing Functionality Requirements

When creating or modifying user-facing functionality, you MUST update all relevant documentation and examples:

**Always Update**:
1. **README files** - Update any README.md files that describe the changed functionality
2. **Example scripts** - Update or create example scripts in `examples/` or `backtest_scripts/`
3. **API documentation** - Update `docs/API_REFERENCE.md` if APIs changed
4. **User guides** - Update relevant guides in `docs/` directory
5. **Quick start sections** - Update quick start commands if workflow changed
6. **Templates** - Update template files or code snippets that users copy

**Common Documentation Locations**:
- `README.md` - Main project documentation
- `backtest_scripts/README.md` - Backtest script documentation
- `src/data_engine/README.md` - Data engine documentation
- `docs/BACKTESTING_GUIDE.md` - Comprehensive backtesting guide
- `docs/API_REFERENCE.md` - API documentation
- `SETUP.md` - Setup and configuration instructions

**Examples to Update**:
- Code snippets in README files
- Batch scripts in `backtest_scripts/`
- Python examples in `examples/`
- Template scripts users might copy
- Command-line usage examples

**Why This Matters**:
- Users rely on documentation to understand how to use features
- Outdated documentation causes confusion and support burden
- Examples serve as templates for user implementations
- Consistency between code and docs is critical for user experience

## Documentation Standards

- Keep examples concise and focused on one concept
- Use realistic, relatable examples (common stocks, standard date ranges)
- Include expected output or results when relevant
- Update file paths if project structure changes
- Verify all commands and code snippets actually work
- Use consistent formatting across all documentation

## DO NOT Create Summary Documentation in docs/

**IMPORTANT**: Do NOT proactively create summary or completion documentation files in the root `docs/` folder.

**Never create files like** (in root `docs/`):
- `*_SUMMARY.md`
- `*_COMPLETE.md`
- `*_IMPLEMENTATION_SUMMARY.md`
- `CHANGES_SUMMARY.md`

**Why**:
- Creates documentation bloat in main docs folder
- Summary docs become stale quickly
- User can see changes via git history
- Adds unnecessary files to maintain

**Exception**:
- [+] **Progress docs in `docs/progress/` are REQUIRED** (see Progress Documentation System section)
- Technical architecture docs (`docs/ARCHITECTURE.md`) are fine when they document system design, not changes
- ONLY create summary docs in root `docs/` if explicitly requested by the user

**Instead of creating summaries in root docs**:
- Create timestamped progress docs in `docs/progress/` for large features (see next section)
- Communicate changes directly to the user in conversation
- Update existing README or docs to reflect new functionality

## Progress Documentation System

**CRITICAL**: When completing substantial feature implementations or large bodies of work, create progress documentation to track completion for multi-environment AI-assisted development.

### When to Create Progress Docs

Create a progress doc when:
1. **Major Feature Implementation Complete**: A significant new feature is fully implemented and tested (e.g., multi-symbol portfolio system, risk management framework)
2. **Large Integration Complete**: A major component is integrated into the system (e.g., GUI integration with new backend system)
3. **Major Performance Optimization**: Significant performance improvements that required substantial code changes
4. **Large Refactoring Complete**: Large-scale cleanup or architectural refactoring across multiple modules
5. **Implementation Status Update**: Tracking ongoing implementation progress for multi-phase features (updated periodically)

### When NOT to Create Progress Docs

[-] **Do NOT create progress docs for**:
- Small bug fixes (one-file changes, < 50 lines)
- Minor code adjustments or tweaks
- Simple parameter changes
- Documentation-only updates
- Small utility functions
- Single-line fixes or typo corrections

**Rule of Thumb**: Only create progress docs when the work represents **substantial effort** (multiple days, multiple files, significant complexity, or multiple related changes).

### Where to Create Progress Docs

**Location**: `docs/progress/`

All progress documentation goes in this dedicated folder, **not** in the root `docs/` directory.

### Naming Convention

**Format**: `YYYYMMDD_<TOPIC>.md`, completion date, topic in UPPER_SNAKE_CASE (e.g. `20261008_CONSOLE_PHASE2A.md`, `20260711_FUTURES_RETEST.md`). Append `_STATUS` for a doc tracking an ongoing multi-phase feature that gets updated as it goes.

### Progress Doc Template

```markdown
# <Topic> - YYYY-MM-DD

## Summary
1-3 sentence overview of what was accomplished.

## Changes Made
- **File/area**: what changed and why (bulleted, concise)

## Commits
- `<short-hash>` <commit message>

## Known Issues / Remaining Work
- What's left, what broke, what needs follow-up

## Validation
- How the changes were verified (tests run, manual checks, EC2 deploy)
```

Keep it factual; include commit hashes; focus on context and decisions the commit messages do not capture.

### What NOT to Put in Progress Docs

[-] **Do NOT create progress docs for**:
- User-facing guides (goes in `docs/`)
- API references (goes in `docs/`)
- How-to tutorials (goes in `docs/`)
- General documentation (goes in `docs/`)
- Improvement roadmaps/plans (goes in `docs/`)

### Multi-Environment Context

**Purpose**: When switching between machines or AI assistants, progress docs help maintain context.

**AI Usage**:
Session-to-session continuity comes from the message log and handoffs in the main checkout (`_claude-code-logs/`, `_handoffs/`; see CLAUDE.md "Message log" and "Session handoffs"). Progress docs are for substantial features only:
1. Check STATUS docs for ongoing work
2. Check VALIDATION docs to see what's been tested

### Updating Progress Docs

**For STATUS docs**: Update periodically as implementation progresses
- Add new completion dates
- Update status ( -> [+])
- Track blockers or challenges

**For other types**: Generally write once when complete, don't update
- Exception: If validation reveals issues, update VALIDATION doc

### Cleanup Policy

**Old Progress Docs**: Archive progress docs older than 1 year
- Move to `docs/progress/archive/YYYY/`
- Keep recent progress for context
- Archive preserves history

**Example**:
```
docs/progress/
├── archive/
│   └── 2023/
│       └── 20230615_OLD_FEATURE_STATUS.md
├── 20241102_RISK_MANAGEMENT_VALIDATION.md
└── 20241115_MULTI_SYMBOL_STATUS.md
```
