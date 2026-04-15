# Project Structure and File Organization

**CRITICAL**: Maintain clean project organization with proper separation of concerns.

## Root Directory Rules

**NO Script Files in Root Directory:**
- The project root directory MUST NOT contain script files (`.py`, `.bat`, `.sh`)
- All scripts belong in appropriate subdirectories:
  - `src/` - Source code (backtesting, strategies, data engine, GUI)
  - `tests/` - Test files (unit tests, integration tests)
  - `scripts/` - Utility scripts, launchers, and test runners (`.py`, `.bat`, `.sh`)
  - `examples/` - Example code and tutorials
  - `backtest_scripts/` - Pre-configured batch files for running specific backtests

**Allowed Root Files:**
- Configuration files (`.ini`, `.yaml`, `.toml`)
- Documentation (`.md`)
- Requirements files (`requirements.txt`, `environment.yml`)
- Project metadata (`setup.py`, `pyproject.toml`)
- Git files (`.gitignore`, `.gitattributes`)
- License and README
- Makefiles (`Makefile`)
- `conftest.py` - pytest configuration (required by pytest to be in root)

**Explicitly NOT Allowed in Root:**
- Python scripts (`.py`) - Exception: `conftest.py` for pytest
- Shell scripts (`.sh`)
- Batch scripts (`.bat`)
- Test files
- Temporary files

## Documentation Organization

**Specialized Documentation Must Be Co-located:**
- GUI documentation -> `src/gui/docs/`
- Data engine documentation -> `src/data_engine/docs/`
- General framework documentation -> `docs/`

**Never Create Documentation in Root:**
- [-] `GUI_USER_GUIDE.md` in root
- [+] `src/gui/docs/USER_GUIDE.md`

## Script Organization

**Launcher Scripts:**
- Place in `scripts/` directory
- Examples:
  - `scripts/run_gui.py` - GUI launcher
  - `scripts/run_gui_tests.bat` - Test runner (Windows)

**Test Scripts:**
- Python test files -> `tests/` or `tests/integration/`
- Batch test runners -> `scripts/`
- Examples:
  - `tests/integration/test_phase1_backend.py` - Integration test
  - `scripts/run_gui_tests.bat` - Test runner batch file

**Backtest Scripts:**
- Pre-configured backtest batch files -> `backtest_scripts/`
- Organized by category:
  - `backtest_scripts/basic/` - Simple strategies
  - `backtest_scripts/intermediate/` - More complex
  - `backtest_scripts/advanced/` - Advanced strategies

## File Naming Conventions

**Python Modules:**
- Use `snake_case.py`
- Be descriptive: `strategy_utils.py` not `utils.py`

**Test Files:**
- Prefix with `test_`: `test_phase1_backend.py`
- Match the module they test: `test_gui_controller.py`

**Documentation:**
- Use `UPPER_SNAKE_CASE.md` for guides: `USER_GUIDE.md`
- Use descriptive names: `IMPLEMENTATION_PLAN.md`

## Temporary Files and Cleanup

**DO NOT Leave Temporary Files**

**Common temporary file patterns to delete**:
- `*.tmp`, `*.tmp.*` - Temporary files from various operations
- `*.bak`, `*.backup` - Backup files from editors or tools
- `*~` - Editor backup files (vim, emacs)
- `*.swp`, `*.swo` - Vim swap files
- `nul` - Windows null device file accidentally created
- `requirements.txt.tmp.*` - Pip temporary files

**How to clean up**:
1. After completing any operation that generates temp files, delete them immediately
2. Check for orphaned temp files periodically: `find . -name "*.tmp*" -o -name "*.bak"`
3. Never commit temp files to the repository

## IDE Configuration Files

**pyrightconfig.json**:
- This is a Pyright/Pylance (VSCode) type checker configuration
- It's IDE-specific and user-preference dependent
- **Added to .gitignore** - each developer should configure their own

**Other IDE files to keep out of repo**:
- `.vscode/` - VSCode settings (in .gitignore)
- `*.code-workspace` - Workspace files (in .gitignore)
- `.idea/` - PyCharm settings (add if you use PyCharm)
- `*.sublime-project`, `*.sublime-workspace` - Sublime Text (add if needed)

**Exception**: Shared team IDE settings can be committed if the entire team uses the same IDE and agrees on settings.
