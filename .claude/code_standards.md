# Python Code Standards

## Readability and Scalability
- Write clean, maintainable Python code that scales well
- Use clear, descriptive variable and function names that explain their purpose
- Follow consistent naming conventions throughout the codebase
- Structure code logically with proper separation of concerns
- Keep functions focused on a single responsibility

## Comments Policy
- **Minimize comments** - code should be self-explanatory
- Only add comments when explaining complex logic that cannot be made clear through code alone
- Prefer refactoring unclear code over adding explanatory comments
- Document non-obvious business rules or algorithm rationale when necessary

## Anti-Overengineering Rules

- Make only the minimal changes necessary. A bug fix does not need surrounding cleanup. A simple feature does not need extra configurability.
- Do not add abstractions until there are at least 3 concrete use cases for them.
- Do not add docstrings, comments, or type annotations to code you didn't change.
- Do not add error handling, fallbacks, or validation for scenarios that cannot happen in this codebase.
- If a solution requires a comment to explain it, simplify the code instead.
- Functions do one thing. If you find yourself writing "and" to describe what a function does, split it.
- Maximum function length: 30 lines. Flag anything longer rather than silently splitting it.
- Nesting depth limit: 3 levels. Use early returns to flatten conditionals.
- Prefer existing utilities and patterns in this codebase over introducing new ones.
- Think harder and examine the codebase for existing infrastructure before proposing new architecture.

## Code Quality Standards
- **Ensure logical soundness** - all code must be logically correct and well-reasoned
- **Conform to existing patterns** - follow the project's established:
  - Code structure and organization
  - Naming conventions
  - Configuration approaches
  - Design patterns and practices
- **Verify executability** - always double-check that code will run without errors
- Test edge cases and validate assumptions before committing

## Before Submitting Code
1. Verify the code follows existing project conventions
2. Ensure all dependencies and imports are correct
3. Confirm the code is logically sound and handles edge cases
4. Test that the code actually runs without errors
5. Check that variable/function names are clear and consistent with the codebase
6. **Run unit tests** - Always run relevant unit tests before submitting code

## Strategy Registry Maintenance

**CRITICAL**: Whenever you add a new strategy OR make significant changes to an existing strategy's parameters, you MUST update the strategy documentation file.

### Files to Update

When adding or modifying strategies, update these files:

1. **`backtest_scripts/LIST_ALL_STRATEGIES.bat`** (Windows)
2. **`backtest_scripts/LIST_ALL_STRATEGIES.sh`** (Linux/Mac)

These files serve as the **user-facing catalog** of all available strategies and their parameters.

### What Constitutes a "Significant Change"

Update the strategy list when:

1. **Adding a new strategy** to `src/strategies/`
2. **Adding new parameters** to an existing strategy
3. **Changing default parameter values** that users should know about
4. **Adding optional features** (e.g., volatility filter, ATR stop)
5. **Changing strategy behavior** in a user-visible way
6. **Removing or deprecating** a strategy

### What to Include

For each strategy, document:

- **Strategy name** (exact class name for CLI usage)
- **Brief description** (one line explaining the strategy logic)
- **Parameters** (list all configurable parameters)
- **Default values** (what users get if they don't specify parameters)
- **Special requirements** (e.g., "Requires 2 symbols" for PairsTrading)
- **Category** (MA, Momentum, Mean Reversion, etc.)

### Example Format

```batch
echo   7. BreakoutStrategy (Enhanced)
echo      - Price breakout strategy with advanced filters
echo      - Parameters: breakout_window, exit_window, volatility_filter,
echo                    volume_confirmation, use_atr_stop
echo      - Default: 20/10 windows, filters disabled
echo      - Optional: Volatility filter, volume confirmation, ATR trailing stop
```

### Workflow

When implementing or modifying a strategy:

1. **Write the strategy code** in `src/strategies/`
2. **Add to `__init__.py`** in the appropriate strategies module
3. **Update LIST_ALL_STRATEGIES.bat and .sh** with the new/changed strategy
4. **Commit all three changes together** in the same commit

This ensures the strategy catalog stays synchronized with the actual codebase.

### Why This Matters

- Users run `LIST_ALL_STRATEGIES.bat` to see what's available
- Outdated strategy lists cause confusion about what parameters exist
- Strategy catalog is the **first place users look** to understand capabilities
- Accurate documentation prevents support questions and errors
