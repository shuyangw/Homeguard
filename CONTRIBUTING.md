# Contributing to Homeguard

Thank you for your interest in contributing to Homeguard!

## Getting Started

1. Fork the repository
2. Set up your environment (see [SETUP.md](SETUP.md))
3. Create a feature branch from `main`

## Development Guidelines

- Follow the coding standards in [CLAUDE.md](CLAUDE.md)
- Run tests before committing: `make test` or `pytest tests/`
- Update documentation for user-facing changes
- Use the `fintech` conda environment for all Python operations

## Code Standards

- Write self-explanatory code with minimal comments
- Use descriptive naming for functions and variables
- Follow existing patterns in the codebase
- Never hardcode sensitive data (API keys, IPs, etc.) - use `.env` files

## Testing

Before submitting changes:

```bash
# Run all tests
pytest tests/

# Run specific test module
pytest tests/engine/test_backtest_engine.py
```

## Pull Request Process

1. Update README/docs if your changes affect user-facing functionality
2. Run the full test suite and ensure all tests pass
3. Create a PR with a clear description of changes
4. Reference any related issues

## Documentation

When adding or modifying features:
- Update relevant docs in `docs/`
- Update `docs/architecture/` for structural changes
- Use timestamp prefixes for new docs: `YYYYMMDD_filename.md`

## Questions?

Check the [documentation hub](docs/README.md) or open an issue for discussion.
