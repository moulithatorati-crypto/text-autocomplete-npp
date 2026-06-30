# Contributing to NPP Text Auto-Completion

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR-USERNAME/text-autocomplete-npp.git`
3. Create a branch: `git checkout -b feature/your-feature`
4. Set up development environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   pip install -e .
   ```

## Code Style

### Python Style Guide

- Follow PEP 8
- Use type hints for all functions
- Maximum line length: 100 characters
- Use docstrings for all public functions/classes

### Code Formatting

```bash
# Format code
black . --line-length 100

# Check linting
flake8 . --max-line-length 100

# Type checking
mypy . --ignore-missing-imports
```

## Testing

```bash
# Run tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=. --cov-report=html
```

## Pull Request Process

1. Update README.md if needed
2. Add tests for new functionality
3. Ensure all tests pass: `pytest`
4. Format code: `black . --line-length 100`
5. Write clear PR description
6. Link related issues

## Reporting Bugs

### Before Submitting a Bug

- Check existing issues
- Reproduce with latest code
- Collect debug information

### Bug Report Template

```markdown
**Environment:**
- OS: [Windows/Linux/Mac]
- Python: [3.11/3.12]
- PyTorch: [version]
- GPU: [if applicable]

**Description:**
[Clear description of the bug]

**Reproducer:**
[Minimal code to reproduce]

**Expected behavior:**
[What should happen]

**Actual behavior:**
[What actually happens]
```

## Feature Requests

Use GitHub Issues with the `enhancement` label. Include:
- Clear description of the feature
- Use cases / motivation
- Possible implementation approach

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
