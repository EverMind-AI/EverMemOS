.PHONY: setup-hooks uninstall-hooks lint test clean help

# Default target
help:
	@echo "Available targets:"
	@echo "  setup-hooks      - Install pre-commit hooks (pre-commit + commit-msg)"
	@echo "  lint             - Run linters"
	@echo "  test             - Run tests"
	@echo "  clean            - Clean up generated files"
	@echo "  help             - Show this help message"

# Install pre-commit hooks (clean install)
setup-hooks:
	@echo "Removing existing hooks..."
	rm -f .git/hooks/pre-commit .git/hooks/commit-msg
	@echo "Installing pre-commit hooks..."
	pre-commit install --hook-type pre-commit --hook-type commit-msg
	@echo "Done! Pre-commit hooks installed."

# Run linters
lint:
	@echo "Running black..."
	black src/
	@echo "Running i18n check..."
	PYTHONPATH=src python -m devops_scripts.i18n.i18n_tool check

# Run tests
test:
	PYTHONPATH=src pytest tests/

# Clean up
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	rm -rf .pytest_cache 2>/dev/null || true
	@echo "Cleaned up generated files."
