.PHONY: setup capture core test lint serve record doctor

# Build the capture helper and install the Python core with local transcription.
setup: capture core

capture:
	cd capture && swift build -c release

core:
	cd core && uv sync --extra local

test:
	cd core && uv run pytest

lint:
	cd core && uv run ruff check . && uv run ruff format --check .

serve:
	cd core && uv run ouvinte serve

record:
	cd core && uv run ouvinte record

# Checks providers, records 5 seconds to confirm both tracks have signal, and
# round-trips a throwaway note and reminder through Notes and Reminders.
doctor:
	cd core && uv run ouvinte doctor --capture-test --export-test
