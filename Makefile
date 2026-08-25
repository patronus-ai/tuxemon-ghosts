PY := ./.venv/bin/python

.PHONY: check check-fast lint types test slow
check: lint types test slow
check-fast: lint types test

lint:
	$(PY) -m ruff check tuxghost tests

types:
	$(PY) -m mypy tuxghost tests

test:
	PYTHONHASHSEED=0 $(PY) -m pytest -q

slow:
	# exit 5 means "no tests collected", which is correct until Task 6 adds
	# the first slow test. Any other non-zero status is a real failure.
	PYTHONHASHSEED=0 TUXGHOST_RUN_SLOW=1 $(PY) -m pytest -q -m slow || [ $$? -eq 5 ]
