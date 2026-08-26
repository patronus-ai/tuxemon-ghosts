PY := ./.venv/bin/python

.PHONY: check check-fast lint lint-patched types test slow patch unpatch
check: lint lint-patched types test slow
check-fast: lint lint-patched types test

# Applies patches/*.patch to the vendored tuxemon/ clone, in numeric filename
# order (the shell glob already sorts them), producing the tree every other
# target in this Makefile assumes exists. Fails fast: a patch that does not
# apply cleanly stops the loop and the target, rather than silently leaving
# the tree half-patched.
patch:
	cd tuxemon && for p in ../patches/*.patch; do echo "applying $$p"; git apply "$$p" || exit 1; done

# Destroys the applied tree and restores the pristine vendored clone:
# discards tracked-file edits (git checkout -- .) and removes untracked
# files the patch series added (git clean -fd) -- e.g. tuxemon/core/clock.py
# and tuxemon/core/ids.py. The clone's two `git stash` entries are left
# alone; neither command touches the stash.
#
# DESTRUCTIVE: the applied tree is git-ignored by the outer repo, so it is
# NOT otherwise recoverable once this runs -- patches/*.patch are the sole
# durable record, and `make patch` must genuinely restore an identical tree
# from them. This was verified end to end (`make unpatch && make patch &&
# make check`, tree confirmed byte-identical) before this target shipped;
# see docs/STATUS.org.
unpatch:
	cd tuxemon && git checkout -- . && git clean -fd

lint:
	$(PY) -m ruff check tuxghost tests

# `mypy` has `follow_imports = "skip"` for `tuxemon.*` and `ruff check`
# above only scans `tuxghost tests` -- neither ever looks at the vendored
# engine files our patches actually edit. That hole is real: reverting a
# patched call site back to `uuid4()` leaves an unused
# `from tuxemon.core.ids import new_id` import, and nothing catches it.
# Lint exactly the engine files the currently-applied patches touch,
# derived from `git -C tuxemon diff`/untracked status rather than a
# hardcoded list, so this stays correct as patches are added or changed.
# No mypy gate here on purpose: upstream is unannotated and `--strict`
# would drown in pre-existing findings that are not ours to fix.
lint-patched:
	@files=$$( { git -C tuxemon diff --name-only -- '*.py'; git -C tuxemon ls-files --others --exclude-standard -- '*.py'; } | sort -u ); \
	if [ -z "$$files" ]; then \
		echo "lint-patched: no patched .py files found in tuxemon/ -- are the patches applied? (cd tuxemon && for p in ../patches/*.patch; do git apply \$$p; done)"; \
		exit 1; \
	fi; \
	echo "$$files" | sed 's|^|tuxemon/|' | xargs $(PY) -m ruff check

types:
	$(PY) -m mypy tuxghost tests

test:
	PYTHONHASHSEED=0 $(PY) -m pytest -q

slow:
	# exit 5 means "no tests collected", which is correct until Task 6 adds
	# the first slow test. Any other non-zero status is a real failure.
	PYTHONHASHSEED=0 TUXGHOST_RUN_SLOW=1 $(PY) -m pytest -q -m slow || [ $$? -eq 5 ]
