# tuxemon-ghosts

A deterministic trace format and offline replay executor for Tuxemon. This
file is the entry point for any session working in this repo — and, per
the rule below, is the one file in this project allowed to be Markdown.

## Read this first

**Read `docs/STATUS.org` before anything else** — before the spec, before
any plan, before any task report under `.superpowers/sdd/`. It states
what is true of the tree *right now*, measured, with real digests. Those
other documents state what was *intended*, or what was true when they
were written; the tree has moved since.

**Authority order when documents disagree**: `docs/STATUS.org`'s own
measured claims (re-run them if you doubt them) outrank the `docs/*.org`
measurement notes, which outrank the spec, which outrank the plan. A plan
is a proposal for how to get somewhere; a spec is a binding statement of
what the format and its guarantees are; a measurement is what actually
happened when someone ran the thing; `STATUS.org` is the current
synthesis of all of it. If a plan step and a measurement conflict, the
measurement wins and the plan was wrong. Update `STATUS.org` before
trusting your memory of an earlier task's report.

## Documentation is org-mode

Every doc in this repo is `.org`, never `.md` — `docs/STATUS.org`, specs,
plans, measurement notes, all of it. `CLAUDE.md` is the one named
exception (Claude Code reads it as Markdown specifically). If you're
about to `Write` a `.md` file anywhere else in this repo, stop; you
almost certainly want `.org`.

## The trace format

Format version 1. `format_version` mismatch is an outright refusal —
never downgradable, never guessed at, regardless of `allow_mismatch`
(see `tuxghost/trace.py`'s module docstring for the full refuse/warn
matrix; several other header fields warn-and-continue, but
`format_version` never does). Traces are **step-indexed, never
wall-clock-indexed**: `PlayerInput.timestamp` (upstream's `time.time()`
default) deliberately never enters the format, and `clock_epoch` is a
single pinned instant interpreted in UTC, not a stream of real
timestamps. Anything that reads real wall-clock time to decide what
happened next is a determinism bug, not a feature.

## Never run the game without dummy SDL drivers

Every invocation of anything that touches `tuxemon`'s client — tests,
manual scripts, a REPL — must run with:

```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy
```

Without both, a headless run can block on window/audio device creation
or behave differently machine to machine, which is exactly the class of
nondeterminism this whole project exists to eliminate.

## Never edit `tuxemon/` directly

Every engine change is a patch file under `patches/`, applied with
`make patch` (see `docs/STATUS.org` for what each of the six patches
does). The applied `tuxemon/` tree is **git-ignored** — nothing under it
is ever committed directly, and `make unpatch` destroys it without
recovery. The patch files are the *sole durable record* of every engine
change this project has made. If you find yourself editing a file under
`tuxemon/` and it isn't through regenerating a patch, stop: that edit
will vanish the next time someone runs `make unpatch`, and worse, nobody
downstream will ever know it existed.

To change engine behavior: edit the applied tree, confirm the change
does what you want, then regenerate the affected patch file from a
pristine-vs-edited diff (see the patch files themselves and the git
history for the mechanics — plan workspaces under `.superpowers/sdd/` are
deleted at each subsystem's close-out, so do not expect past task reports
to still be there; `docs/STATUS.org` and the commits are the durable
record — in particular,
regenerate patches from real file snapshots, not through text-mode
Python I/O, which has silently corrupted CRLF-native vendored files in
this project before). Verify the regenerated series still applies
cleanly from pristine (`make unpatch && make patch`) before trusting it.

`tuxemon/` also carries two `git stash` entries from the original vendor
clone. Leave them alone — `make patch`/`make unpatch` never touch the
stash, and neither should you.

## `EXEMPTIONS` must stay `{}`

`tuxghost/digest.py`'s `EXEMPTIONS` dict is a list of known-unfixed
nondeterminism, not a list of fields that are allowed to vary. Its
target size is zero, and it is currently `{}`. Do not add an entry to
make a failing test pass. If a field genuinely diverges and you believe
it's acceptable to exempt, that is a decision for a human, argued in the
open, not a quiet fix — and even then, `tests/test_digest.py`'s
`test_exemptions_each_name_a_defect` and
`test_exemptions_are_all_reachable_in_the_digested_tree` mean an entry
must name a real reason and land on a path the digest can actually
reach; a dead or unreasoned entry is caught, not silently accepted.
`state_of()` digests THREE subtrees — `npc_state`, `world_state`, and
`persistent_npc_state` — and the reachability test walks all three, so
an entry under any of them, including `persistent_npc_state.*`, is a
legitimate, checkable path.

## The non-negotiable: a regression test must be demonstrated to fail against the bug it pins

Writing an assertion that happens to be true right now is not the same
as writing a test that would catch the thing going wrong. Every
regression test in this project must be **shown** to fail against the
defect it claims to pin — mutate the code to reintroduce the bug, run
the test, watch it fail with a real error message, restore the code,
run it again, watch it pass — and the real "before" and "after" output
belongs in the report, not a description of what you expect it would
show.

This is not caution for its own sake. On this project, six tests
already passed *vacuously* — green, but proving nothing — before being
caught, each in a different way:

- One compared an object to itself: `local_session` is a module-level
  singleton, so two variables that looked like "session A" and
  "session B" were the literal same object by the time the assertion
  ran, making an `x == x`/`x is x` check that read as verification while
  checking nothing.
- One primed the very session singleton it was supposed to be testing,
  inside its own setup: a `boot_from_save` clock-reset test called
  `build_client()` to produce its fixture, which — as a side effect —
  already settled `local_session`'s clock state to the value under
  test, so `boot_from_save`'s own clock-handling code could be dead-coded
  entirely and the test still passed, riding for free on the fixture's
  own setup rather than exercising the code it named.
- One digest was "deterministic" across nine runs — several of them
  unseeded, across three different seeds — for the worst possible
  reason: the earliest version of the state digest never captured
  anything that actually changed. The probed route never moved the
  player, and the digest was blind to the state stack, the only thing
  changing, so nine runs all hashed to the same value and read as proof
  of determinism. Two controls exposed it: instrumenting `random`
  itself, which revealed 152 live RNG calls the digest never saw, and
  requiring digests to diverge across different seeds, which only
  started happening once party HP actually varied between runs (see
  `docs/2026-08-25-determinism-spike.org`).
- One digest test's pass depended entirely on what time of day someone
  happened to run the suite: `tests/test_digest.py`'s `_run()` predated
  `pin_clock` and never called it, so it silently read real wall-clock
  time on every run. During a single working session, real local time
  crossed into night partway through, and the run crashed (`AttributeError: 'NullRenderer' object
  has no attribute 'layer'`, a pre-existing headless-rendering gap the
  night-only `set_layer` event action exposed). Nothing about the
  test's assertions had changed; only the wall clock had. Fixed by
  pinning `_run()` to an explicit epoch, the same discipline every other
  clock-dependent test in this project needed.
- A CLI path test passed only because it never used a relative path:
  every path it exercised was an absolute `tmp_path` fixture path, so a
  real bug — the process `chdir`'d into the vendored `tuxemon/`
  directory *before* parsing arguments, silently breaking every relative
  trace path a real user would type — went undetected until a reviewer
  deliberately tried one.
- One asserted only that a string was non-empty: the exemption-register
  test's `assert reason.strip()` accepted any placeholder text at all as
  a valid reason to exempt a field from the digest, so it would have
  waved through `"asdf"` exactly as happily as a real justification.
  Strengthened to `re.search(r"patch \d{4}", reason)` — a reason must
  name the patch that actually closes the defect — after a real,
  unrelated bug (an unfiltered `uuid4()` nonce silently entering the
  digest) slipped past both round-0 tests undetected.

Beyond the tests: **the implementation plan itself was the source of six
defects** that would each have produced a fully green suite while the
feature quietly did not work — a design sketch is not verified just
because it reads plausibly. Treat every plan step the same way: verify
it against the real code before trusting it, not after.

When you finish a task, the report needs real command output for this —
not "this would fail if reverted," but the actual failing assertion,
the actual restored-and-passing run afterward, and a diff or `git
status` confirming the file was genuinely restored, not left mutated.

## Running the gates

- `make check-fast` — lint, `lint-patched`, `mypy --strict`, and the fast
  pytest suite. Use this for iteration.
- `make check` — everything in `check-fast` plus the `slow` suite (gated
  behind `TUXGHOST_RUN_SLOW=1`, full battles, seconds to minutes).
  Required before calling anything done.
- `make lint-patched` exists because the other two lint/type gates have a
  real hole: `mypy` has `follow_imports = "skip"` for `tuxemon.*` (upstream
  is unannotated and not ours to fix), and `ruff check tuxghost tests`
  never looks inside `tuxemon/` at all. Neither gate would catch, say, a
  patch reverted back to `uuid4()` leaving an unused `new_id` import.
  `lint-patched` lints exactly the vendored files the *currently applied*
  patch series touches (derived from `git -C tuxemon diff`/untracked
  status, not a hardcoded list), so it's correct as patches change.

**Do not run two full test suites concurrently.** The suite depends on
shared, mutable state outside the repo — `~/.tuxemon/cache/l18n/*.mo` —
which a race between two concurrent `make check` runs has corrupted
before. Run one at a time, sequentially.

## `patch` / `unpatch`

`make patch` applies `patches/*.patch` to the vendored `tuxemon/` clone,
in numeric order, failing fast on the first patch that doesn't apply
cleanly. `make unpatch` restores the pristine vendored tree
(`git checkout -- .` + `git clean -fd` inside `tuxemon/`) — **this is
destructive**: the applied tree is git-ignored, so anything `unpatch`
throws away is gone unless `patch` can rebuild it from the patch files
alone. Both were run end to end (`make unpatch && make patch && make
check`) and the rebuilt tree confirmed byte-identical to the tree before
`unpatch` ran — see `docs/STATUS.org` for the result.
