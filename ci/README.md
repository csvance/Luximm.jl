# Luximm.jl self-hosted CI

`jimm-ci` is an interactive Julia TUI that runs the Luximm.jl parity test
suite on a self-hosted Linux VM and reports results back to GitHub via
the Checks API. There is **no webhook listener, no public endpoint, and
no domain name involved**, every run is started by the maintainer, by
hand, over SSH.

## Why a custom, self-hosted runner

Hosted CI providers (GitHub Actions, Buildkite cloud, etc.) are the
obvious default, and were ruled out on purpose. Luximm.jl's test suite is
a *parity* suite: every variant of every backbone is checked against
the corresponding `timm` PyTorch reference, which means each run needs
two large, expensive-to-produce artifacts:

1. **Reference weights.** Luximm covers many ResNet, ViT, and adjacent
   variants. The matching `timm` checkpoints on HuggingFace add up to
   hundreds of gigabytes across the supported variant set. Pulling
   them fresh on every CI run would saturate egress, blow past
   ephemeral-runner disk quotas, and burn HuggingFace bandwidth for no
   added signal.
2. **Parity fixtures.** Each variant's parity test consumes an HDF5
   fixture produced by running the `timm` model under PyTorch and
   dumping inputs, intermediate activations, and outputs. Generating
   one fixture takes a few minutes of CPU and a PyTorch + timm
   environment (a couple of gigabytes of wheels). Doing this from
   scratch per run would dominate wall time and make the CI feedback
   loop unusable.

The self-hosted runner keeps both artifacts on a persistent state
directory (`<state>/parity/` for fixtures, `<state>/hf-cache/` for
weights, `<state>/python-env/` for the PyTorch venv). The first run
for any new variant pays the dump cost once; every later run, on any
PR or master commit, is a cache hit. Combined with the path-filtered
job routing in `PathFilter.jl`, a typical PR run only re-tests the
families it actually touched, against fixtures that already exist on
disk.

The trade-off is operational: one VM the maintainer has to maintain,
and per-PR human approval before any code from a contributor runs on
it. The `jimm-ci` TUI exists to make that approval step a single
keystroke rather than a chore.

The same binary covers four jobs:

* **Default (`jimm-ci`)** launches a TUI showing every PR and recent
  master commit eligible for CI. The maintainer picks one, watches the
  combined stdout/stderr stream live in the same pane, and may cancel
  the build or schedule a back-to-back run of every pending master
  commit.
* **`jimm-ci --dry-run`** prints the discovered jobs and exits.
* **`jimm-ci --master`** and **`jimm-ci --sha <sha>`** force a full
  sweep against a specific commit, bypassing discovery and the
  already-tested filter. Used to re-test a commit after fixing a bug in
  the runner.
* **`jimm-ci --skip-pending`** posts `conclusion=skipped` Check Runs on
  every currently pending master commit, clearing a backlog. PRs are
  never auto-skipped, decline a PR by pressing `s` on its row in the
  TUI instead.

```
                 ┌──────────────────────────────────────┐
                 │ GitHub                               │
                 │   ├── Pulls / Commits / Compare      │
                 │   └── Checks API                     │
                 └────────────┬─────────────────────────┘
                              │ HTTPS (outbound only)
                  ┌───────────▼─────────────┐
                  │ jimm-ci (Julia TUI)     │
                  │  launched over SSH      │
                  │  git mirror + worktree  │
                  │  julia Pkg.test         │
                  └─────────────────────────┘
```

## TUI

On launch, `jimm-ci` performs an eager discovery (open PRs in the repo
plus master commits from the last 30 days that have no completed
`jimm-ci` Check Run) and presents the result sorted newest first:

```
 ┌──────────────────────────── jimm-ci ──────────────────────────────┐
 │ 4 pending                                                          │
 │┌── queue ──────────────────────────────────────────────────────────│
 ││   2026-05-21 09:14  PR      #42 [owner/Luximm.jl]   Add InceptionNeXt   [infra,bit,…]
 ││ ⚠ 2026-05-21 08:50  PR      #43 [contrib/Luximm.jl] Fix typo            [infra]
 ││   2026-05-21 04:21  master  master @ 3aa12b3                          [infra,bit,…]
 ││   2026-05-20 14:02  master  master @ 097de48                          [infra,bit,…]
 │└───────────────────────────────────────────────────────────────────│
 │  4 job(s) pending    [↑↓/jk] move  [Enter/y] run  [A] run all master  [s] skip  [r] refresh  [q] quit │
 └────────────────────────────────────────────────────────────────────┘
```

The `⚠` glyph and warning-coloured row mark a PR whose head branch lives
in a fork. Pressing `Enter` or `y` on a fork row opens a confirmation
modal that names the fork and head SHA; the build only starts after a
second `y`. Same-repo PRs and master commits skip the modal. The
`[owner/repo]` column is always shown, including for non-fork PRs, so
the source of every commit is explicit.

### Keybindings

| Key | List view | Running view |
| --- | --- | --- |
| `↑` / `↓`, `j` / `k` | move cursor | scroll log pane |
| `Enter` / `y` | run the selected job | — |
| `A` | enqueue the selected master commit **and every older pending master commit**, then drain serially | — |
| `s` | post `skipped` Check Runs for the selected commit and drop it | — |
| `r` | refresh the list (background fetch; UI stays responsive) | — |
| `q` / `Esc` | quit | — |
| `c` | — | cancel the current build (modal confirm) |
| `C` | — | cancel the current build **and** the back-to-back queue |
| `PgUp` / `PgDn` | — | scroll the log pane |

In the fork-confirm modal (opened automatically when `Enter` / `y` is
pressed on a fork PR row), `y` runs the build and `n` / `Esc` dismisses
the modal without running.

When a job is running, the same pane switches to a live, auto-following
log view. Output is streamed line-by-line as the Julia subprocess writes
to stdout/stderr; the full log is also persisted to
`<state>/logs/<sha>/<family>.log`.

PR approval and `n`-equivalent decline both happen by selecting the PR
row and pressing `y` or `s` respectively. Master commits are not gated,
merged code is presumed reviewed at merge time, but the TUI still
requires a per-commit confirmation (one keystroke each, or use `A` to
drain them back-to-back).

## Layout

```
ci/
└── JimmCI/                  # Julia package, drives the runner end-to-end
    ├── Project.toml         # Tachikoma / HTTP / JSON3 / JSONWebTokens pins
    ├── bin/jimm-ci          # thin shell wrapper used over SSH
    ├── src/
    │   ├── JimmCI.jl        # module root + CLI entry point
    │   ├── Config.jl        # env-var-driven configuration
    │   ├── GitHubApp.jl     # JWT + installation token + Checks/Compare/Pulls
    │   ├── Jobs.jl          # Job struct + discovery (open PRs, recent master)
    │   ├── PathFilter.jl    # changed paths → test families
    │   ├── Builder.jl       # git worktree + Pkg.test + Check Run lifecycle
    │   ├── SkipMarker.jl    # post `skipped` Check Runs
    │   └── Tui.jl           # Tachikoma.jl model/update/view
    └── test/
        ├── runtests.jl
        └── test_path_filter.jl
```

## What the runner does

On every TUI launch (or `--dry-run` invocation), `jimm-ci`:

1. Lists open PRs, including fork PRs (marked with a `⚠` glyph in the
   TUI and gated behind a fork-confirm modal). For each, calls Compare
   to map changed paths → test families, then queries the Checks API
   for the head commit and keeps only families with no completed
   `jimm-ci / <family>` Check Run yet. PR-scope jobs use
   `REPRESENTATIVE_VARIANT` per family.
2. Lists master commits in the last 30 days; for each commit missing
   any `jimm-ci` Check Run, queues a full-sweep job.
3. Presents the union in the TUI, sorted newest-first.

When the user picks a job, the builder:

* posts an `in_progress` Check Run for each family;
* dumps the family's timm parity fixture(s) into a persistent
  `<state>/parity/` directory (symlinked into the worktree's
  `data/parity/`) via `uv run python test/parity/dump_<family>_io.py …`
  if the fixture is missing, plus, for the five families with a feature
  pyramid, the `<variant>_featsonly_io.h5` fixtures via the shared
  `test/parity/dump_features_only_io.py` sidecar;
* runs `julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'`
  inside a detached `git worktree` with `JIMM_TEST_FAMILIES` /
  `JIMM_TEST_VARIANTS` set;
* streams stdout/stderr to the TUI **and** to a per-family log file
  under `<state>/logs/<sha>/<family>.log`;
* completes the Check Run with the tail of the build log (capped at
  ~60 KB to stay under the GitHub limit).

The parity fixtures and the Python venv that produces them live under
`<state>/` (`parity/` and `python-env/`), so the first run for a given
variant pays the dump cost (a few minutes) and every subsequent run is
a cache hit.

## Configuration

All configuration is environment-driven. Secrets are loaded from files,
not env strings.

| Variable | Required | Default | Purpose |
| --- | --- | --- | --- |
| `JIMM_CI_APP_ID` | yes | — | GitHub App ID |
| `JIMM_CI_INSTALLATION_ID` | yes | — | App installation on the repo |
| `JIMM_CI_PRIVATE_KEY_FILE` | yes | — | Path to App PEM |
| `JIMM_CI_REPO_OWNER` | yes | — | e.g. `cvance` |
| `JIMM_CI_REPO_NAME` | yes | — | e.g. `Luximm.jl` |
| `JIMM_CI_STATE_DIR` | no | `/var/lib/jimm-ci` | Mirror, worktrees, logs, depot |
| `JIMM_CI_HF_TOKEN_FILE` | no | — | HuggingFace token for parity weights |
| `JIMM_CI_JULIA` | no | `/usr/local/bin/julia` | Julia binary path |
| `JIMM_CI_UV` | no | PATH lookup, then `/usr/local/bin/uv`, then `~/.local/bin/uv` | `uv` binary path, for the Python parity sidecars |
| `JIMM_CI_PARITY_DIR` | no | `<state>/parity` | Where dumped HDF5 fixtures persist across worktrees |
| `UV_PROJECT_ENVIRONMENT` | no | `<state>/python-env` | Persistent venv for the Python parity sidecars (PyTorch + timm) |
| `JULIA_NUM_THREADS` | no | `4` | Forwarded to test jobs |
| `JIMM_CI_LOG_LEVEL` | no | `INFO` | Set to `DEBUG` for verbose runner logs |

## Setup on Debian 13

Assumes a fresh Debian 13 VM with internet access. Run everything as
`root` unless noted.

### 1. System packages and `ci` user

```bash
apt-get update
apt-get install -y --no-install-recommends \
    build-essential git ca-certificates curl jq tar xz-utils \
    python3 python3-venv

adduser --system --group --home /home/ci --shell /bin/bash ci
```

### 2. Install `uv` (still needed for the Python parity sidecars)

```bash
sudo -u ci bash -lc 'curl -LsSf https://astral.sh/uv/install.sh | sh'
ln -sf /home/ci/.local/bin/uv /usr/local/bin/uv
```

### 3. Install Julia via juliaup

```bash
sudo -u ci bash -lc 'curl -fsSL https://install.julialang.org | sh -s -- --yes'
ln -sf /home/ci/.juliaup/bin/julia /usr/local/bin/julia
```

### 4. Clone the repository, install the runner, create state dirs

```bash
install -d -o ci -g ci /opt/jimm-ci
sudo -u ci git clone https://github.com/<OWNER>/Luximm.jl.git /opt/jimm-ci/Luximm.jl

# Install JimmCI via Pkg.Apps. This drops a launcher into ~ci/.julia/bin/
# (Julia's user app directory) and pulls in Tachikoma/HTTP/JSON3/etc.
sudo -u ci julia -e '
    using Pkg
    Pkg.Apps.develop(path = "/opt/jimm-ci/Luximm.jl/ci/JimmCI")
'

install -d -o ci -g ci \
    /var/lib/jimm-ci \
    /var/lib/jimm-ci/julia-depot \
    /var/lib/jimm-ci/hf-cache \
    /var/lib/jimm-ci/parity \
    /var/lib/jimm-ci/python-env

# Make jimm-ci available on every shell PATH (the alternative is to add
# ~ci/.julia/bin to PATH in the ci user's .profile).
ln -sf /home/ci/.julia/bin/jimm-ci /usr/local/bin/jimm-ci
```

`Pkg.Apps.develop` tracks the cloned checkout, so `git pull` followed by
`jimm-ci` is enough to redeploy — no re-instantiate step.

If you'd rather pin a tagged release than track the cloned checkout,
use `Pkg.Apps.add(url = "https://github.com/<OWNER>/Luximm.jl",
subdir = "ci/JimmCI")` instead. The launcher writes to
`~/.julia/bin/jimm-ci` either way.

The repo ships a thin `ci/JimmCI/bin/jimm-ci` shell wrapper as a
fallback for environments where running `Pkg.Apps.develop` is awkward;
it does `julia --project=... -e 'using JimmCI; JimmCI.cli_main()'` and
honors `JIMM_CI_JULIA` for the Julia binary path, and `JIMM_CI_UV` for
`uv`'s.

The runner creates `mirror.git`, `work/`, and `logs/` under
`/var/lib/jimm-ci/` on first invocation. The parity sidecars need a
Python environment with PyTorch + timm; `uv` provisions it lazily into
`/var/lib/jimm-ci/python-env/` on the first dump (expect ~2 GB and a
few minutes the first time) and reuses it on every subsequent run.

### 5. Register the GitHub App

In the GitHub UI (Settings → Developer settings → GitHub Apps → New
GitHub App):

* **Homepage URL:** anything (the repo URL is fine).
* **Webhook:** uncheck "Active". The app does not need a webhook URL or
  secret.
* **Permissions (repository):**
  * Checks: **Read & write**
  * Contents: **Read-only**
  * Metadata: **Read-only**
  * Pull requests: **Read-only**
* **Subscribe to events:** none.
* **Where can this app be installed:** Only on this account.

After creation:

1. Click **Generate a private key**; save the downloaded `.pem`.
2. Note the **App ID** from the App settings page.
3. **Install** the App on the target repository.
4. Open the App's **Install App → Configure** view; the URL contains the
   `installation_id`. (Or fetch it via the API: `GET
   /repos/{owner}/{repo}/installation`.)

### 6. Place secrets on disk

```bash
install -d -m 0750 -o root -g ci /etc/jimm-ci

# App private key
install -m 0640 -o root -g ci /path/to/jimm-ci.<id>.private-key.pem \
    /etc/jimm-ci/app.pem

# HuggingFace token used by parity tests to fetch weights
printf '%s' '<hf_xxx...>' > /etc/jimm-ci/hf-token
chmod 0640 /etc/jimm-ci/hf-token
chown root:ci /etc/jimm-ci/hf-token
```

### 7. Configure the shell environment

The `ci` user needs the GitHub App configuration in its environment
whenever it runs the CLI. The cleanest place is `/home/ci/.profile` (or
`/home/ci/.bashrc`):

```bash
sudo -u ci tee -a /home/ci/.profile <<'SH'
export JIMM_CI_APP_ID=123456
export JIMM_CI_INSTALLATION_ID=7890123
export JIMM_CI_PRIVATE_KEY_FILE=/etc/jimm-ci/app.pem
export JIMM_CI_HF_TOKEN_FILE=/etc/jimm-ci/hf-token
export JIMM_CI_REPO_OWNER=<owner>
export JIMM_CI_REPO_NAME=Luximm.jl
export JIMM_CI_JULIA=/usr/local/bin/julia
# Only needed if uv is not on the service PATH and not in a default location:
# export JIMM_CI_UV=/home/ci/.local/bin/uv
export UV_PROJECT_ENVIRONMENT=/var/lib/jimm-ci/python-env
export JULIA_NUM_THREADS=4
SH
```

## Running CI

Everything below runs as the `ci` user.

### Dry-run discovery

```bash
ssh -t ci@<vm> jimm-ci --dry-run
```

Prints the jobs that would be presented in the TUI and exits.

### Interactive run

```bash
ssh -t ci@<vm> jimm-ci
```

Launches the TUI. The `-t` is important, it allocates a PTY so
Tachikoma can render. Use the keybindings above to pick jobs, run, and
cancel.

### Run a specific commit on demand

```bash
# Re-test the current master HEAD even if it already has check runs:
ssh ci@<vm> jimm-ci --master

# Or pin to a specific commit (any length >= 7):
ssh ci@<vm> jimm-ci --sha 12fd3b8d
```

Both flags **bypass discovery and the already-tested filter** entirely
and run a full sweep on the chosen commit. Use them to retrigger a run
after fixing a bug in the runner, or to re-verify a specific master
commit. The TUI is not launched, output is streamed to the terminal
directly.

### Skip pending master commits without testing them

```bash
ssh ci@<vm> jimm-ci --skip-pending
```

Posts a `skipped` Check Run for every family on every currently-pending
**master** commit so the runner stops reconsidering them. PRs are never
auto-skipped here, decline a PR by selecting its row in the TUI and
pressing `s` instead.

## Operations

### Logs

Per-build logs are at `/var/lib/jimm-ci/logs/<sha>/<family>.log`. The
tail of each log is also embedded into the corresponding GitHub Check
Run output (capped at ~60 KB to stay under the GitHub limit), so the
GitHub UI is usually enough for spot-checking; SSH into the VM only when
you need the full log.

### Redeploy

```bash
ssh ci@<vm> 'sudo -u ci git -C /opt/jimm-ci/Luximm.jl pull --ff-only'
```

No service to restart; because `Pkg.Apps.develop` tracks the clone, the
next `jimm-ci` invocation picks up the new code automatically. If
`ci/JimmCI/Project.toml` gained a new dependency, also run:

```bash
ssh ci@<vm> 'sudo -u ci julia -e "using Pkg; Pkg.Apps.develop(path = \"/opt/jimm-ci/Luximm.jl/ci/JimmCI\")"'
```

which re-resolves and regenerates the launcher.

On the first `jimm-ci` run after upgrading to a version that includes
fork PR support, the runner adds a `refs/pull/*/head` refspec to the
mirror so fork commits are reachable by `git worktree`. The next
`git fetch --prune origin` then pulls every open PR head into
`refs/remotes/origin/pr/*`, a one-time bandwidth bump on the order of
the existing mirror; subsequent fetches are incremental.

### Forcing a re-test of an already-tested commit

The runner skips any commit whose `jimm-ci / <family>` Check Runs are
all already `completed`. To force a re-test:

1. In the GitHub UI, delete the existing Check Runs for that commit
   (Checks tab → ⋯ → **Re-run** triggers a fresh run via the API too,
   but the simplest path is just to re-create the App's check runs).
2. Or use `jimm-ci --sha <sha>` to bypass the filter for a single
   commit.

### Rotate the App private key

1. Generate a new key in the GitHub App settings; download the `.pem`.
2. Replace `/etc/jimm-ci/app.pem` (keep `0640 root:ci`).
3. Revoke the old key in the GitHub UI once a `jimm-ci --dry-run` is
   confirmed to still authenticate.

### Adding a new test family

The CI's family routing lives in `ci/JimmCI/src/PathFilter.jl`. When a
new model family is added under `src/Models/<Family>/` with a matching
`test/test_<family>.jl` and a `test/parity/dump_<family>_io.py` sidecar,
update **all** of these or the family will silently skip on CI (a
`get(..., nothing)` lookup returns early, the parity testset finds no
fixture, and the Check Run goes green without testing anything):

1. `ci/JimmCI/src/PathFilter.jl` — `_FAMILY_PREFIXES`, `_FAMILY_EXACT`,
   `ALL_FAMILIES`, `REPRESENTATIVE_VARIANT`.
2. `ci/JimmCI/src/Builder.jl` — `_FAMILY_SIDECAR` (family → dump script);
   without an entry `_ensure_fixtures!` dumps nothing.
3. `scripts/test_variant.sh` — the `case "$variant"` family→sidecar block
   (mirrors `test/_filter.jl::_jimm_variant_family`).

If the new family also builds a feature pyramid (`features_only = true`),
it needs a second fixture per variant and three more edits, or every
variant's pyramid testset skips on a missing `<variant>_featsonly_io.h5`:
`FEATURES_ONLY_VARIANTS` in `test/parity/dump_features_only_io.py`,
`_FEATSONLY_FAMILIES` in `Builder.jl`, and the `has_pyramid=1` marker in
`scripts/test_variant.sh`.
4. `test/_ci_driver.jl` — `_DRIVER_ORDER` and `_dispatch_family` (and the
   scaffold `isdefined` asserts).
5. `test/_filter.jl` — `_JIMM_DEFAULT_FAMILIES` and `_jimm_variant_family`.

The root `CLAUDE.md` repeats the source-side checklist (README table +
PathFilter); keep all of these in sync.

## Security notes

* The VM never accepts inbound connections from anyone but you over SSH.
  GitHub talks to it only through outbound HTTPS calls made by the CLI.
* Pull requests from forks are surfaced in the TUI with a `⚠` glyph
  and the head repo path (e.g. `[contrib/Luximm.jl]`). Approving one is
  a two-step action: `Enter` / `y` opens a confirmation modal naming
  the fork and head SHA, and the build only starts after a second `y`.
  Same-repo PRs and master commits skip the modal. The structural
  filter that previously dropped fork PRs is gone; the modal is the
  remaining defense-in-depth layer, so always review the diff at the
  SHA before pressing the second `y`.
* Per-PR approval (the `y` keystroke in the TUI) is the only thing
  standing between an attacker-pushed commit and code execution on the
  VM. Always review the diff at the SHA you are about to approve, not
  just "the PR."
* Secrets live under `/etc/jimm-ci/` with `0640 root:ci`. The `ci`
  account is a system user.
