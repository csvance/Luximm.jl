module BuilderMod

using Logging

using ..ConfigMod
using ..GitHubAppMod
using ..PathFilter
using ..Jobs

export Builder, run_job

const LOG = Logging.global_logger()

# Maps test family → Python sidecar that dumps its parity fixtures.
# Mirrors the family resolution in scripts/test_variant.sh.
const _FAMILY_SIDECAR = Dict{String,String}(
    "bit" => "test/parity/dump_resnetv2_bit_io.py",
    "resnet" => "test/parity/dump_resnet_io.py",
    "convnext" => "test/parity/dump_convnext_io.py",
    "convnextv2" => "test/parity/dump_convnextv2_io.py",
    "vgg" => "test/parity/dump_vgg_io.py",
    "seresnet" => "test/parity/dump_seresnet_io.py",
    "vit" => "test/parity/dump_vit_io.py",
    "coatnet" => "test/parity/dump_coatnet_io.py",
)

# Sidecar for the `features_only` (feature-pyramid) fixtures, which are a
# second fixture per variant named `<variant>_featsonly_io.h5` rather than a
# per-family script. Only the five families with a pyramid have them; the
# sidecar itself no-ops on a variant from any other family.
const _FEATSONLY_SIDECAR = "test/parity/dump_features_only_io.py"
const _FEATSONLY_FAMILIES = Set(["resnet", "seresnet", "bit", "convnext", "convnextv2"])

struct Builder
    cfg::Config
    gh::GitHubApp
end

# ── Build-cancel token shared with the TUI ────────────────────────────
# Named `BuildCancel` (not `CancelToken`) to avoid colliding with
# Tachikoma's own `CancelToken`, which cancels task-queue tasks. Ours
# signals the running subprocess to exit; the task itself unwinds
# normally so check_runs are still posted as `cancelled`.

mutable struct BuildCancel
    cancelled::Bool
    proc::Union{Nothing,Base.Process}
    lock::ReentrantLock
end
BuildCancel() = BuildCancel(false, nothing, ReentrantLock())

function request_cancel!(t::BuildCancel)
    @lock t.lock begin
        t.cancelled = true
        p = t.proc
        if p !== nothing && process_running(p)
            try
                # Child is spawned with `detach = true`, so libuv puts it in
                # a new session (PID == PGID). Signal the whole group so the
                # inner `Pkg.test()` Julia and any torch workers go down too —
                # plain `kill(p, …)` only hits the direct child and leaves
                # grandchildren writing to the pipe forever.
                ccall(:kill, Cint, (Cint, Cint), -getpid(p), Base.SIGTERM)
            catch
            end
        end
    end
end

is_cancelled(t::BuildCancel) = @lock t.lock t.cancelled
_attach!(t::BuildCancel, p::Base.Process) = @lock t.lock (t.proc = p)
_detach!(t::BuildCancel) = @lock t.lock (t.proc = nothing)

export BuildCancel, request_cancel!, is_cancelled

# ── Subprocess runner streaming stdout+stderr to file + callback ──────

function _read_tail(path::AbstractString; limit::Int = MAX_OUTPUT_TEXT)
    isfile(path) || return ""
    data = read(path)
    if length(data) <= limit
        return String(data)
    end
    head = b"[... log truncated ...]\n"
    tail = data[(end-(limit-length(head))+1):end]
    return String(vcat(head, tail))
end

"""
    _stream_subprocess(cmd, env, log_path, on_line, token)

Run `cmd` with merged stdout/stderr, append each line to `log_path` *and*
invoke `on_line(line)`. Honors a `BuildCancel`: on cancel, sends SIGTERM,
waits up to 10 s, then SIGKILL. Returns the child's exit code (or a
nonzero sentinel on cancellation).
"""
function _stream_subprocess(
    cmd::Cmd,
    env::Dict{String,String},
    log_path::AbstractString,
    on_line::Function,
    token::BuildCancel;
    cwd::Union{Nothing,AbstractString} = nothing,
)
    mkpath(dirname(log_path))
    full_cmd = addenv(cmd, env)
    cwd === nothing || (full_cmd = setenv(full_cmd; dir = cwd))
    full_cmd = Cmd(full_cmd; detach = true)

    pipe = Pipe()
    open(log_path, "w") do logio
        write(logio, "\$ ")
        write(logio, string(cmd))
        write(logio, "\n")
        flush(logio)

        proc = run(pipeline(full_cmd; stdout = pipe, stderr = pipe); wait = false)
        _attach!(token, proc)
        close(pipe.in)   # close parent's copy of the write end

        # Watchdog: if cancellation arrives while we're blocked in readline
        # or waiting on the process, escalate SIGTERM to SIGKILL after 10s.
        watchdog = Threads.@spawn begin
            while process_running(proc)
                if is_cancelled(token)
                    sleep(10)
                    if process_running(proc)
                        try
                            ccall(:kill, Cint, (Cint, Cint), -getpid(proc), Base.SIGKILL)
                        catch
                        end
                    end
                    return
                end
                sleep(0.5)
            end
        end

        try
            while !eof(pipe)
                line = readline(pipe; keep = false)
                println(logio, line)
                flush(logio)
                try
                    on_line(line)
                catch e
                    @warn "on_line callback raised" exception=e
                end
            end
        finally
            wait(proc)
            _detach!(token)
            try
                ; fetch(watchdog);
            catch
                ;
            end
        end

        return proc.exitcode
    end
end

# ── Git plumbing ──────────────────────────────────────────────────────

function _git(
    cfg::Config,
    args::Vector{String};
    cwd::AbstractString,
    log_path::AbstractString,
)
    env = copy(ENV)
    env["GIT_TERMINAL_PROMPT"] = "0"
    cmd = Cmd(["git"; args])
    mkpath(dirname(log_path))
    open(log_path, "w") do io
        write(io, "\$ ", string(cmd), "\n")
    end
    try
        run(
            pipeline(
                setenv(cmd, env; dir = cwd);
                stdout = log_path,
                stderr = log_path,
                append = true,
            ),
        )
    catch e
        output = _read_tail(log_path)
        error("git $(join(args, " ")) failed (cwd=$cwd):\n$output")
    end
    return nothing
end

function _is_bare_repo(path::AbstractString)
    isdir(path) || return false
    env = copy(ENV);
    env["GIT_TERMINAL_PROMPT"] = "0"
    try
        out = read(setenv(`git -C $path rev-parse --is-bare-repository`, env), String)
        return strip(out) == "true"
    catch
        return false
    end
end

# GitHub publishes every open PR's head at refs/pull/<N>/head on the base
# repo, including PRs whose head branch lives in a fork. Mapping those into
# `refs/remotes/origin/pr/*` makes the fork SHA reachable by `git worktree
# add` without needing per-fork remotes.
const _PR_REFSPEC = "+refs/pull/*/head:refs/remotes/origin/pr/*"

function _ensure_pr_refspec!(cfg::Config)
    env = copy(ENV);
    env["GIT_TERMINAL_PROMPT"] = "0"
    existing = try
        read(
            setenv(`git -C $(cfg.mirror_dir) config --get-all remote.origin.fetch`, env),
            String,
        )
    catch
        ""
    end
    occursin(_PR_REFSPEC, existing) && return
    _git(
        cfg,
        ["-C", cfg.mirror_dir, "config", "--add", "remote.origin.fetch", _PR_REFSPEC];
        cwd = cfg.mirror_dir,
        log_path = joinpath(cfg.log_dir, "mirror-refspec.log"),
    )
end

function _ensure_mirror!(b::Builder)
    cfg = b.cfg
    if isdir(cfg.mirror_dir) && !_is_bare_repo(cfg.mirror_dir)
        @warn "mirror dir is not a usable bare repo; recloning" path=cfg.mirror_dir
        rm(cfg.mirror_dir; recursive = true, force = true)
    end
    if !isdir(cfg.mirror_dir)
        mkpath(dirname(cfg.mirror_dir))
        _git(
            cfg,
            [
                "clone",
                "--mirror",
                "https://github.com/$(repo_fullname(cfg)).git",
                cfg.mirror_dir,
            ],
            cwd = dirname(cfg.mirror_dir),
            log_path = joinpath(cfg.log_dir, "mirror-init.log"),
        )
        _ensure_pr_refspec!(cfg)
        _git(
            cfg,
            ["fetch", "--prune", "origin"];
            cwd = cfg.mirror_dir,
            log_path = joinpath(cfg.log_dir, "mirror-fetch.log"),
        )
    else
        _ensure_pr_refspec!(cfg)
        _git(
            cfg,
            ["fetch", "--prune", "origin"];
            cwd = cfg.mirror_dir,
            log_path = joinpath(cfg.log_dir, "mirror-fetch.log"),
        )
    end
end

function _make_worktree!(b::Builder, sha::AbstractString)
    cfg = b.cfg
    wt = joinpath(cfg.workspace_dir, sha)
    isdir(wt) && rm(wt; recursive = true, force = true)
    mkpath(dirname(wt))
    mkpath(cfg.parity_dir)
    _git(
        cfg,
        ["-C", cfg.mirror_dir, "worktree", "prune"];
        cwd = cfg.mirror_dir,
        log_path = joinpath(cfg.log_dir, sha, "worktree-prune.log"),
    )
    _git(
        cfg,
        ["-C", cfg.mirror_dir, "worktree", "add", "--detach", wt, sha];
        cwd = cfg.mirror_dir,
        log_path = joinpath(cfg.log_dir, sha, "worktree.log"),
    )
    return wt
end

function _drop_worktree!(b::Builder, wt::AbstractString)
    cfg = b.cfg
    try
        _git(
            cfg,
            ["-C", cfg.mirror_dir, "worktree", "remove", "--force", wt];
            cwd = cfg.mirror_dir,
            log_path = joinpath(cfg.log_dir, basename(wt), "worktree-remove.log"),
        )
    catch e
        @warn "worktree remove failed" wt exception=e
        isdir(wt) && rm(wt; recursive = true, force = true)
    end
end

# ── Env construction ─────────────────────────────────────────────────

function _env_for_run(cfg::Config, families::Vector{String}, variants::Dict{String,String})
    env = copy(ENV)
    env["JULIA_NUM_THREADS"] = get(ENV, "JULIA_NUM_THREADS", "4")
    env["HF_HUB_CACHE"] = cfg.hf_cache
    env["JULIA_DEPOT_PATH"] = cfg.julia_depot
    env["JULIA_LOAD_PATH"] = "@:@v#.#:@stdlib"
    env["JIMM_TEST_FAMILIES"] = join(families, ",")
    var_list = String[]
    for f in families
        v = get(variants, f, "")
        isempty(v) || push!(var_list, v)
    end
    if isempty(var_list)
        delete!(env, "JIMM_TEST_VARIANTS")
    else
        env["JIMM_TEST_VARIANTS"] = join(var_list, ",")
    end
    env["JIMM_PARITY_DIR"] = cfg.parity_dir
    # CoAtNet *_in12k repos use HF Xet storage, whose chunked transport can
    # stall on some networks; force the classic blob download path.
    env["HF_HUB_DISABLE_XET"] = "1"
    if cfg.hf_token !== nothing
        env["HF_TOKEN"] = cfg.hf_token
        env["HUGGING_FACE_HUB_TOKEN"] = cfg.hf_token
    end
    return env
end

function _env_for_sidecar(cfg::Config)
    env = copy(ENV)
    env["UV_PROJECT_ENVIRONMENT"] = cfg.python_env
    env["HF_HUB_CACHE"] = cfg.hf_cache
    env["JIMM_PARITY_DIR"] = cfg.parity_dir
    # See _env_for_run: avoid the HF Xet stall on in12k checkpoint downloads.
    env["HF_HUB_DISABLE_XET"] = "1"
    if cfg.hf_token !== nothing
        env["HF_TOKEN"] = cfg.hf_token
        env["HUGGING_FACE_HUB_TOKEN"] = cfg.hf_token
    end
    return env
end

# ── Parity fixture dump ──────────────────────────────────────────────

function _ensure_fixtures!(
    b::Builder,
    job::Job,
    wt::AbstractString,
    family::AbstractString,
    variant::AbstractString,
    on_line::Function,
    token::BuildCancel,
)
    sidecar = get(_FAMILY_SIDECAR, family, nothing)
    sidecar === nothing && return

    if !isempty(variant)
        # Every family's test file has both a 3-channel parity testset and
        # an `in_chans=1` testset (the latter exercises timm's
        # `adapt_input_conv` stem path). Dump both fixtures or the in1c
        # testset silently skips with "fixture missing".
        for ic in (3, 1)
            _dump_variant_fixture!(b, job, wt, family, sidecar, variant, ic, on_line, token)
        end
        _dump_featsonly_fixture!(b, job, wt, family, variant, on_line, token)
        return
    end

    # ── Full sweep ──
    log_path = joinpath(b.cfg.log_dir, job.head_sha, "$family-dump.log")
    for ic in (3, 1)
        ic_args = ic == 3 ? String[] : ["--in-chans", "1"]
        cmd = Cmd(
            String["uv", "run", "--project", wt, "python", sidecar, "--all", ic_args...],
        )
        env = _env_for_sidecar(b.cfg)
        rc = _stream_subprocess(cmd, env, log_path, on_line, token; cwd = wt)
        rc == 0 || error(
            "parity dump for $family/all in_chans=$ic " * "failed (rc=$rc); see $log_path",
        )
    end

    # The `features_only` fixtures are a separate sidecar and a separate file
    # per variant; without this the pyramid testsets silently skip.
    if family in _FEATSONLY_FAMILIES
        fo_log = joinpath(b.cfg.log_dir, job.head_sha, "$family-featsonly-dump.log")
        cmd = Cmd(
            String[
                "uv",
                "run",
                "--project",
                wt,
                "python",
                _FEATSONLY_SIDECAR,
                "--all",
                "--family",
                family,
            ],
        )
        rc = _stream_subprocess(
            cmd,
            _env_for_sidecar(b.cfg),
            fo_log,
            on_line,
            token;
            cwd = wt,
        )
        rc == 0 || error("features_only dump for $family/all failed (rc=$rc); see $fo_log")
    end

    # The worktree scripts may not know about JIMM_PARITY_DIR, so they
    # write to <wt>/data/parity/. Promote any new fixtures into parity_dir,
    # then mirror everything cached there back into the worktree so test
    # files that hardcode `data/parity/` find them too.
    wt_parity = joinpath(wt, "data", "parity")
    if isdir(wt_parity)
        for f in readdir(wt_parity; join = true)
            endswith(f, ".h5") || continue
            dest = joinpath(b.cfg.parity_dir, basename(f))
            isfile(dest) || cp(f, dest)
        end
    end
    for f in readdir(b.cfg.parity_dir; join = true)
        endswith(f, ".h5") || continue
        _link_fixture_into_worktree(f, wt)
    end
    return
end

# Dump a single (variant, in_chans) parity fixture into `cfg.parity_dir`
# if it isn't already cached, then symlink it into the worktree. The 3-ch
# fixture is named `<variant>_io.h5`; non-default channel counts get the
# `_in<N>c` suffix that the test files expect.
function _dump_variant_fixture!(
    b::Builder,
    job::Job,
    wt::AbstractString,
    family::AbstractString,
    sidecar::AbstractString,
    variant::AbstractString,
    in_chans::Int,
    on_line::Function,
    token::BuildCancel,
)
    suffix = in_chans == 3 ? "" : "_in$(in_chans)c"
    fixture = joinpath(b.cfg.parity_dir, "$(variant)$(suffix)_io.h5")
    if !isfile(fixture)
        args = ["--variant", variant, "--in-chans", string(in_chans), "--out", fixture]
        log_path = joinpath(b.cfg.log_dir, job.head_sha, "$(family)$(suffix)-dump.log")
        cmd = Cmd(String["uv", "run", "--project", wt, "python", sidecar, args...])
        env = _env_for_sidecar(b.cfg)
        rc = _stream_subprocess(cmd, env, log_path, on_line, token; cwd = wt)
        rc == 0 || error(
            "parity dump for $family/$variant in_chans=$in_chans " *
            "failed (rc=$rc); see $log_path",
        )
    end
    _link_fixture_into_worktree(fixture, wt)
end

# Dump the `features_only` (feature-pyramid) fixture for one variant. Named
# `<variant>_featsonly_io.h5` and consumed by
# `run_variant_feature_pyramid_parity`. Families without a pyramid are
# skipped here rather than in the sidecar, so no subprocess is spawned.
function _dump_featsonly_fixture!(
    b::Builder,
    job::Job,
    wt::AbstractString,
    family::AbstractString,
    variant::AbstractString,
    on_line::Function,
    token::BuildCancel,
)
    family in _FEATSONLY_FAMILIES || return
    fixture = joinpath(b.cfg.parity_dir, "$(variant)_featsonly_io.h5")
    if !isfile(fixture)
        args = ["--variant", variant, "--out", fixture]
        log_path = joinpath(b.cfg.log_dir, job.head_sha, "$(family)-featsonly-dump.log")
        cmd =
            Cmd(String["uv", "run", "--project", wt, "python", _FEATSONLY_SIDECAR, args...])
        env = _env_for_sidecar(b.cfg)
        rc = _stream_subprocess(cmd, env, log_path, on_line, token; cwd = wt)
        rc == 0 ||
            error("features_only dump for $family/$variant failed (rc=$rc); see $log_path")
    end
    isfile(fixture) && _link_fixture_into_worktree(fixture, wt)
    return
end

# Symlink a cached fixture from `cfg.parity_dir` into the worktree's
# `data/parity/` so test files baked into the worktree's SHA find it
# regardless of whether they read `JIMM_PARITY_DIR`. The new test files
# (post-`f5ebc4e`) ignore the symlink and resolve via env; the old test
# files traverse it. The symlink dies with the worktree.
function _link_fixture_into_worktree(fixture::AbstractString, wt::AbstractString)
    wt_dir = joinpath(wt, "data", "parity")
    mkpath(wt_dir)
    dest = joinpath(wt_dir, basename(fixture))
    (isfile(dest) || islink(dest)) && return
    try
        symlink(fixture, dest)
    catch
        cp(fixture, dest; force = true)
    end
end

# ── Job execution ────────────────────────────────────────────────────
#
# The driver (`test/_ci_driver.jl`) runs every requested family inside one
# Julia process and emits two structured stdout markers per family:
#
#     ==> JIMM_FAMILY_BEGIN: family=<name>
#     ==> JIMM_FAMILY_END:   family=<name> rc=<0|1>
#
# `_run_driver!` watches those markers to drive the per-family check_run
# lifecycle while buffering each family's intermediate lines for the
# check_run's `output.text` field.

mutable struct _FamilyState
    name::String
    variant::String
    check_id::Union{Int,Nothing}
    log_path::String
    log_io::Union{IOStream,Nothing}
    completed::Bool
end

_FamilyState(name, variant, log_dir) = _FamilyState(
    String(name),
    String(variant),
    nothing,
    joinpath(log_dir, "$(name).log"),
    nothing,
    false,
)

const _MARKER_BEGIN_RE = r"^==> JIMM_FAMILY_BEGIN: family=(\S+)"
const _MARKER_END_RE = r"^==> JIMM_FAMILY_END: family=(\S+) rc=(\d+)"

function _start_family_check!(b::Builder, job::Job, st::_FamilyState, on_line::Function)
    st.check_id === nothing || return
    name = check_name(st.name, st.variant)
    try
        check = create_check_run(
            b.gh,
            repo_fullname(b.cfg),
            job.head_sha,
            name;
            status = "in_progress",
        )
        st.check_id = check.id
        job.check_runs[st.name] = check.id
        on_line("==> [check-run] $(name) → in_progress (id=$(check.id))")
    catch e
        @warn "create check_run failed" family=st.name exception=e
    end
end

function _finish_family_check!(
    b::Builder,
    job::Job,
    st::_FamilyState,
    conclusion::AbstractString;
    rc::Int = -1,
    extra_text::AbstractString = "",
    on_line::Function = identity,
)
    st.completed = true
    _close_family_log!(st)
    _start_family_check!(b, job, st, on_line)
    st.check_id === nothing && return

    name = check_name(st.name, st.variant)
    title_suffix =
        conclusion == "success" ? "passed" :
        conclusion == "failure" ? "failed" :
        conclusion == "cancelled" ? "cancelled" : conclusion
    summary = if rc >= 0
        "Exit code $(rc). Variant: `$(isempty(st.variant) ? "all" : st.variant)`. " *
        "Sweep: `$(job.full_sweep)`."
    else
        "Variant: `$(isempty(st.variant) ? "all" : st.variant)`. Sweep: `$(job.full_sweep)`."
    end

    text = _read_tail(st.log_path)
    if !isempty(extra_text)
        text = isempty(text) ? extra_text : string(extra_text, "\n\n", text)
        if length(text) > MAX_OUTPUT_TEXT
            head = "[... log truncated ...]\n"
            tail = text[(end-(MAX_OUTPUT_TEXT-length(head))+1):end]
            text = string(head, tail)
        end
    end

    suffix = rc >= 0 ? " (rc=$(rc))" : ""
    on_line("==> [check-run] $(name) → $(conclusion)$(suffix)")
    try
        complete_check_run(
            b.gh,
            repo_fullname(b.cfg),
            st.check_id;
            conclusion = conclusion,
            output = Dict(
                "title" => "$(st.name) $(title_suffix)",
                "summary" => summary,
                "text" => text,
            ),
        )
    catch e
        @warn "complete check_run failed" family=st.name conclusion exception=e
    end
end

function _open_family_log!(st::_FamilyState)
    mkpath(dirname(st.log_path))
    st.log_io = open(st.log_path, "w")
end

function _close_family_log!(st::_FamilyState)
    io = st.log_io
    io === nothing && return
    try
        flush(io)
        close(io)
    catch
    end
    st.log_io = nothing
end

function _append_family_log!(st::_FamilyState, line::AbstractString)
    io = st.log_io
    io === nothing && return
    try
        println(io, line)
    catch
    end
end

function _preflight_fixtures!(
    b::Builder,
    job::Job,
    wt::AbstractString,
    variants::Dict{String,String},
    on_line::Function,
    token::BuildCancel,
)
    ready = String[]
    failed = Dict{String,String}()
    for family in job.families
        is_cancelled(token) && throw(InterruptException())
        variant = get(variants, family, "")
        try
            _ensure_fixtures!(b, job, wt, family, variant, on_line, token)
            push!(ready, family)
        catch e
            if is_cancelled(token) || e isa InterruptException
                rethrow()
            end
            msg = sprint(showerror, e)
            on_line("==> fixture dump for $(family) failed: $(msg)")
            failed[family] = msg
        end
    end
    return ready, failed
end

function _run_driver!(
    b::Builder,
    job::Job,
    wt::AbstractString,
    variants::Dict{String,String},
    on_line::Function,
    token::BuildCancel,
)
    log_dir = joinpath(b.cfg.log_dir, job.head_sha)
    mkpath(log_dir)

    ready, preflight_failed = _preflight_fixtures!(b, job, wt, variants, on_line, token)

    state = Dict{String,_FamilyState}()
    for family in job.families
        state[family] = _FamilyState(family, get(variants, family, ""), log_dir)
    end

    for (family, msg) in preflight_failed
        _finish_family_check!(
            b,
            job,
            state[family],
            "failure";
            extra_text = "Parity-fixture dump failed:\n$(msg)",
            on_line = on_line,
        )
    end

    if isempty(ready)
        on_line("==> no families ready to run (all preflight failed)")
        return
    end

    current = Ref{Union{Nothing,String}}(nothing)

    function on_driver_line(line::AbstractString)
        on_line(line)

        m = match(_MARKER_BEGIN_RE, line)
        if m !== nothing
            fam = String(m.captures[1])
            cur = current[]
            if cur !== nothing && haskey(state, cur) && !state[cur].completed
                _finish_family_check!(
                    b,
                    job,
                    state[cur],
                    "failure";
                    extra_text = "Driver started a new family without emitting JIMM_FAMILY_END for this one.",
                    on_line = on_line,
                )
            end
            current[] = fam
            haskey(state, fam) || return
            _open_family_log!(state[fam])
            _start_family_check!(b, job, state[fam], on_line)
            return
        end

        m = match(_MARKER_END_RE, line)
        if m !== nothing
            fam = String(m.captures[1])
            rc = parse(Int, m.captures[2])
            haskey(state, fam) && _finish_family_check!(
                b,
                job,
                state[fam],
                rc == 0 ? "success" : "failure";
                rc = rc,
                on_line = on_line,
            )
            current[] = nothing
            return
        end

        cur = current[]
        cur !== nothing && haskey(state, cur) && _append_family_log!(state[cur], line)
    end

    driver_log = joinpath(log_dir, "driver.log")
    env = _env_for_run(b.cfg, ready, variants)
    cmd = Cmd([
        b.cfg.julia_binary,
        "--project=.",
        "-e",
        "using Pkg; Pkg.instantiate(); include(\"test/_ci_driver.jl\")",
    ])

    rc_overall = 1
    crashed_err::Union{Nothing,String} = nothing
    cancelled = false
    try
        rc_overall =
            _stream_subprocess(cmd, env, driver_log, on_driver_line, token; cwd = wt)
    catch e
        if is_cancelled(token) || e isa InterruptException
            cancelled = true
        else
            crashed_err = sprint(showerror, e)
            @warn "driver subprocess raised" exception=e
        end
    end

    cur = current[]
    cur === nothing || _close_family_log!(state[cur])

    for family in ready
        st = state[family]
        st.completed && continue
        if cancelled || is_cancelled(token)
            _finish_family_check!(
                b,
                job,
                st,
                "cancelled";
                extra_text = "Build was cancelled before this family finished.",
                on_line = on_line,
            )
        else
            note =
                crashed_err === nothing ?
                "Driver exited (rc=$(rc_overall)) before this family emitted JIMM_FAMILY_END." :
                "Driver subprocess raised: $(crashed_err)"
            _finish_family_check!(
                b,
                job,
                st,
                "failure";
                rc = rc_overall,
                extra_text = note,
                on_line = on_line,
            )
        end
    end

    cancelled && throw(InterruptException())
    return
end

"""
    run_job(builder, job; on_line, token)

Run every family in `job` inside a single Julia subprocess driven by
`test/_ci_driver.jl`, streaming each line of the build's combined
stdout/stderr through `on_line(line)`. Per-family GitHub check_runs are
driven by the `JIMM_FAMILY_BEGIN` / `JIMM_FAMILY_END` markers the driver
emits. Cancellation is honored via `token` (see `BuildCancel`).
"""
function run_job(
    builder::Builder,
    job::Job;
    on_line::Function = identity,
    token::BuildCancel = BuildCancel(),
)
    on_line(
        "==> starting $(job.label) sha=$(job.head_sha) " *
        "families=$(join(job.families, ",")) sweep=$(job.full_sweep)",
    )
    _ensure_mirror!(builder)
    wt = _make_worktree!(builder, job.head_sha)
    try
        variants = Dict{String,String}()
        for family in job.families
            variants[family] = job.full_sweep ? "" : get(REPRESENTATIVE_VARIANT, family, "")
        end
        _run_driver!(builder, job, wt, variants, on_line, token)
    finally
        _drop_worktree!(builder, wt)
    end
    return nothing
end

end # module
