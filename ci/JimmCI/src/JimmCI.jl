module JimmCI

using Dates
using Logging
using Tachikoma
@tachikoma_app

include("PathFilter.jl")
include("Config.jl")
include("GitHubApp.jl")
include("Jobs.jl")
include("Builder.jl")
include("SkipMarker.jl")
include("Tui.jl")

using .PathFilter
using .ConfigMod
using .GitHubAppMod
using .Jobs
using .BuilderMod
using .SkipMarker
using .Tui

export cli_main

# ── CLI ──────────────────────────────────────────────────────────────

const USAGE = """
jimm-ci — interactive runner for the self-hosted Luximm.jl CI

USAGE
    jimm-ci                  launch the TUI (default)
    jimm-ci --dry-run        list discovered jobs and exit
    jimm-ci --master         re-run a full sweep against the current master HEAD
    jimm-ci --sha <sha>      re-run a full sweep against a specific commit
    jimm-ci --skip-pending   mark every pending master commit `skipped`
    jimm-ci --help           show this help

Discovery, the TUI, and --skip-pending all share the same env-driven
configuration (JIMM_CI_APP_ID, JIMM_CI_INSTALLATION_ID,
JIMM_CI_PRIVATE_KEY_FILE, JIMM_CI_REPO_OWNER, JIMM_CI_REPO_NAME, …).
"""

struct CliArgs
    dry_run::Bool
    master::Bool
    sha::Union{String, Nothing}
    skip_pending::Bool
    help::Bool
end

function _parse_args(argv::AbstractVector)
    dry_run = false
    master = false
    sha::Union{String, Nothing} = nothing
    skip_pending = false
    help = false
    i = 1
    while i <= length(argv)
        a = argv[i]
        if a == "--dry-run"
            dry_run = true
        elseif a == "--master"
            master = true
        elseif a == "--skip-pending"
            skip_pending = true
        elseif a == "--help" || a == "-h"
            help = true
        elseif a == "--sha"
            i += 1
            i <= length(argv) || error("--sha requires a value")
            sha = argv[i]
        elseif startswith(a, "--sha=")
            sha = a[(length("--sha=") + 1):end]
        else
            error("unknown argument: $a")
        end
        i += 1
    end
    return CliArgs(dry_run, master, sha, skip_pending, help)
end

# ── Non-interactive paths ────────────────────────────────────────────

function _print_jobs(jobs::Vector{Job})
    if isempty(jobs)
        println("no jobs to run")
        return
    end
    for j in jobs
        scope = j.full_sweep ? "full-sweep" : "representative"
        pr_part = j.kind == Jobs.PR_JOB ? " pr=#$(j.pr_number)" : ""
        println(
            "$(j.label): sha=$(first(j.head_sha, 12)) " *
                "families=$(join(j.families, ",")) scope=$(scope)$(pr_part)",
        )
    end
    return
end

function _explicit_job(gh::GitHubApp, cfg::Config, args::CliArgs)
    if args.master
        sha = get_default_branch_head(gh, repo_fullname(cfg), "master")
        return Job(
            sha,
            sha,
            collect(ALL_FAMILIES),
            true,
            "master@$(first(sha, 8)) (manual)",
            Jobs.MASTER_JOB,
        )
    end
    if args.sha !== nothing
        s = lowercase(args.sha)
        length(s) < 7 && error("--sha $(repr(args.sha)) is too short")
        return Job(
            s,
            s,
            collect(ALL_FAMILIES),
            true,
            "$(first(s, 8)) (manual)",
            Jobs.MASTER_JOB,
        )
    end
    return nothing
end

function _run_explicit(cfg::Config, gh::GitHubApp, job::Job)
    builder = Builder(cfg, gh)
    return BuilderMod.run_job(builder, job; on_line = ln -> println(ln))
end

function _skip_pending(cfg::Config, gh::GitHubApp)
    jobs = discover_jobs(cfg, gh)
    master_jobs = [j for j in jobs if j.kind == Jobs.MASTER_JOB]
    if isempty(master_jobs)
        println("nothing to skip (no pending master commits)")
        return
    end
    for j in master_jobs
        println("skipping $(first(j.head_sha, 12)): $(join(j.families, ","))")
        mark_skipped(gh, repo_fullname(cfg), j.head_sha, j.families; source = :skip)
    end
    return
end

# ── Entry point ──────────────────────────────────────────────────────

function cli_main(argv::AbstractVector = ARGS)
    args = try
        _parse_args(argv)
    catch e
        println(stderr, "jimm-ci: ", sprint(showerror, e))
        println(stderr, USAGE)
        exit(2)
    end
    if args.help
        print(USAGE)
        return
    end

    _logfile = open("jimm-ci.log", "a")
    min_level =
        get(ENV, "JIMM_CI_LOG_LEVEL", "INFO") == "DEBUG" ? Logging.Debug : Logging.Info
    Logging.global_logger(ConsoleLogger(_logfile, min_level))
    @info "jimm-ci starting" time = now(UTC)

    cfg = try
        ConfigMod.from_env()
    catch e
        println(stderr, "jimm-ci: ", sprint(showerror, e))
        exit(2)
    end
    gh = GitHubApp(cfg.app_id, cfg.installation_id, cfg.private_key)

    if args.skip_pending
        _skip_pending(cfg, gh)
        return
    end

    explicit = _explicit_job(gh, cfg, args)
    if explicit !== nothing
        if args.dry_run
            _print_jobs([explicit])
            return
        end
        _run_explicit(cfg, gh, explicit)
        return
    end

    if args.dry_run
        jobs = discover_jobs(cfg, gh)
        _print_jobs(jobs)
        return
    end

    # Default: launch the TUI.
    Tui.run_tui(cfg, gh)
    return
end

# `@main` registers this as the package entry point for `julia -m JimmCI`,
# which is the launcher shim that `Pkg.Apps.add` installs to `~/.julia/bin/`.
function (@main)(args::Vector{String})
    try
        cli_main(args)
    catch e
        if e isa InterruptException
            return 130
        end
        bt = catch_backtrace()
        @error "fatal exception" exception = (e, bt)
        showerror(stderr, e, bt)
        println(stderr)
        return 1
    end
    return 0
end

end # module
