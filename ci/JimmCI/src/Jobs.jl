module Jobs

using Dates
using ..ConfigMod
using ..GitHubAppMod
using ..PathFilter

export Job,
    discover_jobs,
    CHECK_NAME_PREFIX,
    MAX_OUTPUT_TEXT,
    MASTER_LOOKBACK_DAYS,
    check_name,
    has_completed_check,
    missing_families

const MAX_OUTPUT_TEXT = 60_000   # GitHub Checks API caps at 65535; leave headroom.
const MASTER_LOOKBACK_DAYS = 30
const CHECK_NAME_PREFIX = "jimm-ci / "

@enum JobKind PR_JOB MASTER_JOB

mutable struct Job
    head_sha::String
    base_sha::String
    families::Vector{String}
    full_sweep::Bool
    label::String
    kind::JobKind
    pr_number::Union{Int, Nothing}
    pr_title::Union{String, Nothing}
    head_repo::Union{String, Nothing} # `<owner>/<repo>` of the PR head; nothing for master
    is_fork::Bool                    # head_repo != base repo
    created_at::DateTime              # chronological-sort key (UTC)
    check_runs::Dict{String, Int}
end

Job(
    head_sha,
    base_sha,
    families,
    full_sweep,
    label,
    kind;
    pr_number = nothing,
    pr_title = nothing,
    head_repo = nothing,
    is_fork = false,
    created_at = now(UTC),
) = Job(
    String(head_sha),
    String(base_sha),
    collect(String, families),
    full_sweep,
    String(label),
    kind,
    pr_number,
    pr_title === nothing ? nothing : String(pr_title),
    head_repo === nothing ? nothing : String(head_repo),
    is_fork,
    created_at,
    Dict{String, Int}(),
)

check_name(family::AbstractString, variant::AbstractString) =
    variant == "" ? string(CHECK_NAME_PREFIX, family) :
    string(CHECK_NAME_PREFIX, family, " (", variant, ")")

"""True if any completed `jimm-ci / <family>…` Check Run exists on the commit."""
function has_completed_check(check_runs, family::AbstractString)
    prefix = string(CHECK_NAME_PREFIX, family)
    for run in check_runs
        name = String(get(run, "name", ""))
        if !(name == prefix || startswith(name, prefix * " "))
            continue
        end
        String(get(run, "status", "")) == "completed" && return true
    end
    return false
end

missing_families(families, check_runs) =
    [f for f in families if !has_completed_check(check_runs, f)]

# ── Discovery ─────────────────────────────────────────────────────────

function _parse_iso(s)
    s === nothing && return now(UTC)
    s = String(s)
    isempty(s) && return now(UTC)
    # GitHub timestamps look like "2026-05-21T14:23:45Z".
    endswith(s, "Z") && (s = chop(s; tail = 1))
    try
        return DateTime(s)
    catch
        return now(UTC)
    end
end

# Pure helper: classify a GitHub `pulls` element's head as same-repo or
# fork. Returns `(head_repo, is_fork)`. Extracted so the test suite can
# exercise it against synthetic dicts without going through the HTTP layer.
function _classify_pr_head(pr)
    # GitHub returns `head.repo == null` when the source fork has been
    # deleted; `get(head, "repo", Dict())` would still return `nothing`
    # in that case, so unwrap explicitly before reaching for `full_name`.
    _full_name(side) =
    let r = get(side, "repo", nothing)
        r === nothing ? nothing : get(r, "full_name", nothing)
    end
    head_repo = _full_name(get(pr, "head", Dict()))
    base_repo = _full_name(get(pr, "base", Dict()))
    is_fork = head_repo !== nothing && base_repo !== nothing && head_repo != base_repo
    return (head_repo, is_fork)
end

function _pr_jobs(cfg, gh)
    out = Job[]
    pulls = list_open_pulls(gh, repo_fullname(cfg))
    for pr in pulls
        number = get(pr, "number", nothing)
        head = get(pr, "head", Dict())
        base = get(pr, "base", Dict())
        head_sha = get(head, "sha", nothing)
        base_sha = get(base, "sha", nothing)
        if head_sha === nothing || base_sha === nothing || !(number isa Integer)
            continue
        end

        # Fork PRs are surfaced too, but flagged via `is_fork` so the TUI can
        # render them with a warning glyph and require a second confirm.
        head_repo, is_fork = _classify_pr_head(pr)

        local paths
        try
            paths = compare(gh, repo_fullname(cfg), String(base_sha), String(head_sha))
        catch e
            @warn "PR #$number compare failed" exception = e
            continue
        end
        fams = families_for_paths(paths)
        isempty(fams) && continue

        check_runs = list_check_runs(gh, repo_fullname(cfg), String(head_sha))
        todo = missing_families(fams, check_runs)
        isempty(todo) && continue

        push!(
            out,
            Job(
                String(head_sha),
                String(base_sha),
                todo,
                false,
                string("pr-", number, "@", first(String(head_sha), 8)),
                PR_JOB;
                pr_number = Int(number),
                pr_title = get(pr, "title", "") |> String,
                head_repo = head_repo,
                is_fork = is_fork,
                created_at = _parse_iso(get(pr, "updated_at", nothing)),
            ),
        )
    end
    return out
end

function _master_jobs(cfg, gh)
    out = Job[]
    since = Dates.format(
        now(UTC) - Day(MASTER_LOOKBACK_DAYS),
        dateformat"yyyy-mm-ddTHH:MM:SS\Z",
    )
    local commits
    try
        commits = list_commits(gh, repo_fullname(cfg); sha = "master", since = since)
    catch e
        @warn "listing master commits failed" exception = e
        commits = Any[]
    end

    for commit in commits
        sha = get(commit, "sha", nothing)
        sha isa AbstractString || continue
        parents = get(commit, "parents", Any[])
        parent_sha = isempty(parents) ? String(sha) : String(get(parents[1], "sha", sha))

        local check_runs
        try
            check_runs = list_check_runs(gh, repo_fullname(cfg), String(sha))
        catch e
            @warn "list_check_runs failed" sha = first(String(sha), 8) exception = e
            continue
        end
        all(has_completed_check(check_runs, f) for f in ALL_FAMILIES) && continue

        committed_at = _parse_iso(
            get(get(commit, "commit", Dict()), "committer", Dict()) |>
                d -> get(d, "date", nothing),
        )

        push!(
            out,
            Job(
                String(sha),
                String(parent_sha),
                collect(ALL_FAMILIES),
                true,
                string("master@", first(String(sha), 8)),
                MASTER_JOB;
                created_at = committed_at,
            ),
        )
    end
    return out
end

"""
    discover_jobs(cfg, gh) -> Vector{Job}

Return every untested commit eligible to run, sorted chronologically
newest-first for display in the TUI.
"""
function discover_jobs(cfg, gh)
    jobs = vcat(_pr_jobs(cfg, gh), _master_jobs(cfg, gh))
    sort!(jobs; by = j -> j.created_at, rev = true)
    return jobs
end

end # module
