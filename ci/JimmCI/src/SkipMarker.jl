module SkipMarker

using Dates
using ..GitHubAppMod
using ..Jobs: check_name

export mark_skipped

"""
    mark_skipped(gh, repo, sha, families; source=:run)

Post a completed `skipped` Check Run for each `jimm-ci / <family>` on the
given commit so the runner stops considering it. `source` controls the
summary wording (`:run` for the in-TUI decline path, `:skip` for the
explicit skip CLI).
"""
function mark_skipped(
        gh::GitHubApp,
        repo::AbstractString,
        sha::AbstractString,
        families;
        source::Symbol = :run,
    )
    today_str = Dates.format(Dates.today(), dateformat"yyyy-mm-dd")
    summary =
        source === :skip ?
        "Marked skipped by `jimm-ci --skip-pending` on $today_str. " *
        "Re-running CI on this commit requires deleting this check or " *
        "pushing a new commit on top." :
        "Cancelled by `jimm-ci` on $today_str. " *
        "Push a new commit on the PR to re-prompt."
    output = Dict{String, Any}("title" => "Skipped", "summary" => summary)
    for family in families
        create_check_run(
            gh,
            repo,
            sha,
            check_name(family, "");
            status = "completed",
            conclusion = "skipped",
            output = output,
        )
    end
    return nothing
end

end # module
