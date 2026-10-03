module GitHubAppMod

using Dates
using HTTP
using JSON3
using JSONWebTokens

export GitHubApp,
    create_check_run,
    complete_check_run,
    compare,
    get_default_branch_head,
    list_open_pulls,
    list_commits,
    list_check_runs,
    CheckRun

const API = "https://api.github.com"

struct CheckRun
    id::Int
    name::String
    head_sha::String
    html_url::String
end

mutable struct GitHubApp
    app_id::Int
    installation_id::Int
    private_key::String
    token::Union{String, Nothing}
    token_exp::Float64
    lock::ReentrantLock
end

GitHubApp(app_id::Integer, installation_id::Integer, private_key::AbstractString) =
    GitHubApp(
    Int(app_id),
    Int(installation_id),
    String(private_key),
    nothing,
    0.0,
    ReentrantLock(),
)

# ── Auth ──────────────────────────────────────────────────────────────

function _mint_jwt(gh::GitHubApp)
    now_s = floor(Int, time())
    claims = Dict("iat" => now_s - 60, "exp" => now_s + 540, "iss" => string(gh.app_id))
    enc = JSONWebTokens.RS256(gh.private_key)
    return JSONWebTokens.encode(enc, claims)
end

function _refresh_token!(gh::GitHubApp)
    app_jwt = _mint_jwt(gh)
    r = HTTP.request(
        "POST",
        string(API, "/app/installations/", gh.installation_id, "/access_tokens");
        headers = [
            "Authorization" => "Bearer $app_jwt",
            "Accept" => "application/vnd.github+json",
            "X-GitHub-Api-Version" => "2022-11-28",
        ],
        status_exception = true,
    )
    data = JSON3.read(String(r.body))
    gh.token = String(data["token"])
    # GitHub-issued installation tokens are valid for ~1h; keep a 5m safety margin.
    gh.token_exp = time() + 3300
    return gh.token
end

function installation_token(gh::GitHubApp)
    @lock gh.lock begin
        if gh.token !== nothing && time() < gh.token_exp - 300
            return gh.token
        end
        return _refresh_token!(gh)
    end
end

# ── Core request helper ───────────────────────────────────────────────

function _auth_headers(gh::GitHubApp; extra = ())
    token = installation_token(gh)
    base = [
        "Authorization" => "Bearer $token",
        "Accept" => "application/vnd.github+json",
        "X-GitHub-Api-Version" => "2022-11-28",
    ]
    return vcat(base, collect(extra))
end

function _request(
        gh::GitHubApp,
        method::AbstractString,
        path::AbstractString;
        body = nothing,
        query = nothing,
    )
    url = startswith(path, "http") ? path : string(API, path)
    headers = _auth_headers(gh)
    opts = Dict{Symbol, Any}(:headers => headers, :status_exception => false)
    if body !== nothing
        opts[:body] = JSON3.write(body)
        push!(headers, "Content-Type" => "application/json")
    end
    if query !== nothing
        opts[:query] = query
    end
    r = HTTP.request(method, url; opts...)
    if r.status == 401
        # Drop cached token and retry once.
        @lock gh.lock begin
            gh.token = nothing
            gh.token_exp = 0.0
        end
        opts[:headers] = _auth_headers(gh)
        r = HTTP.request(method, url; opts...)
    end
    if r.status >= 400
        msg = String(r.body)
        if length(msg) > 500
            msg = msg[1:500] * "…"
        end
        error("GitHub $method $path failed: HTTP $(r.status) $msg")
    end
    return r
end

# ── Pagination ────────────────────────────────────────────────────────

# Extract the `next` link from a GitHub Link header. The header looks
# like:  <https://api.github.com/foo?page=2>; rel="next", <…>; rel="last"
function _next_link(r::HTTP.Response)
    link = HTTP.header(r, "Link")
    isempty(link) && return nothing
    for part in split(link, ',')
        m = match(r"^\s*<([^>]+)>\s*;\s*rel=\"([^\"]+)\"\s*$", part)
        m === nothing && continue
        if m.captures[2] == "next"
            return String(m.captures[1])
        end
    end
    return nothing
end

function _paginated(
        gh::GitHubApp,
        path::AbstractString;
        per_page::Int = 100,
        max_pages::Int = 20,
    )
    sep = occursin('?', path) ? '&' : '?'
    url = string(path, sep, "per_page=", per_page)
    out = Any[]
    pages = 0
    while url !== nothing && pages < max_pages
        r = _request(gh, "GET", url)
        page = JSON3.read(String(r.body))
        if !(page isa AbstractVector)
            break
        end
        append!(out, page)
        url = _next_link(r)
        pages += 1
    end
    return out
end

# ── Checks API ────────────────────────────────────────────────────────

function create_check_run(
        gh::GitHubApp,
        repo::AbstractString,
        head_sha::AbstractString,
        name::AbstractString;
        status::AbstractString = "in_progress",
        conclusion::Union{Nothing, AbstractString} = nothing,
        details_url::Union{Nothing, AbstractString} = nothing,
        output = nothing,
    )
    body = Dict{String, Any}("name" => name, "head_sha" => head_sha, "status" => status)
    conclusion === nothing || (body["conclusion"] = conclusion)
    details_url === nothing || (body["details_url"] = details_url)
    output === nothing || (body["output"] = output)
    r = _request(gh, "POST", "/repos/$repo/check-runs"; body = body)
    data = JSON3.read(String(r.body))
    return CheckRun(
        Int(data["id"]),
        String(data["name"]),
        String(data["head_sha"]),
        String(data["html_url"]),
    )
end

function complete_check_run(
        gh::GitHubApp,
        repo::AbstractString,
        check_run_id::Integer;
        conclusion::AbstractString,
        output = nothing,
    )
    body = Dict{String, Any}("status" => "completed", "conclusion" => conclusion)
    output === nothing || (body["output"] = output)
    _request(gh, "PATCH", "/repos/$repo/check-runs/$check_run_id"; body = body)
    return nothing
end

# ── Compare / Pulls / Commits / Check Runs listing ────────────────────

"""Return the list of changed file paths between `base` and `head`."""
function compare(
        gh::GitHubApp,
        repo::AbstractString,
        base::AbstractString,
        head::AbstractString,
    )
    r = _request(gh, "GET", "/repos/$repo/compare/$base...$head")
    data = JSON3.read(String(r.body))
    files = get(data, "files", nothing)
    files === nothing && return String[]
    return [String(f["filename"]) for f in files]
end

function get_default_branch_head(
        gh::GitHubApp,
        repo::AbstractString,
        branch::AbstractString = "master",
    )
    r = _request(gh, "GET", "/repos/$repo/branches/$branch")
    data = JSON3.read(String(r.body))
    return String(data["commit"]["sha"])
end

function list_open_pulls(gh::GitHubApp, repo::AbstractString)
    return _paginated(gh, "/repos/$repo/pulls?state=open")
end

"""
List commits on `sha` (branch or commit) committed at or after `since`,
where `since` is an ISO-8601 timestamp (e.g. `2026-04-20T00:00:00Z`).
"""
function list_commits(
        gh::GitHubApp,
        repo::AbstractString;
        sha::AbstractString,
        since::AbstractString,
    )
    return _paginated(gh, "/repos/$repo/commits?sha=$sha&since=$since")
end

"""Return all check runs for `ref`. Result paginated under `check_runs`."""
function list_check_runs(gh::GitHubApp, repo::AbstractString, ref::AbstractString)
    out = Any[]
    per_page = 100
    page = 1
    while true
        r = _request(
            gh,
            "GET",
            "/repos/$repo/commits/$ref/check-runs?per_page=$per_page&page=$page",
        )
        data = JSON3.read(String(r.body))
        runs = get(data, "check_runs", nothing)
        runs === nothing && return out
        append!(out, runs)
        length(runs) < per_page && return out
        page += 1
        page > 20 && return out
    end
    return
end

end # module
