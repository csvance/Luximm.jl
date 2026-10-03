# HuggingFace Hub download helpers.
#
# Two entry points:
#
# * `hf_download(url, dest)` is a flat URL-to-path fetch. Streams into a
#   sibling .partial file and atomically renames on success.
#
# * `hf_hub_download(repo_id, filename; revision, cache_dir)` mirrors
#   the cache layout of `huggingface_hub` (Python). Files end up at
#   `<cache>/models--<org>--<name>/blobs/<etag>` with a relative symlink
#   at `<cache>/models--<org>--<name>/snapshots/<commit>/<filename>` and
#   the commit recorded in `refs/<revision>`. This means the same blob
#   is shared between Luximm and any `timm`/`huggingface_hub` install on
#   the same machine: whichever tool downloads first, the other sees a
#   cache hit.
#
# Both helpers send the `HUGGING_FACE_HUB_TOKEN` env var as a Bearer
# token when set (required for gated repos, harmless for public ones).

using Downloads.Curl: setopt, CURLOPT_FOLLOWLOCATION

# -- raw URL fetch -------------------------------------------------------

"""
    hf_download(url, dest) -> String

Download `url` to `dest` unless `dest` already exists. Returns `dest`.

If `HUGGING_FACE_HUB_TOKEN` is set, it is sent as a `Bearer` token in
the `Authorization` header. The download streams into a sibling temp
file and is renamed into place only on success, so an interrupted call
(network drop, ^C) never leaves a partial file at `dest` that a later
call would mistake for a cache hit.
"""
function hf_download(url::AbstractString, dest::AbstractString)
    isfile(dest) && return dest
    mkpath(dirname(dest))
    tmp = dest * ".partial-$(getpid())-$(rand(UInt32))"
    try
        Downloads.download(url, tmp; headers = _hf_headers())
        mv(tmp, dest; force = true)
    catch
        isfile(tmp) && rm(tmp; force = true)
        rethrow()
    end
    return dest
end

# -- HuggingFace-Hub-shaped cache ---------------------------------------

"""
    hf_hub_cache_dir() -> String

Cache root that matches `huggingface_hub` (Python). Honors the same
env-var precedence:

1. `HF_HUB_CACHE` if set,
2. otherwise `\$HF_HOME/hub` if `HF_HOME` is set,
3. otherwise `~/.cache/huggingface/hub`.

Files downloaded into this directory by Luximm are visible to `timm` /
`huggingface_hub`, and vice versa.
"""
function hf_hub_cache_dir()
    cache = get(ENV, "HF_HUB_CACHE", "")
    isempty(cache) || return cache
    home = get(ENV, "HF_HOME", "")
    isempty(home) || return joinpath(home, "hub")
    return joinpath(expanduser("~"), ".cache", "huggingface", "hub")
end

"""
    hf_hub_download(repo_id, filename; revision="main",
                     cache_dir=hf_hub_cache_dir(),
                     repo_type="model") -> String

Resolve `<repo_id>/<filename>` against `revision` (a branch, tag, or
commit) and return the local snapshot path, downloading only what is
not already cached. The on-disk layout matches `huggingface_hub`:

    <cache_dir>/models--<org>--<name>/blobs/<etag>
    <cache_dir>/models--<org>--<name>/snapshots/<commit>/<filename>
        -> ../../blobs/<etag>
    <cache_dir>/models--<org>--<name>/refs/<revision>     # text: <commit>

This means a `timm.create_model(..., pretrained=True)` call and a
`hf_hub_download` call against the same repo see each other's cached
blob.

The function always performs a no-redirect HEAD against
`https://huggingface.co/<repo_id>/resolve/<revision>/<filename>` to
look up the current commit (`X-Repo-Commit`) and the blob's etag
(`X-Linked-ETag` for LFS-backed files, `ETag` otherwise). If the HEAD
fails (e.g. offline), the function falls back to the most recently
recorded commit in `refs/<revision>` and returns the existing snapshot
path if present; otherwise the original error is rethrown.

Set `repo_type="dataset"` for dataset repos; default `"model"` matches
`timm`'s usage.
"""
function hf_hub_download(
        repo_id::AbstractString,
        filename::AbstractString;
        revision::AbstractString = "main",
        cache_dir::AbstractString = hf_hub_cache_dir(),
        repo_type::AbstractString = "model",
    )
    repo_prefix = repo_type == "model" ? "models" : repo_type * "s"
    repo_dir = joinpath(cache_dir, repo_prefix * "--" * replace(repo_id, "/" => "--"))
    refs_path = joinpath(repo_dir, "refs", revision)
    url = "https://huggingface.co/$(repo_id)/resolve/$(revision)/$(filename)"

    commit_sha, etag = try
        _hf_head_metadata(url)
    catch err
        # Offline fallback: trust the previously-recorded commit if any.
        if isfile(refs_path)
            cached_commit = strip(read(refs_path, String))
            cached_path =
                joinpath(repo_dir, "snapshots", String(cached_commit), filename)
            isfile(cached_path) && return cached_path
        end
        rethrow()
    end

    snap_dir = joinpath(repo_dir, "snapshots", commit_sha)
    snap_path = joinpath(snap_dir, filename)
    blob_path = joinpath(repo_dir, "blobs", etag)

    # Best-case: snapshot symlink already resolves to a real file.
    if ispath(snap_path)
        # Refresh refs to record the commit we just observed.
        _write_atomic(refs_path, commit_sha)
        return snap_path
    end

    mkpath(joinpath(repo_dir, "blobs"))
    mkpath(snap_dir)
    mkpath(joinpath(repo_dir, "refs"))

    if !isfile(blob_path)
        tmp = blob_path * ".partial-$(getpid())-$(rand(UInt32))"
        try
            Downloads.download(url, tmp; headers = _hf_headers())
            mv(tmp, blob_path; force = true)
        catch
            isfile(tmp) && rm(tmp; force = true)
            rethrow()
        end
    end

    if !ispath(snap_path)
        rel = relpath(blob_path, dirname(snap_path))
        try
            symlink(rel, snap_path)
        catch err
            # Symlink can race against a concurrent writer that created
            # the same link. If the link is already there, fine.
            ispath(snap_path) || rethrow()
        end
    end

    _write_atomic(refs_path, commit_sha)
    return snap_path
end

# -- helpers ------------------------------------------------------------

function _hf_headers()
    headers = Pair{String, String}[]
    token = get(ENV, "HUGGING_FACE_HUB_TOKEN", "")
    isempty(token) || push!(headers, "Authorization" => "Bearer $token")
    return headers
end

# Returns (commit_sha::String, etag::String). Uses a one-off Downloader
# with FOLLOWLOCATION=0 so we read HF's 302 directly (the CDN response
# strips X-Repo-Commit and X-Linked-Etag, so following the redirect
# loses the metadata we need).
function _hf_head_metadata(url::AbstractString)
    dl = Downloads.Downloader()
    dl.easy_hook = (easy, info) -> setopt(easy, CURLOPT_FOLLOWLOCATION, 0)
    resp = Downloads.request(
        url;
        method = "HEAD",
        headers = _hf_headers(),
        downloader = dl,
        throw = false,
    )
    if !(200 <= resp.status < 400)
        error("HEAD $url returned status $(resp.status)")
    end
    commit_sha = ""
    etag = ""
    linked_etag = ""
    for (k, v) in resp.headers
        kl = lowercase(k)
        kl == "x-repo-commit" && (commit_sha = String(v))
        kl == "etag" && (etag = String(v))
        kl == "x-linked-etag" && (linked_etag = String(v))
    end
    # Prefer the LFS etag; fall back to the small-file etag.
    blob_id = !isempty(linked_etag) ? linked_etag : etag
    blob_id = strip(blob_id, ('"', ' '))
    if isempty(commit_sha) || isempty(blob_id)
        error(
            "HEAD $url did not return commit/etag headers " *
                "(commit=$(commit_sha), etag=$(blob_id))",
        )
    end
    return (String(commit_sha), String(blob_id))
end

function _write_atomic(path::AbstractString, contents::AbstractString)
    mkpath(dirname(path))
    tmp = path * ".tmp-$(getpid())-$(rand(UInt32))"
    open(tmp, "w") do io
        write(io, contents)
    end
    mv(tmp, path; force = true)
    return path
end
