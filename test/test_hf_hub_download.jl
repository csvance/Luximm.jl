# Verifies that hf_hub_download:
#   1. Uses the HuggingFace Hub cache layout (blobs/<etag>,
#      snapshots/<commit>/<file>, refs/<revision>) so the same blob
#      is shared with timm / huggingface_hub.
#   2. Resolves a fresh `cache_dir` from network the first time and
#      short-circuits on subsequent calls.
#   3. Falls back to the previously-recorded commit/snapshot when the
#      HEAD lookup fails (e.g. offline).

using Test
using Luximm

@testset "hf_hub_download (HF cache layout)" begin
    if get(ENV, "HF_OFFLINE", "") == "1"
        @info "skipping: HF_OFFLINE=1"
        return
    end

    # tiny stable file from one of the BiT repos: ~570B
    repo = "timm/resnetv2_50x1_bit.goog_in21k"
    filename = "config.json"

    cache = mktempdir(; cleanup = false)
    try
        # Round 1: cold cache, must hit the network.
        path = hf_hub_download(repo, filename; cache_dir = cache)
        @test isfile(path)
        @test islink(path)

        repo_dir = joinpath(cache, "models--timm--resnetv2_50x1_bit.goog_in21k")
        @test isdir(joinpath(repo_dir, "blobs"))
        @test isdir(joinpath(repo_dir, "snapshots"))
        @test isfile(joinpath(repo_dir, "refs", "main"))

        commit_sha = strip(read(joinpath(repo_dir, "refs", "main"), String))
        @test length(commit_sha) >= 7         # looks like a sha
        @test startswith(path, joinpath(repo_dir, "snapshots", commit_sha))

        # The symlink target should sit under blobs/ and be a real file.
        blob_path = realpath(path)
        @test startswith(blob_path, joinpath(repo_dir, "blobs"))
        @test isfile(blob_path)
        @test filesize(blob_path) > 0

        # Round 2: warm cache, same call must return same path without
        # re-downloading (we can't observe network traffic, but we can
        # at least confirm the path is stable and the blob unchanged).
        mtime_before = mtime(blob_path)
        sleep(0.01)
        path2 = hf_hub_download(repo, filename; cache_dir = cache)
        @test path2 == path
        @test mtime(realpath(path2)) == mtime_before  # blob untouched

        # Round 3: simulate HEAD failure (e.g. offline) by pointing the
        # repo at a URL that does not exist. The cached refs/main must
        # let us short-circuit and return the existing snapshot path.
        bad_repo = "definitely-not-a-real-org/$(basename(tempname()))"
        bad_repo_dir = joinpath(cache, "models--" * replace(bad_repo, "/" => "--"))
        mkpath(joinpath(bad_repo_dir, "snapshots", commit_sha))
        mkpath(joinpath(bad_repo_dir, "refs"))
        write(joinpath(bad_repo_dir, "refs", "main"), commit_sha)
        fake_snap = joinpath(bad_repo_dir, "snapshots", commit_sha, filename)
        # Plant a file at the expected snapshot location so the
        # offline fallback has something to return.
        write(fake_snap, "cached content for offline fallback")
        offline_path = hf_hub_download(bad_repo, filename; cache_dir = cache)
        @test offline_path == fake_snap
    finally
        rm(cache; recursive = true, force = true)
    end
end

@testset "hf_hub_cache_dir env precedence" begin
    saved = (get(ENV, "HF_HUB_CACHE", nothing), get(ENV, "HF_HOME", nothing))
    try
        delete!(ENV, "HF_HUB_CACHE")
        delete!(ENV, "HF_HOME")
        @test hf_hub_cache_dir() ==
            joinpath(expanduser("~"), ".cache", "huggingface", "hub")

        ENV["HF_HOME"] = "/tmp/jimm_hf_home_test"
        @test hf_hub_cache_dir() == "/tmp/jimm_hf_home_test/hub"

        ENV["HF_HUB_CACHE"] = "/tmp/jimm_hf_hub_cache_test"
        @test hf_hub_cache_dir() == "/tmp/jimm_hf_hub_cache_test"
    finally
        # restore env exactly as we found it
        saved[1] === nothing ? delete!(ENV, "HF_HUB_CACHE") :
            (ENV["HF_HUB_CACHE"] = saved[1])
        saved[2] === nothing ? delete!(ENV, "HF_HOME") : (ENV["HF_HOME"] = saved[2])
    end
end
