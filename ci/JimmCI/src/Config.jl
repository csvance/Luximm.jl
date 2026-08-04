module ConfigMod

export Config, repo_fullname

"""
    Config

Environment-driven configuration. Mirrors the env-var contract of the
previous Python runner so `<state>/` layouts remain compatible.
"""
struct Config
    app_id::Int
    installation_id::Int
    private_key::String

    repo_owner::String
    repo_name::String

    state_dir::String
    julia_depot::String
    hf_cache::String
    mirror_dir::String
    workspace_dir::String
    log_dir::String
    parity_dir::String
    python_env::String

    julia_binary::String
    uv_binary::String
    hf_token::Union{String,Nothing}
end

repo_fullname(c::Config) = string(c.repo_owner, "/", c.repo_name)

function _require_env(name::AbstractString)
    val = get(ENV, name, nothing)
    val === nothing && error("$name is required but not set")
    return val
end

function _read_file_stripped(path::AbstractString)
    return strip(read(path, String))
end

# Resolve the `uv` executable the same way `julia_binary` is resolved: an
# absolute path, not a bare name off PATH. The builder frequently runs from a
# service manager whose PATH does not include the per-user install dir uv's
# installer writes to (`~/.local/bin`), and a bare "uv" then fails at spawn
# with a bare ENOENT that says nothing about what is missing. Order: explicit
# override, then PATH, then the install locations ci/README.md documents.
function _default_uv_binary()
    override = get(ENV, "JIMM_CI_UV", nothing)
    override === nothing || return override
    found = Sys.which("uv")
    found === nothing || return String(found)
    for candidate in
        ("/usr/local/bin/uv", joinpath(homedir(), ".local", "bin", "uv"), "/usr/bin/uv")
        isfile(candidate) && return candidate
    end
    # Nothing resolved; keep the bare name so the error surfaces at the call
    # site, where `_uv_cmd` explains the fix.
    return "uv"
end

function from_env()
    state = get(ENV, "JIMM_CI_STATE_DIR", "/var/lib/jimm-ci")
    key_path = _require_env("JIMM_CI_PRIVATE_KEY_FILE")

    hf_token = get(ENV, "HF_TOKEN", nothing)
    hf_token_file = get(ENV, "JIMM_CI_HF_TOKEN_FILE", nothing)
    if hf_token === nothing && hf_token_file !== nothing
        hf_token = String(_read_file_stripped(hf_token_file))
    end

    return Config(
        parse(Int, _require_env("JIMM_CI_APP_ID")),
        parse(Int, _require_env("JIMM_CI_INSTALLATION_ID")),
        read(key_path, String),
        _require_env("JIMM_CI_REPO_OWNER"),
        _require_env("JIMM_CI_REPO_NAME"),
        state,
        get(ENV, "JULIA_DEPOT_PATH", joinpath(state, "julia-depot")),
        get(ENV, "HF_HUB_CACHE", joinpath(state, "hf-cache")),
        joinpath(state, "mirror.git"),
        joinpath(state, "work"),
        joinpath(state, "logs"),
        get(ENV, "JIMM_CI_PARITY_DIR", joinpath(state, "parity")),
        get(ENV, "UV_PROJECT_ENVIRONMENT", joinpath(state, "python-env")),
        get(ENV, "JIMM_CI_JULIA", "/usr/local/bin/julia"),
        _default_uv_binary(),
        hf_token,
    )
end

end # module
