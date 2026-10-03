using Documenter
using DocumenterCodeBlocks
using DocumenterLandingPage
using Luximm

DocMeta.setdocmeta!(Luximm, :DocTestSetup, :(using Luximm); recursive = true)

makedocs(;
    modules = [Luximm],
    sitename = "Luximm.jl",
    authors = "Carroll Vance and contributors",
    repo = Documenter.Remotes.GitHub("csvance", "Luximm.jl"),
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://csvance.github.io/Luximm.jl",
        edit_link = "master",
        assets = String[],
        sidebar_sitename = false,
    ),
    pages = [
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
        "Porting Backbones" => "porting.md",
        "Testing" => "testing.md",
        "API Reference" => [
            "Overview" => "api/index.md",
            "Models" => "api/models.md",
            "Layers" => "api/layers.md",
            "Interop" => "api/interop.md",
        ],
    ],
    checkdocs = :exports,
    doctest = false,
    warnonly = [:missing_docs],
    plugins = [LandingPage(), CodeBlocks()],
)

deploydocs(; repo = "github.com/csvance/Luximm.jl")
