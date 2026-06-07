#!/usr/bin/env julia
#
# Minimal ImageNet top-5 demo for a Luximm in1k classifier.
#
# Loads a JPEG, applies the standard timm/torchvision eval transform
# (resize-shorter-side + center crop + ImageNet normalization), runs a
# pretrained 1000-class model, and prints the top-5 classes with confidences.
#
# Usage:
#   julia --project=utils utils/predict.jl <image.jpg> [variant]
#
# Examples:
#   julia --project=utils utils/predict.jl cat.jpg
#   julia --project=utils utils/predict.jl cat.jpg resnet18_a1_in1k
#   julia --project=utils utils/predict.jl cat.jpg coatnet_0_rw_224_sw_in1k
#
# `variant` defaults to :resnet50_a1_in1k. It must be an in1k (1000-class)
# variant from one of Luximm's *_VARIANTS tables; the script asserts the
# released head is 1000-way.

# CoAtNet *_in12k repos use HF Xet storage, which stalls on some networks; the
# in1k variants don't, but set this defensively so any variant downloads cleanly.
get(ENV, "HF_HUB_DISABLE_XET", "") == "" && (ENV["HF_HUB_DISABLE_XET"] = "1")

using FileIO            # load() — JPEG decoded via the ImageIO backend
using ImageCore         # RGB color type + channelview
using ImageTransformations: imresize   # JuliaImages spatial resize
using Downloads         # fetch the ImageNet class-label list
using Random: Xoshiro
using NNlib: softmax
using Lux
using Luximm

# timm/torchvision ImageNet normalization (the default for the in1k recipes).
const IMAGENET_MEAN = (0.485f0, 0.456f0, 0.406f0)
const IMAGENET_STD = (0.229f0, 0.224f0, 0.225f0)
# Center-crop ratio: resize the shorter side to input/CROP_PCT, then crop.
const CROP_PCT = 0.875f0
# 1000 human-readable labels, one per line, in ImageNet class-index order
# (line 1 == class 0), matching the order Luximm models emit logits in.
const LABELS_URL =
    "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"

# Full preprocessing: load → resize shorter side to input/CROP_PCT → center
# crop `input` → normalize → WHCN tensor `(input, input, 3, 1)`.
#
# Uses JuliaImages: `imresize` (bilinear by default) for the spatial resize and
# `channelview` to split the RGB pixels into a `(3, H, W)` numeric array. See
# https://juliaimages.org/latest/examples/spatial_transformation/SpatialTransformations/.
function preprocess(path::AbstractString, input::Int)
    # `RGB{Float32}.` normalizes any input (grayscale, RGBA, …) to 3-channel
    # RGB in [0, 1] before resizing.
    img = RGB{Float32}.(load(path))                # (H, W) RGB{Float32}
    H, W = size(img)
    resize_to = round(Int, input / CROP_PCT)
    scale = resize_to / min(H, W)
    img = imresize(img, (round(Int, H * scale), round(Int, W * scale)))
    # center crop input×input
    rH, rW = size(img)
    top = (rH - input) ÷ 2 + 1
    left = (rW - input) ÷ 2 + 1
    crop = img[top:top+input-1, left:left+input-1]
    ch = channelview(crop)                         # (3, input, input): [c, h, w]
    # → WHCN, normalized: x[w, h, c, 1] = (pixel[c, h, w] - mean) / std
    x = Array{Float32}(undef, input, input, 3, 1)
    @inbounds for c = 1:3, h = 1:input, w = 1:input
        x[w, h, c, 1] = (Float32(ch[c, h, w]) - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
    end
    return x
end

function imagenet_labels()
    cache = joinpath(@__DIR__, "imagenet_classes.txt")
    isfile(cache) || Downloads.download(LABELS_URL, cache)
    return readlines(cache)
end

function main(args)
    isempty(args) && error(
        "usage: julia --project=utils utils/predict.jl <image.jpg> [variant]",
    )
    path = args[1]
    isfile(path) || error("image not found: $path")
    variant = length(args) >= 2 ? Symbol(args[2]) : :resnet50_a1_in1k

    nc = default_num_classes(variant)
    nc == 1000 || error(
        "variant $variant ships a $(nc)-class head; this demo expects an " *
        "in1k (1000-class) classifier.",
    )

    @info "building $variant and loading pretrained weights…"
    model, load_weights = create_pretrained(variant)   # num_classes defaults to 1000
    ps, st = Lux.setup(Xoshiro(0), model)
    ps, st = load_weights(ps, st)
    st = Lux.testmode(st)

    # Infer the model's native input resolution from its variant config.
    input = _input_size(variant)
    x = preprocess(path, input)

    logits, _ = model(x, ps, st)                       # (1000, 1)
    probs = softmax(vec(logits))
    labels = imagenet_labels()

    order = sortperm(probs; rev = true)
    println("\nTop-5 predictions for $(basename(path))  ($(variant), $(input)×$(input)):")
    for (rank, ci) in enumerate(order[1:5])
        cls = ci - 1                                   # 1-based Julia idx → 0-based class id
        label = ci <= length(labels) ? labels[ci] : "class $cls"
        println("  $rank. $(rpad(label, 28)) $(round(100 * probs[ci]; digits = 2))%  [class $cls]")
    end
end

# Native input side length for `variant`, read from whichever *_VARIANTS table
# owns it. All registered in1k variants train at a square resolution.
function _input_size(variant::Symbol)
    for tbl in (
        Luximm.RESNET_VARIANTS,
        Luximm.SERESNET_VARIANTS,
        Luximm.VGG_VARIANTS,
        Luximm.VIT_VARIANTS,
        Luximm.COATNET_VARIANTS,
        Luximm.BIT_VARIANTS,
        Luximm.CONVNEXT_VARIANTS,
        Luximm.CONVNEXTV2_VARIANTS,
    )
        haskey(tbl, variant) && return tbl[variant].default_input_size
    end
    return 224
end

main(ARGS)
