"""
Photometric transforms for DataAugmentation.jl.

All transforms preserve the input element type
(`RGB{N0f8}`, `Gray{N0f8}`, ...).
"""

"""
    AdjustSaturation(δ = 0.4)

Multiply the saturation of an RGB image by a factor sampled from `[1-δ, 1+δ]`.
Element type is preserved.
"""
struct AdjustSaturation <: Transform
    δ::Float32
end
AdjustSaturation(δ::Real = 0.4f0) = AdjustSaturation(Float32(δ))

DataAugmentation.getrandstate(tfm::AdjustSaturation) =
    1f0 + (rand(Float32) * 2f0 - 1f0) * tfm.δ

function DataAugmentation.apply(tfm::AdjustSaturation, item::Image;
                                randstate = DataAugmentation.getrandstate(tfm))
    img = itemdata(item)
    factor = randstate
    hsv = HSV.(img)
    for i in eachindex(hsv)
        s = clamp(hsv[i].s * factor, 0f0, 1f0)
        hsv[i] = HSV(hsv[i].h, s, hsv[i].v)
    end
    return Image(_preserve_eltype(img, hsv))
end


"""
    AdjustHue(δ = 0.1)

Shift the hue of an RGB image by a value sampled from `[-δ, δ]` (in turns).
Element type is preserved.
"""
struct AdjustHue <: Transform
    δ::Float32
end
AdjustHue(δ::Real = 0.1f0) = AdjustHue(Float32(δ))

DataAugmentation.getrandstate(tfm::AdjustHue) =
    (rand(Float32) * 2f0 - 1f0) * tfm.δ

function DataAugmentation.apply(tfm::AdjustHue, item::Image;
                                randstate = DataAugmentation.getrandstate(tfm))
    img = itemdata(item)
    shift = randstate
    hsv = HSV.(img)
    for i in eachindex(hsv)
        h = mod(hsv[i].h + shift, 1f0)
        hsv[i] = HSV(h, hsv[i].s, hsv[i].v)
    end
    return Image(_preserve_eltype(img, hsv))
end


"""
    Grayscale()

Convert an RGB image to grayscale and back to the original color type.
Use with `Maybe` for stochastic application.
Element type is preserved.
"""
struct Grayscale <: Transform end

function DataAugmentation.apply(::Grayscale, item::Image; randstate = nothing)
    img = itemdata(item)
    gray = Gray.(img)
    return Image(_preserve_eltype(img, gray))
end


"""
    Posterize(bits = 6)

Reduce the number of bits per color channel.
`bits` ∈ 1:8. Element type is preserved.
"""
struct Posterize <: Transform
    bits::Int
end
Posterize(bits::Integer = 6) = Posterize(clamp(Int(bits), 1, 8))

function DataAugmentation.apply(tfm::Posterize, item::Image; randstate = nothing)
    img = itemdata(item)
    shift = 8 - tfm.bits
    mask = UInt8(0xff << shift)

    out = similar(img)
    for i in eachindex(img)
        pix = img[i]
        if pix isa RGB
            r = reinterpret(N0f8, reinterpret(UInt8, pix.r) & mask)
            g = reinterpret(N0f8, reinterpret(UInt8, pix.g) & mask)
            b = reinterpret(N0f8, reinterpret(UInt8, pix.b) & mask)
            out[i] = RGB{N0f8}(r, g, b)
        else
            # Gray{N0f8} or similar
            v = reinterpret(eltype(pix), reinterpret(UInt8, pix) & mask)
            out[i] = v
        end
    end
    return Image(out)
end


"""
    Solarize(threshold = 0.5)

Invert all pixel values above `threshold` (solarization).
`threshold` is in [0, 1]. Element type is preserved.
Use with `Maybe` for stochastic application.
"""
struct Solarize <: Transform
    threshold::Float32
end
Solarize(threshold::Real = 0.5f0) = Solarize(Float32(clamp(threshold, 0, 1)))

function DataAugmentation.apply(tfm::Solarize, item::Image; randstate = nothing)
    img = itemdata(item)
    thr = tfm.threshold

    out = similar(img)
    for i in eachindex(img)
        pix = img[i]
        if pix isa RGB
            r = pix.r > thr ? 1f0 - pix.r : pix.r
            g = pix.g > thr ? 1f0 - pix.g : pix.g
            b = pix.b > thr ? 1f0 - pix.b : pix.b
            out[i] = RGB{N0f8}(r, g, b)
        else
            # Gray or other single-channel
            v = pix > thr ? 1f0 - pix : pix
            out[i] = oftype(pix, v)
        end
    end
    return Image(out)
end


"""
    AdjustSharpness(factor = 2.0)

Adjust image sharpness.
`factor = 1` leaves the image unchanged,
`factor > 1` sharpens,
`factor < 1` blurs.
Element type is preserved.
Requires `ImageFiltering`.
Use with `Maybe` for stochastic application.
"""
struct AdjustSharpness <: Transform
    factor::Float32
end
AdjustSharpness(factor::Real = 2.0f0) = AdjustSharpness(Float32(factor))

function DataAugmentation.apply(tfm::AdjustSharpness, item::Image; randstate = nothing)
    img = itemdata(item)
    factor = tfm.factor

    # Mild blur used as the "unsharp" base
    blurred = ImageFiltering.imfilter(img, ImageFiltering.Kernel.gaussian(1.0))

    # Unsharp mask: sharpened = img + factor * (img - blurred)
    # Equivalent to: (1 + factor) * img - factor * blurred
    sharpened = (1f0 + factor) .* img .- factor .* blurred

    # Clamp to valid range and restore element type
    sharpened = clamp01.(sharpened)
    return Image(_preserve_eltype(img, sharpened))
end


"""
    GaussianBlur(σ = (0.1, 2.0))

Apply a Gaussian blur with σ sampled from the given interval.
Requires `ImageFiltering`.
Element type is preserved.
"""
struct GaussianBlur <: Transform
    σmin::Float32
    σmax::Float32
end
GaussianBlur(σ::NTuple{2,Real} = (0.1f0, 2.0f0)) =
    GaussianBlur(Float32(σ[1]), Float32(σ[2]))

DataAugmentation.getrandstate(tfm::GaussianBlur) =
    tfm.σmin + rand(Float32) * (tfm.σmax - tfm.σmin)

function DataAugmentation.apply(tfm::GaussianBlur, item::Image;
                                randstate = DataAugmentation.getrandstate(tfm))
    img = itemdata(item)
    σ = randstate
    blurred = ImageFiltering.imfilter(img, ImageFiltering.Kernel.gaussian(σ))
    return Image(_preserve_eltype(img, blurred))
end