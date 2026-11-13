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