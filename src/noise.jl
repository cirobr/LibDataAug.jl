"""
    GaussianNoise(sigma = 0.08f0)

Additive Gaussian noise for `ArrayItem` (the tensor produced by
`ImageToTensor` / `Normalize`).

`sigma` is the standard deviation in tensor units.
Apply after normalization so the noise scale is independent of the
original pixel range. `sigma = 0` is a no-op.

Validation sets should omit this transform (or pass `sigma = 0`).
"""
struct GaussianNoise <: Transform
    sigma::Float32
end

GaussianNoise(sigma::Real = 0.08f0) = GaussianNoise(Float32(sigma))

function apply(tfm::GaussianNoise, item::ArrayItem; randstate = nothing)
    σ = tfm.sigma
    σ == 0 && return item
    x = itemdata(item)
    noise = randn(eltype(x), size(x)) .* σ
    return ArrayItem(x .+ noise)
end

function apply(tfm::GaussianNoise, item::Item; randstate = nothing)
    error("GaussianNoise expects an ArrayItem (apply after ImageToTensor)")
end
