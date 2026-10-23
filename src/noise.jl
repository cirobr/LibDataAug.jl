"""
Gaussian noise transform for DataAugmentation.jl.

Applies additive Gaussian noise to an `ArrayItem` (the tensor produced by
`ImageToTensor` / `Normalize`). The standard deviation is in the same units
as the tensor values. Use after normalization so `sigma` is independent of
the original pixel range.

Validation sets should pass `sigma=0` (or omit the transform) to stay
deterministic.
"""

struct GaussianNoise <: Transform
    sigma::Float32
end

"""
    GaussianNoise(sigma=0.08f0)

Additive Gaussian noise. `sigma` is the standard deviation in tensor units.
`sigma=0` is a no-op.
"""
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
