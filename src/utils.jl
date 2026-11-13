# ---------- wrap / unwrap ----------

"""
    wrap_data(img::AbstractArray{<:Color{N0f8}})

Wrap an image as a `DataAugmentation.Image`.
"""
function wrap_data(img::AbstractArray{<:Color{N0f8}})
    return Image(img)
end

"""
    wrap_data(tgt::AbstractArray{Int16})

Wrap an integer mask as a `MaskMulti`.
Labels are shifted by +1 so that class values start at 1
(required by `MaskMulti`). The original 0 becomes 1, void 255 becomes 256, etc.
"""
function wrap_data(tgt::AbstractArray{Int16})
    return MaskMulti(tgt .+ 1, 1:256)
end

"""
    unwrap_data(wrap)

Extract the underlying array from an `Image` or `MaskMulti`,
removing any offset axes.
"""
function unwrap_data(wrap)
    return OffsetArrays.no_offset_view(wrap.data)
end

# ---------- apply_data_augmentation ----------

"""
    apply_data_augmentation(img; intensity_tfm, geometric_tfm)

Apply intensity then geometric transforms to a single image.
Returns the unwrapped image array (element type preserved).
"""
function apply_data_augmentation(
    img::AbstractArray{<:Color{N0f8}};
    intensity_tfm,
    geometric_tfm,
)
    img_wrap = wrap_data(img)
    img_wrap = apply(intensity_tfm |> geometric_tfm, img_wrap)
    return unwrap_data(img_wrap)
end

"""
    apply_data_augmentation(img, tgt; intensity_tfm, geometric_tfm)

Apply intensity transforms to the image only, then geometric transforms
to both the image and the mask (paired).
Returns `(image, mask)` with original element types preserved.
"""
function apply_data_augmentation(
    img::AbstractArray{<:Color{N0f8}},
    tgt::Union{AbstractArray{Int16}, AbstractArray{<:Color{N0f8}}};
    intensity_tfm,
    geometric_tfm,
)
    img_wrap = wrap_data(img)
    tgt_wrap = wrap_data(tgt)

    img_wrap = apply(intensity_tfm, img_wrap)
    img_wrap, tgt_wrap = apply(geometric_tfm, (img_wrap, tgt_wrap))

    return unwrap_data(img_wrap), unwrap_data(tgt_wrap)
end
