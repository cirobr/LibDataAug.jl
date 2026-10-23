# wrap
function wrap_data(img::AbstractArray{<:Color{N0f8}})
      return Image(img)
end

function wrap_data(img::AbstractArray{<:Color{N0f8}}, mask::AbstractArray{Int16})
    return Image(img), MaskMulti(mask .+ 1, 1:256)
end


# unwrap
function unwrap_data(img_wrap)
      return OffsetArrays.no_offset_view(img_wrap.data)
end

function unwrap_data(img_wrap, mask_wrap)
    img  = OffsetArrays.no_offset_view(img_wrap.data)
    mask = OffsetArrays.no_offset_view((mask_wrap.data))
    return img, mask
end


# augmentation
function apply_data_augmentation(
      img::AbstractArray{<:Color{N0f8}};
      intensity_tfm,
      geometric_tfm,
      image_tfm=ImageToTensor(),
)
      img_wrap = wrap_data(img)
      img_wrap = apply(intensity_tfm |> geometric_tfm |> image_tfm, img_wrap)
      return unwrap_data(img_wrap)
end

function apply_data_augmentation(
    img::AbstractArray{<:Color{N0f8}},
    mask::AbstractArray{Int16};
    intensity_tfm,
    geometric_tfm,
    image_tfm=ImageToTensor(),
    mask_tfm=OneHot(),
)
    img_wrap, mask_wrap = wrap_data(img, mask)
    # augmentation
    img_wrap = apply(intensity_tfm, img_wrap)                           # intensity (img only)
    img_wrap, mask_wrap = apply(geometric_tfm, (img_wrap, mask_wrap))   # geometric (img, mask)
    # tensor representations
    img_wrap  = apply(image_tfm, img_wrap)
    mask_wrap = apply(mask_tfm, mask_wrap)
    # augmented output
    return unwrap_data(img_wrap, mask_wrap)
end