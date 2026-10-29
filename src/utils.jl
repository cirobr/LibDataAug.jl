# wrap
function wrap_data(img::AbstractArray{<:Color{N0f8}})
      return Image(img)
end

function wrap_data(tgt::AbstractArray{Int16})
    return MaskMulti(tgt .+ 1, 1:256)
end


# unwrap
function unwrap_data(wrap)
      return OffsetArrays.no_offset_view(wrap.data)
end


# augmentation
function apply_data_augmentation(
    img::AbstractArray{<:Color{N0f8}};
    intensity_tfm,
    geometric_tfm,
    # image_to_tensor_tfm=ImageToTensor(),
)
    # wrap data
    img_wrap = wrap_data(img)
    # augmentation & tensor representation
    img_wrap = apply(intensity_tfm |> geometric_tfm |> ImageToTensor(), img_wrap)
    # augmented output
    return unwrap_data(img_wrap)
end

function apply_data_augmentation(
    img::AbstractArray{<:Color{N0f8}},
    tgt::Union{AbstractArray{Int16}, AbstractArray{<:Color{N0f8}}};
    intensity_tfm,
    geometric_tfm,
    # image_to_tensor_tfm=ImageToTensor(),
    target_to_tensor_tfm,
)
    # wrap data
    img_wrap = wrap_data(img)
    tgt_wrap = wrap_data(tgt)
    # augmentation
    img_wrap = apply(intensity_tfm, img_wrap)
    img_wrap, tgt_wrap = apply(geometric_tfm |> ImageToTensor(), (img_wrap, tgt_wrap))
    # tensor representations
    # img_wrap  = apply(ImageToTensor(), img_wrap)
    # tgt_wrap  = apply(ImageToTensor(), tgt_wrap)
    # augmented output
    return unwrap_data(img_wrap), unwrap_data(tgt_wrap)
end