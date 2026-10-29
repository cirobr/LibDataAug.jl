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
)
    # wrap data
    img_wrap = wrap_data(img)
    # augmentation
    img_wrap = apply(intensity_tfm |> geometric_tfm, img_wrap)
    
    return unwrap_data(img_wrap)
end

function apply_data_augmentation(
    img::AbstractArray{<:Color{N0f8}},
    tgt::Union{AbstractArray{Int16}, AbstractArray{<:Color{N0f8}}};
    intensity_tfm,
    geometric_tfm,
)
    # wrap data
    img_wrap = wrap_data(img)
    tgt_wrap = wrap_data(tgt)
    # augmentation
    img_wrap = apply(intensity_tfm, img_wrap)
    img_wrap, tgt_wrap = apply(geometric_tfm, (img_wrap, tgt_wrap))

    return unwrap_data(img_wrap), unwrap_data(tgt_wrap)
end
