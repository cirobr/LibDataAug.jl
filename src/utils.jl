# pair image/mask augmentation
function wrap_pair(img, mask)
      return Image(img), MaskMulti(mask .+ 1, 1:256)
end

function unwrap_pair(img_wrap, mask_wrap; classnrs)
      img  = OffsetArrays.no_offset_view(img_wrap.data)
      mask = OffsetArrays.no_offset_view((mask_wrap.data .- 1) .|> Int16)
      mask = map(v -> v in classnrs ? Int16(v) : Int16(255), mask)
      return img, mask
end

function apply_data_augmentation(
      img::AbstractArray{<:Color{N0f8}},
      mask::AbstractArray{Int16};
      intensity_tfm,
      geometric_tfm,
      classnrs,
)
      img_wrap, mask_wrap = wrap_pair(img, mask)
      img_wrap = apply(intensity_tfm, img_wrap)   # intensity (img only)
      img_wrap, mask_wrap = apply(geometric_tfm, (img_wrap, mask_wrap))   # geometric (img, mask)

      return unwrap_pair(img_wrap, mask_wrap; classnrs=classnrs)   # (aug img, aug mask)
end


# image augmentation
function wrap_img(img)
      return Image(img)
end

function unwrap_img(img_wrap)
      return OffsetArrays.no_offset_view(img_wrap.data) # .|> RGB{N0f8}
end

function apply_img_augmentation(
      img::AbstractArray{<:Color{N0f8}};
      intensity_tfm,
      geometric_tfm,
)
      img_wrap = wrap_img(img)
      img_wrap = apply(intensity_tfm, img_wrap)
      img_wrap = apply(geometric_tfm, img_wrap)

      return unwrap_img(img_wrap)
end