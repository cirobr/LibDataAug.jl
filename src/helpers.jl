"""
    _preserve_eltype(img, out)

Convert `out` to the element type of `img`.
Used by photometric and perspective transforms to keep RGB{N0f8} / Gray{N0f8}.
"""
_preserve_eltype(img, out) = oftype(img, out)
