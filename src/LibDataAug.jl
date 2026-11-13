module LibDataAug


export ClassCentricCrop, RandomCentricCrop, GaussianNoise
export AdjustSaturation, AdjustHue, Grayscale, Posterize, Solarize, AdjustSharpness, GaussianBlur
export RandomPerspective

import DataAugmentation
import DataAugmentation: apply, getrandstate, itemdata,
    Transform, Image, MaskMulti, Item, ArrayItem, ImageToTensor,
    OneOf, ScaleKeepAspect, Crop, PinOrigin
import OffsetArrays
import Colors: Color, HSV, Gray, RGB, N0f8
import ImageFiltering
import ImageTransformations
import CoordinateTransformations
import StaticArrays: SMatrix, SVector
import LinearAlgebra: svd

include("helpers.jl")
include("utils.jl")

include("crops.jl")
include("noise.jl")
include("photometric.jl")
include("perspective.jl")


end # module
