module LibDataAug


# libraries
export ClassCentricCrop, RandomCentricCrop, GaussianNoise

import DataAugmentation
import DataAugmentation: apply, getrandstate, itemdata,
    Transform, Image, MaskMulti, Item, ArrayItem, ImageToTensor,
    OneOf, ScaleKeepAspect, Crop, PinOrigin
import OffsetArrays
import Colors: Color, HSV, Gray, RGB, N0f8
import ImageFiltering

include("helpers.jl")
include("utils.jl")

include("crops.jl")
include("noise.jl")
include("photometric.jl")


end