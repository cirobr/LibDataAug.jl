module LibDataAug


# libraries
export ClassCentricCrop, RandomCentricCrop, GaussianNoise

import DataAugmentation
import DataAugmentation: apply, getrandstate, itemdata,
    Transform, Image, MaskMulti, Item, ArrayItem,
    OneOf, ScaleKeepAspect, Crop, PinOrigin
import OffsetArrays
import Colors: Color, RGB, N0f8

include("utils.jl")
include("crops.jl")
include("noise.jl")


end
