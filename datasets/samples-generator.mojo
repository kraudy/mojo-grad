"""
Simple sample generator for the mojo-grad implementation.
ref: https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/datasets/_samples_generator.py#L905
"""
from tensor import Tensor
from math import cos, sin, pi
from random import random_float64, seed, random_si64
from utils.index import Index

#inout can be used with initialized or uninitialized values, out with uninitialized ones
fn shuffle(inout arr: Tensor[DType.int64]):
    var n = arr.dim(0)
    for i in range(n - 1, 0, -1):
        var j = random_si64(0, i + 1)
        #var temp = arr[i]
        var temp = arr[i]  # Direct indexing for getting values
        arr[Index(i)] = arr[Index(j)]    # Direct assignment for setting values
        arr[Index(j)] = temp

fn linspace(start: Float64, stop: Float64, num: Int) -> Tensor[DType.float64]:
    var result = Tensor[DType.float64](num)
    var step = (stop - start) / Float64(num - 1)
    for i in range(num):
        result[i] = start + step * Float64(i)
    return result

fn normal(scale: Float64, size: Tuple[Int, Int]) -> Tensor[DType.float64]:
    #var result = Tensor[DType.float64](size.get[0](), size.get[1]())
    var result = Tensor[DType.float64](size[0], size[1])
    for i in range(size[0]):
        for j in range(size[1]):
            result[Index(i, j)] = random_float64() * scale
    return result

fn make_moons(n_samples: Int = 100, shuffle:Bool = True, noise:Float64 = 0.0, random_state:Int = 0):
    pass