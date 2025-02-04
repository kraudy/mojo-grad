"""
Simple sample generator for the mojo-grad implementation.
ref: https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/datasets/_samples_generator.py#L905
"""
from tensor import Tensor
from math import cos, sin, pi
from random import random_float64, seed, random_si64
from utils.index import Index

from python import Python, PythonObject

#inout can be used with initialized or uninitialized values, out with uninitialized ones
fn my_shuffle(mut arr: Tensor[DType.int64]):
    var n = arr.dim(0)
    for i in range(n - 1, 0, -1):
        var j = random_si64(0, i + 1)
        #var temp = arr[i]
        var temp = arr[i]  # Direct indexing for getting values
        arr[Index(i)] = arr[Index(j)]    # Direct assignment for setting values
        arr[Index(j)] = temp

#TODO: Fix for n = 3 nan error
fn linspace(start: Float64, stop: Float64, num: Int) -> Tensor[DType.float64]:
    var result = Tensor[DType.float64](num)
    var step = (stop - start) / Float64(num - 1)
    for i in range(num):
        result[i] = start + step * Float64(i)
    return result

fn normal(scale: Float64, size: Tuple[Int, Int]) -> Tensor[DType.float64]:
    var result = Tensor[DType.float64](size[0], size[1])
    for i in range(size[0]):
        for j in range(size[1]):
            result[Index(i, j)] = random_float64() * scale
    return result

fn apply_cos(t: Tensor[DType.float64]) -> Tensor[DType.float64]:
    var result = Tensor[DType.float64](t.dim(0))
    for i in range(t.dim(0)):
        result[i] = cos(t[i])
    return result

fn apply_sin(t: Tensor[DType.float64]) -> Tensor[DType.float64]:
    var result = Tensor[DType.float64](t.dim(0))
    for i in range(t.dim(0)):
        result[i] = sin(t[i])
    return result

fn make_moons(n_samples: Int = 100, shuffle_data: Bool = True, noise: Float64 = 0.0, random_seed: Int = 0) -> Tuple[Tensor[DType.float64], Tensor[DType.float64]]:
    seed(random_seed)

    var n_samples_out = n_samples // 2
    print(n_samples_out)
    var n_samples_in = n_samples - n_samples_out
    print(n_samples_in)

    var outer_circ_x = apply_cos(linspace(0, pi, n_samples_out))
    var outer_circ_y = apply_sin(linspace(0, pi, n_samples_out))
    var inner_circ_x = Tensor[DType.float64](1.0 - apply_cos(linspace(0, pi, n_samples_in)))
    var inner_circ_y = Tensor[DType.float64](1.0 - apply_sin(linspace(0, pi, n_samples_in)) - 0.5)#- Tensor[DType.float64](0.5)

    var X = Tensor[DType.float64](n_samples, 2)
    for i in range(n_samples_out):
        X[Index(i, 0)] = outer_circ_x[i]
        X[Index(i, 1)] = outer_circ_y[i]
    for i in range(n_samples_in):
        X[Index(n_samples_out + i, 0)] = inner_circ_x[i]
        X[Index(n_samples_out + i, 1)] = inner_circ_y[i]

    var y = Tensor[DType.float64](n_samples)
    for i in range(n_samples_out):
        y[i] = 0.0
    for i in range(n_samples_in):
        y[n_samples_out + i] = 1.0

    if shuffle_data:
        var indices = Tensor[DType.int64](n_samples)
        for i in range(n_samples):
            indices[i] = i
        #var indices = my_shuffle(range(n_samples))
        my_shuffle(indices)
        var X_shuffled = Tensor[DType.float64](n_samples, 2)
        var y_shuffled = Tensor[DType.float64](n_samples)
        for i in range(n_samples):
            #X_shuffled[i, 0] = X[indices[i], 0]
            X_shuffled[Index(i, 0)] = X[Index(indices[i], 0)]
            X_shuffled[Index(i, 1)] = X[Index(indices[i], 1)]
            y_shuffled[i] = y[Index(indices[i])]
        X = X_shuffled
        y = y_shuffled

    if noise > 0.0:
        var noise_tensor = normal(noise, (n_samples, 2))
        var X_with_noise = Tensor[DType.float64](n_samples, 2)
        for i in range(n_samples):
            for j in range(2):
                X_with_noise[Index(i, j)] = X[Index(i, j)] + noise_tensor[Index(i, j)]
        X = X_with_noise

    return (X, y)
