"""
Simple sample generator for the mojo-grad implementation.
ref: https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/datasets/_samples_generator.py#L905
"""
from tensor import Tensor
from math import cos, sin, pi
from random import random_float64, seed, random_si64
from utils.index import Index
from sys.info import simdwidthof
from memory import UnsafePointer, memset_zero

from python import Python, PythonObject

alias type = DType.float64
alias nelts = Int(simdwidthof[type]())

#inout can be used with initialized or uninitialized values, out with uninitialized ones
fn my_shuffle(mut arr: Tensor[DType.int64]):
    var n = arr.dim(0)
    for i in range(n - 1, 0, -1):
        var j = random_si64(0, i + 1)
        var temp = arr[i]  # Direct indexing for getting values
        arr[Index(i)] = arr[Index(j)]    # Direct assignment for setting values
        arr[Index(j)] = temp

fn linspace(n: Int, residue: Int, start: Float64, stop: Float64) -> UnsafePointer[Scalar[type]]:
    # Allocate memory for result
    var result = UnsafePointer[Scalar[type]].alloc(n)
    memset_zero(result, n)
    
    # Calculate step size once
    var step = (stop - start) / Float64(n - 1)
    
    # Fill the array using SIMD operations
    for i in range(0, n - residue, nelts):
        # Create SIMD vector from sequential indices
        var indices = SIMD[type, nelts]()
        # Fill the indices for this chunk
        for j in range(nelts):
            var idx = i + j
            #if idx < n:  # So we don't get out of the n bound
            indices[j] = Float64(idx)
        # Calculate and store values
        var values = start + step * indices
        result.offset(i).store(values)
    
    if residue != 0:
        for i in range(residue):
            result[(n - residue) + i] = start + step * ((n - residue) + i)
    
    return result

fn normal(scale: Float64, size: Tuple[Int, Int]) -> Tensor[DType.float64]:
    var result = Tensor[DType.float64](size[0], size[1])
    for i in range(size[0]):
        for j in range(size[1]):
            result[Index(i, j)] = random_float64() * scale
    return result

fn apply_cos(n: Int, residue: Int, t: UnsafePointer[Scalar[type]]) -> UnsafePointer[Scalar[type]]:
    # Allocate new memory for the result
    var result = UnsafePointer[Scalar[type]].alloc(n)
    memset_zero(result, n)
    

    # Apply cos using SIMD operations
    for i in range(0, n - residue, nelts):
        # Load SIMD chunk from input
        var values = t.offset(i).load[width=nelts]()
        # Apply cos to all elements in the SIMD vector
        var cos_values = cos(values)
        # Store the results
        result.offset(i).store(cos_values)
    
    if residue != 0:
        for i in range(residue):
            result[(n - residue) + i] = cos(t[(n - residue) + i])
    
    return result


fn apply_sin(n: Int, residue: Int, t: UnsafePointer[Scalar[type]]) -> UnsafePointer[Scalar[type]]:
    # Allocate new memory for the result
    print("befor allocation")
    print("N: ", n)
    var result = UnsafePointer[Scalar[type]].alloc(n)
    print("befor memeset_zero")
    memset_zero(result, n)
    print("pass allocation")

    for i in range(0, n - residue, nelts):
        print("apply_sin i: ", i)
        # Load SIMD chunk from input
        var values = t.offset(i).load[width=nelts]()
        # Apply cos to all elements in the SIMD vector
        var sin_values = sin(values)
        # Store the results
        result.offset(i).store(sin_values)

    if residue != 0:
        for i in range(residue):
            result[(n - residue) + i] = sin(t[(n - residue) + i])

    return result

fn make_moons(n_samples: Int = 100, shuffle_data: Bool = True, noise: Float64 = 0.0, random_seed: Int = 0) -> Tuple[Tensor[DType.float64], Tensor[DType.float64]]:
    seed(random_seed)

    var n_samples_out = n_samples // 2
    var residue_out = n_samples_out % nelts
    print(n_samples_out)
    var n_samples_in = n_samples - n_samples_out
    var residue_in = n_samples_in % nelts
    print(n_samples_in)

    print("outer_circ_x, outer_circ_y")
    var outer_circ_x = apply_cos(n_samples_out, residue_out, linspace(n_samples_out, residue_out, 0, pi))
    var outer_circ_y = apply_sin(n_samples_out, residue_out, linspace(n_samples_out, residue_out, 0, pi))

    print("cos_inner_circ_x")
    var cos_inner_circ_x = apply_cos(n_samples_in, residue_in, linspace(n_samples_in, residue_in, 0, pi))
    print("sin_inner_circ_y")
    var sin_inner_circ_y = apply_sin(n_samples_in, residue_in, linspace(n_samples_in, residue_in, 0, pi))
    print("pasa sin_inner_circ_y")
#
    #var inner_circ_x = UnsafePointer[Scalar[type]].alloc(n_samples_out)
    var inner_circ_x = UnsafePointer[Scalar[type]].alloc(n_samples_in)
    memset_zero(inner_circ_x, n_samples_in)  # Initialize to zero
    print("="*25)
    print("inner_circ_x: ", n_samples_in)
    for i in range(0, n_samples_in, nelts): 
        # Load SIMD chunk from input
        var values = cos_inner_circ_x.offset(i).load[width=nelts]()
        var results = 1 - values
        inner_circ_x.offset(i).store(results)
    if residue_in != 0:
        for i in range(residue_in):
            inner_circ_x[(n_samples_in - residue_in) + i] = 1 - cos_inner_circ_x[(n_samples_in - residue_in) + i]
#
#
    var inner_circ_y = UnsafePointer[Scalar[type]].alloc(n_samples_in)
    memset_zero(inner_circ_y, n_samples_in)  # Initialize to zero
    for i in range(0, n_samples_in, nelts): 
        # Load SIMD chunk from input
        var values = sin_inner_circ_y.offset(i).load[width=nelts]()
        var results = 0.5 - values #- 0.5
        inner_circ_y.offset(i).store(results)
    if residue_in != 0:
        for i in range(residue_in):
            inner_circ_y[(n_samples_in - residue_in) + i] = 0.5 - sin_inner_circ_y[(n_samples_in - residue_in) + i]
    
#
    ##var inner_circ_y = (1.0 - sin_inner_circ_y - 0.5)#- Tensor[DType.float64](0.5)
    #print("pasa pointers")
#
    print("="*50)

    var X = Tensor[DType.float64](n_samples, 2)
    for i in range(n_samples_out):
        X[Index(i, 0)] = outer_circ_x[i]
        X[Index(i, 1)] = outer_circ_y[i]
    print("Tensor i: ", n_samples_in)
    for i in range(n_samples_in):
        print("i:", i)
        print("X index:", n_samples_out + i)
        print("inner_circ_x[i]:", inner_circ_x[i])
        print("inner_circ_y[i]:", inner_circ_y[i])

        X[Index(n_samples_out + i, 0)] = inner_circ_x[i]
        X[Index(n_samples_out + i, 1)] = inner_circ_y[i]

    #cos_inner_circ_x.free()
    #sin_inner_circ_y.free()
    print("passed error")
    print("="*50)

    # Clean up memory
    print("outer_circ_x.free()")
    outer_circ_x.free()
    print("outer_circ_y.free()")
    outer_circ_y.free()
    print("cos_inner_circ_x.free()")
    cos_inner_circ_x.free()
    print("sin_inner_circ_y.free()")
    sin_inner_circ_y.free()
    print("inner_circ_x.free()")
    inner_circ_x.free()
    print("inner_circ_x.free()")
    # This one gives error
    #inner_circ_y.free()
#
    var y = Tensor[DType.float64](n_samples)
    for i in range(n_samples_out):
        y[i] = 0.0
    for i in range(n_samples_in):
        y[n_samples_out + i] = 1.0

    #if shuffle_data:
    #    var indices = Tensor[DType.int64](n_samples)
    #    for i in range(n_samples):
    #        indices[i] = i
    #    #var indices = my_shuffle(range(n_samples))
    #    my_shuffle(indices)
    #    var X_shuffled = Tensor[DType.float64](n_samples, 2)
    #    var y_shuffled = Tensor[DType.float64](n_samples)
    #    for i in range(n_samples):
    #        #X_shuffled[i, 0] = X[indices[i], 0]
    #        X_shuffled[Index(i, 0)] = X[Index(indices[i], 0)]
    #        X_shuffled[Index(i, 1)] = X[Index(indices[i], 1)]
    #        y_shuffled[i] = y[Index(indices[i])]
    #    X = X_shuffled
    #    y = y_shuffled

    if noise > 0.0:
        var noise_tensor = normal(noise, (n_samples, 2))
        var X_with_noise = Tensor[DType.float64](n_samples, 2)
        for i in range(n_samples):
            for j in range(2):
                X_with_noise[Index(i, j)] = X[Index(i, j)] + noise_tensor[Index(i, j)]
        X = X_with_noise

    return (X, y)
