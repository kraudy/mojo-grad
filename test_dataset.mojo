from datasets.samples_generator import make_moons
from tensor import Tensor
from python import Python, PythonObject
from testing import assert_almost_equal

fn show_moons() raises:
    var X : Tensor[DType.float64] 
    var y : Tensor[DType.float64] 
    (X, y) = make_moons(100, True, 0.25, 42)
    print("X shape:", X.shape())
    print("y shape:", y.shape())
    for i in range(20):
        print("Sample", i, "- X:", X[i, 0], X[i, 1], "y:", y[i])

    var plt = Python.import_module("matplotlib.pyplot")
    var np = Python.import_module("numpy")

    # Convert Mojo tensors to NumPy arrays
    var X_np = np.zeros((X.dim(0), X.dim(1)))
    var y_np = np.zeros(y.dim(0))

    for i in range(X.dim(0)):
        for j in range(X.dim(1)):
            X_np[i, j] = X[i, j]
        y_np[i] = y[i]

    # Adjust y to be -1 or 1
    y_np = y_np * 2 - 1

    # Create the plot
    plt.figure(figsize=(5,5))
    plt.scatter(X_np.T[0], X_np.T[1], c=y_np, s=20, cmap='jet')
    plt.show()

fn test_make_moons() raises:
    var X: Tensor[DType.float64]
    var y: Tensor[DType.float64]
    #TODO: 3 gives nan error for some reason
    (X, y) = make_moons(4, False, 0.0, 42)  # shuffle=False, noise=0.0, random_seed=0

    for i in range(4):
        var center_x: Float64 = 0.0
        var center_y: Float64 = 0.0
        print(y[i])
        if y[i] == 1.0:
            center_x = 1.0
            center_y = 0.5
        
        print(X[i, 0])
        var dist_sqr = pow(X[i, 0] - center_x, 2) + pow(X[i, 1] - center_y, 2)
        assert_almost_equal(dist_sqr, 1.0, atol=1e-6, msg="Point is not on expected unit circle")
        #assert_almost_equal(dist_sqr, 1.0, 1e-6, "Point is not on expected unit circle")


fn main():
    try:
        #show_moons()
        test_make_moons()
    except e:
        print(e)
    