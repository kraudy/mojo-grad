from mojograd.engine import Value 
from mojograd.nn import Neuron, MLP
from memory import ArcPointer

from testing import assert_almost_equal, assert_true, assert_equal
from python import Python, PythonObject

fn main(): 
    fn test1() raises:
        """ Testing nn """
        var n = Neuron(2)
        #var x = List[Value](Value(1.0), Value(-2.0))
        #TODO: Check if this Arcpointer is neede
        var x = List[ArcPointer[Value]](ArcPointer[Value](Value(1.0)), ArcPointer[Value](Value(-2.0)))
        var y = n(x)
        y.backward()
        print("Neuron test")
        print("y")
        print(repr(y))
        print("Neuron")
        print(repr(n))
        for a in n.parameters():
            print(repr(a[][]))
    
    fn showmoons() raises:
        var sklearn = Python.import_module("sklearn.datasets")
        var plt = Python.import_module("matplotlib.pyplot")

        # Generate the dataset
        var make_moons = sklearn.make_moons
        var result: PythonObject = make_moons(n_samples=100, noise=0.1)
        var X: PythonObject = result[0]
        var y: PythonObject = result[1]

        # Adjust y to be -1 or 1
        y = y * 2 - 1

        # Create the plot
        plt.figure(figsize=(5,5))
        plt.scatter(X.T[0], X.T[1], c=y, s=20, cmap='jet')
        plt.show()
    
    fn create_model() raises:
        # initialize a model 
        model = MLP(2, List[Int](16, 16, 1)) # 2-layer neural network
        print(repr(model))
        print("number of parameters", len(model.parameters()))

        var sklearn = Python.import_module("sklearn.datasets")
        var np = Python.import_module("numpy")

        # Generate the dataset
        var make_moons = sklearn.make_moons
        var result: PythonObject = make_moons(n_samples=100, noise=0.1)
        var X: PythonObject = result[0]
        var y: PythonObject = result[1]

        # Adjust y to be -1 or 1
        y = y * 2 - 1

        fn loss(batch_size: PythonObject) raises:
            var Xb : PythonObject
            var yb : PythonObject

            if batch_size is None:
                Xb = X
                yb = y
            else:
                var total_samples = Float64(X.shape[0])
                var batch_size_int = batch_size.to_int64()
                var indices = np.random.choice(total_samples, batch_size_int, replace=False)
                Xb = np.take(X, indices, axis=0)
                yb = np.take(y, indices, axis=0)
            
            var inputs = List[List[ArcPointer[Value]]]()
            for i in range(Xb.shape[0]):
                var row = List[ArcPointer[Value]]()
                for j in range(Xb.shape[1]):
                    row.append(ArcPointer[Value](Value(Float64(Xb[i, j]))))
                inputs.append(row)

            #var scores = List[ArcPointer[Value]]()
            # This is supposed to be the forward
            var scores = List[List[ArcPointer[Value]]]()
            for input in inputs:
                scores.append(model(x = input[]))
    

    try:
        test1()
        #showmoons()
        create_model()
        

    except e:
        print(e)
