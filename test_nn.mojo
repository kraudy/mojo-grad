from mojograd.engine import Value 
from mojograd.nn import Neuron, MLP, Layer
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

    fn test_neuron() raises:
        # Create Neuron with 2 rand weigths Value and 0 bias Value
        var neuron = Neuron(2)    
        # This is the input to the neuron
        var x = List[ArcPointer[Value]](Value(2), Value(3))

        #TODO: Note how X and W must have the same length. Maybe this can be changed
        var act = neuron(x)
        # Back prop un output Value of the Neuron
        print("Before Backward ==============")
        act.backward()
        print("After Backward ==============")
        print(repr(act))
        print(repr(act._prev[0][]))
        print(repr(x[0][]))
        print("Neuron")
        print(repr(neuron.__repr__(True)))

    fn test_layer() raises:
        # 2 Value inputs
        var x = List[ArcPointer[Value]](Value(2), Value(3))
        # Layer of 1 Neuron with 2 weigts
        var l = Layer(2, 1)
        # Forward
        var res = l(x)
        print("Before backward")
        print(repr(res[0][]))
        res[0][].backward()
        print("After backward")
        print(repr(res[0][]))
        for v in x:
            print("Input grad")
            print(repr(v[][]))
        for v in l.neurons:
            print("Layer neurons")
            print(v[][].__repr__(True))
        print("Layer")
        print(repr(l))

    fn test_mlp() raises:
        #var nouts = List[Int](16, 16, 1)
        var nouts = List[Int](4, 4, 1)
        var x = List[ArcPointer[Value]](Value(2), Value(3))
        var m = MLP(2, nouts)
        var res = m(x)

        print("Before backward =======================")
        print(repr(res[0][]))
        res[0][].backward()
        print("After backward =======================")
        print(repr(res[0][]))
        for v in x:
            print("Input grad")
            print(repr(v[][]))
        for v in m.layers:
            print("MLP Layers")
            print(repr(v[][]))
        print("MLP")
        print(repr(m))

    

    try:
        test1()
        #showmoons()
        test_neuron()
        test_layer()
        test_mlp()
    except e:
        print(e)

