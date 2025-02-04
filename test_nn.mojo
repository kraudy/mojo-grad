from mojograd.engine import Value 
from mojograd.nn import Neuron, MLP, Layer
from memory import ArcPointer
from random import random_float64

from testing import assert_almost_equal, assert_true, assert_equal
from python import Python, PythonObject

fn main(): 
    fn test1() raises:
        """ Creates a Neuron with 10 weights that should received array of 10 inputs"""
        var n = Neuron(10)
        #var x = List[Value](Value(1.0), Value(-2.0))
        #TODO: Check if this Arcpointer is neede
        var x = List[ArcPointer[Value]]()
        #var x = List[ArcPointer[Value]](ArcPointer[Value](Value(1.0)), ArcPointer[Value](Value(-2.0)))
        for _ in range(10):
            var rand = random_float64(-1.0, 1.0)
            x.append(ArcPointer[Value](Value(rand)))
        var y = n(x)
        y.backward()
        print("Neuron test")
        print("y")
        print(repr(y))
        print("Neuron")
        print(repr(n))
        for a in n.parameters():
            print(repr(a[][]))

    fn test2() raises:
        var layers = List[List[Neuron]]()
        for i in range(4):
            var nlist = List[Neuron]()
            for j in range(2 ** i):
                print(j)
                nlist.append(Neuron(nin = (2 ** (i + 1))))
                # The first layer should receive 16 inputs
            print(i)
            layers.append(nlist)
        
        print("Validating created layers")
        print(len(layers))
        for n in layers:
            print(len(n[]))
        
        # Puting the layers from first to last
        var rev_layers = reversed(layers)
        # Generating the input
        var x = List[ArcPointer[Value]]()
        for _ in range(2 ** 4):
            var rand = random_float64(-1.0, 1.0)
            x.append(Value(rand))
        
        # Manually doing the forward
        #var val = List[Value]()
        for l in rev_layers:
            var layer_out = List[ArcPointer[Value]]()
            for n in l[]:
                layer_out.append(n[](x))
            x = layer_out
        
        # x should have the last output and be 1 value
        print("afeter forward")
        print(len(x))
        print(repr(x[0][]))
        x[0][].backward()
        print("afeter backward")
        print(repr(x[0][]))

        print("Neuron repr")
        for l in rev_layers:
            for n in l[]:
                print(repr(n[].__repr__(True)))
    
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
        #test1()
        #test2()
        showmoons()
        #test_neuron()
        #test_layer()
        #test_mlp()
    except e:
        print(e)

