from mojograd.engine import Value 
from mojograd.nn import Neuron, MLP, Layer
from memory import ArcPointer

from testing import assert_almost_equal, assert_true, assert_equal
from python import Python, PythonObject

#fn loss(model: ArcPointer[MLP], X: PythonObject, y: PythonObject, batch_size: PythonObject = None) raises:
fn loss(model: ArcPointer[MLP], X: PythonObject, y: PythonObject, batch_size: PythonObject = PythonObject(None)) raises:
    var Xb : PythonObject
    var yb : PythonObject

    var np = Python.import_module("numpy")

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
    print("Inputs ===============")
    for i in range(Xb.shape[0]):
        print(i)
        var row = List[ArcPointer[Value]]()
        for j in range(Xb.shape[1]):
            print(j)
            row.append(ArcPointer[Value](Value(Float64(Xb[i, j]))))
            print(Float64(Xb[i, j]))
        inputs.append(row)

    #var scores = List[ArcPointer[Value]]()
    # This is supposed to be the forward
    var scores = List[List[ArcPointer[Value]]]()
    print("Scores ===============")
    for input in inputs:
        for i in input[]:
            print(repr(i[][]))
        scores.append(model[](x = input[]))

    var losses = List[ArcPointer[Value]]()
    print("Losses ===============")
    for i in range(len(scores)):
        var yi = Float64(yb[i])
        var scorei = scores[i]
        #losses.append(ArcPointer[Value]((Value(1) + (Value(-1) * Value(yi) * scorei)).relu()))

fn create_model() raises:
    # initialize a model 
    model = MLP(2, List[Int](16, 16, 1)) # 2-layer neural network
    print(repr(model))
    print("number of parameters", len(model.parameters()))

    var sklearn = Python.import_module("sklearn.datasets")

    # Generate the dataset
    var make_moons = sklearn.make_moons
    var result: PythonObject = make_moons(n_samples=100, noise=0.1)
    var X: PythonObject = result[0]
    var y: PythonObject = result[1]

    # Adjust y to be -1 or 1
    y = y * 2 - 1
  
    #try:
    #    loss(ArcPointer[MLP](model), X, y)
    #except e:
    #    print(e)

    var batch_size: PythonObject = None
    var Xb : PythonObject
    var yb : PythonObject

    var np = Python.import_module("numpy")

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
    print("Inputs ===============")
    for i in range(Int(Xb.shape[0])):
    #for i in range(Int(100)):
        print("First For")
        print(Int(Xb.shape[0]))
        print(Xb.shape[0])
        print(i)
        print(Xb[0, 0])
        print("This should break")
        print(Xb[0, 1])
        var row = List[ArcPointer[Value]]()
        for j in range(Int(Xb.shape[1])):
        #for j in range(Int(2)):
            print("Inner For")
            #print(Int(Xb.shape[1]))
            #print(Xb.shape[1])
            print(j)
            var test = Xb.item(i, j)
            print(test)
            #var mifloat :Float64 = test.py_object.

            #row.append(ArcPointer[Value](Value(Float64(Xb[i, j]))))
            #row.append(ArcPointer[Value](Value(Xb[i, j])))
            #row.append(ArcPointer[Value](Value(Float64(test))))

        #inputs.append(row)
    
    print("Passed =========================")

    make_moons = None
    X = None
    y = None
    sklearn = None
    np = None


    #var scores = List[ArcPointer[Value]]()
    # This is supposed to be the forward
    #var scores = List[List[ArcPointer[Value]]]()
    #print("Scores ===============")
    #for input in inputs:
    #    for i in input[]:
    #        print(repr(i[][]))
    #    scores.append(model(x = input[]))

    #var losses = List[ArcPointer[Value]]()
    #print("Losses ===============")
    #for i in range(len(scores)):
    #    var yi = Float64(yb[i])
    #    var scorei = scores[i]


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
        #test1()
        #showmoons()
        #test_neuron()
        #test_layer()
        #test_mlp()
        create_model()
        

    except e:
        print(e)

