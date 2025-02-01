from mojograd.engine import Value 
from mojograd.nn import Neuron, MLP, Layer
from memory import ArcPointer

from testing import assert_almost_equal, assert_true, assert_equal
from python import Python, PythonObject

#fn loss(model: ArcPointer[MLP], X: PythonObject, y: PythonObject, batch_size: PythonObject = None) raises:
fn loss(model: ArcPointer[MLP], X: PythonObject, y: PythonObject, batch_size: PythonObject = PythonObject(None)) raises -> Tuple[ArcPointer[Value], Float64]:
    var Xb : PythonObject
    var yb : PythonObject

    var np = Python.import_module("numpy")

    #TODO: Validate this logic
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
    """These are the inputs to the model layers"""

    print("Inputs ===============")
    for i in range(Int(Xb.shape[0])):
        var row = List[ArcPointer[Value]]()
        for j in range(Int(Xb.shape[1])):
            #TODO: Find a better way to do this conversion
            var value: Float64 = Xb.item(i, j).to_float64()
            row.append(ArcPointer[Value](Value(value)))

        inputs.append(row)
    
    print("Passed =========================")
    print("len input: ", len(inputs))

    # This is the forward
    var scores = List[ArcPointer[Value]]()
    print("Scores ===============")
    for input in inputs:
        # Added [0] to the end to get the only Value of the list after the activation
        scores.append(model[](x = input[])[0])
        """Here, each list of scores becomes a 1 element list."""

    var losses = List[ArcPointer[Value]]()
    print("Losses ===============")
    for i in range(len(scores)):
        #TODO: Find a better way to do this conversion
        var yi: Float64 = yb.item(i).to_float64()
        var scorei = scores[i]

        # Note how the output of each list after the forward in scores is only one value
        losses.append(ArcPointer[Value]((Value(1) + (Value(-1) * Value(yi) * scorei[])).relu()))
    
    print("After calculating losses")
    print(len(losses))

    var data_loss = ArcPointer[Value](Value(0))
    for loss in losses:
        data_loss[] += loss[][]
    data_loss[] *= Value(1.0 / Float64(len(losses)))

    print("Sum of the data loss")
    print(repr(data_loss[]))

    var alpha = 1e-4
    var reg_loss = ArcPointer[Value](Value(0))
    for p in model[].parameters():
        reg_loss[] += (p[][] * p[][])
    reg_loss[] *= Value(alpha)
    var total_loss = ArcPointer[Value](data_loss[] + reg_loss[])

    print("Total loss")
    print(repr(total_loss[]))

    var accuracy_count: Int = 0
    for i in range(len(scores)):
        #TODO: Find a better way to do this conversion
        var yi = Float64(yb.item(i).to_float64())
        var scorei = scores[i]
        if (yi > 0) == (scorei[].data[] > 0):
            accuracy_count += 1

    var accuracy = Float64(accuracy_count) / Float64(len(scores))
    print("accuracy_count: ", accuracy_count)
    print("len scores: ", len(scores))
    print("accuracy: ", accuracy)

    print("total loss: ", repr(total_loss[]), " | Accuracy: ", str(accuracy))

    np = None

    return (total_loss, accuracy)

fn show_predictions(model: ArcPointer[MLP], X: PythonObject, y: PythonObject) raises:
    var np = Python.import_module("numpy")
    var plt = Python.import_module("matplotlib.pyplot")

    var h: Float64 = 0.25
    var x_min: Float64 = np.min(X.T[0]).to_float64() - 1
    var x_max: Float64 = np.max(X.T[0]).to_float64() + 1
    var y_min: Float64 = np.min(X.T[1]).to_float64() - 1
    var y_max: Float64 = np.max(X.T[1]).to_float64() + 1

    var xx: PythonObject = np.arange(x_min, x_max, h)
    var yy: PythonObject = np.arange(y_min, y_max, h)
    var xx_yy: PythonObject = np.meshgrid(xx, yy)
    var Xmesh: PythonObject = np.c_[xx_yy[0].ravel(), xx_yy[1].ravel()]

    var inputs = List[List[ArcPointer[Value]]]()
    for i in range(Int(Xmesh.shape[0])):
        var row = List[ArcPointer[Value]]()
        for j in range(Int(Xmesh.shape[1])):
            var value = Float64(Xmesh[i, j].to_float64())
            row.append(ArcPointer[Value](Value(value)))
        inputs.append(row)

    var scores = List[ArcPointer[Value]]()
    for input in inputs:
        scores.append(model[](x = input[])[0])

    var Z = List[Bool]()
    for score in scores:
        Z.append(score[][].data[] > 0)

    var Z_np = np.zeros(len(Z), dtype=np.bool_)
    for i in range(len(Z)):
        Z_np[i] = Z[i]

    #var Z_np: PythonObject = np.array(Z)
    var Z_reshaped: PythonObject = Z_np.reshape(xx_yy[0].shape)

    plt.figure()
    plt.contourf(xx_yy[0], xx_yy[1], Z_reshaped, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(X.T[0], X.T[1], c=y, s=40, cmap=plt.cm.Spectral)
    plt.xlim(np.min(xx), np.max(xx))
    plt.ylim(np.min(yy), np.max(yy))
    plt.show()

fn create_model() raises:
    # initialize a model 
    model = MLP(2, List[Int](16, 16, 1)) # 2-layer neural network
    print(repr(model))
    print("number of parameters", len(model.parameters()))

    var sklearn = Python.import_module("sklearn.datasets")
    var np = Python.import_module("numpy")
    var plt = Python.import_module("matplotlib.pyplot")

    # Generate the dataset
    var make_moons = sklearn.make_moons
    var result: PythonObject = make_moons(n_samples=100, noise=0.1)
    var X: PythonObject = result[0]
    var y: PythonObject = result[1]

    # Adjust y to be -1 or 1
    y = y * 2 - 1

    #var model_ptr = ArcPointer[MLP](model)
  
    #for k in range(100):
    for k in range(2):
        try:
            var total_loss: ArcPointer[Value]
            var acc: Float64
            # forward
            (total_loss, acc) = loss(model, X, y, PythonObject(None))

            # backward
            #TODO: Implement this with trait 
            for out in model.parameters():
                out[][].grad[] = 0
                """The use of the grad is to update the Neuron data so that the next 
                forward pass gives better results. Which makes the model 'learn'."""

            total_loss[].backward()

            # update (sgd)
            var learning_rate : Float64 = 1.0 - 0.9 * k/100 
            for p in model.parameters():
                print(repr(p[][]))
                p[][].data[] -= learning_rate * p[][].grad[]
                print(repr(p[][]))
                """Note how the grad is used to update the same Value data and the 
                learning_rate damps the influence as the iterations increases"""
            
            #if k % 1 == 0:
            print("Step: ", k, " | loss data: ", total_loss[].data[], " | loss grad: ", total_loss[].grad[] , " | accuracy: ", acc*100)

        except e:
            print(e)

    show_predictions(model, X, y)
    
    make_moons = PythonObject(None)
    X = PythonObject(None)
    y = PythonObject(None)
    sklearn = PythonObject(None)
    np = PythonObject(None)
    plt = PythonObject(None)



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

