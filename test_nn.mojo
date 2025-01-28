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

        fn loss(batch_size: PythonObject = None) raises:
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
                scores.append(model(x = input[]))

            var losses = List[ArcPointer[Value]]()
            print("Losses ===============")
            for i in range(len(scores)):
                var yi = Float64(yb[i])
                var scorei = scores[i]
                #losses.append(ArcPointer[Value]((Value(1) + (Value(-1) * Value(yi) * scorei)).relu()))

        #TODO: This needs to be fixed
        loss()

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


    try:
        #test1()
        #showmoons()
        #create_model()
        test_neuron()
        

    except e:
        print(e)
