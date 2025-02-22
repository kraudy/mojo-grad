from mojograd.engine import Value 
from mojograd.nn import Neuron, MLP, Layer
from memory import ArcPointer

from testing import assert_almost_equal, assert_true, assert_equal
from python import Python, PythonObject

fn show_predictions(model: MLP, X: PythonObject, y: PythonObject) raises:
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

    var inputs = List[List[Value]]()
    for i in range(Int(Xmesh.shape[0])):
        var row = List[Value]()
        for j in range(Int(Xmesh.shape[1])):
            var value = Float64(Xmesh[i, j].to_float64())
            row.append(Value(value))
        inputs.append(row)

    var scores = List[Value]()
    for input in inputs:
        scores.append(model(x = input[])[0])

    var Z = List[Bool]()
    for score in scores:
        Z.append(score[].data[] > 0)

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

#TODO: Add wich data set to use
#TODO: Move this to the MLP class, or maybe a DataLoader
fn make_inputs(Xb: PythonObject) raises -> List[List[Value]]:
    """Generate inputs to the model layers.
    This should take into account the expected input by the model."""

    var inputs = List[List[Value]]()
    for i in range(Int(Xb.shape[0])):
        var row = List[Value]()
        for j in range(Int(Xb.shape[1])):
            #TODO: Find a better way to do this conversion
            var value: Float64 = Xb.item(i, j).to_float64()
            row.append(Value(value))
            """[x0, x1]"""
        inputs.append(row)
    return inputs

#TODO: Move this to the MLP class.
fn make_forward(model: MLP, mut inputs: List[List[Value]]) raises -> List[Value]:
    """Weigth the input against each layer."""
    var scores = List[Value]()
    for input in inputs:
        scores.append(model(x = input[])[0])
        """Here, each output of the model is a 1 element list since the last layer activation is 1 neuron."""
    return scores

fn calculate_losses(model: MLP, scores: List[Value], yb:  PythonObject) raises -> Tuple[Value, Float64]:
    """Validate the weighted output against the expected output."""
    #svm "max-margin" loss
    var data_loss = Value(0)
    var accuracy = 0.0
    for i in range(len(scores)):
        """This is the loss calculation"""
        #TODO: Find a better way to do this conversion
        var yi: Float64 = yb.item(i).to_float64()
        #TODO: Consider using log for classification
        data_loss += (1 - yi * scores[i]).relu()
        """We want to check if the trulabel * prediction is less than 1"""
        if (yi > 0) == (scores[i].data[] > 0):
            accuracy += 1

    data_loss *= (1.0 / Float64(len(scores)))
    """Here we take the mean of the data loss across the sample"""
    accuracy = (accuracy / Float64(len(scores)))

    var alpha = 1e-4
    var reg_loss = 0.0
    for p in model.parameters():
        reg_loss += (p[].data[] ** 2)
    reg_loss *= alpha
    """L2 regularizaiton to prevent overfit"""

    return (data_loss + reg_loss, accuracy)   

fn loss(model: MLP, X: PythonObject, y: PythonObject, batch_size: PythonObject = PythonObject(None)) raises -> Tuple[Value, Float64]:
    var Xb : PythonObject
    var yb : PythonObject

    var np = Python.import_module("numpy")

    if batch_size is None:
        Xb = X
        yb = y
    else:
        var total_samples = Float64(X.shape[0])
        var batch_size_int = batch_size
        var indices = np.random.choice(total_samples, batch_size_int, replace=False)
        Xb = np.take(X, indices, axis=0)
        yb = np.take(y, indices, axis=0)
    
    np = None

    var inputs = make_inputs(Xb)
    """These are the inputs to the model layers"""

    var scores = make_forward(model, inputs)
    """These are the 'outputs' of the model"""

    return calculate_losses(model, scores, yb)

fn create_mlp_model() raises:
    # initialize a model 
    model = MLP(2, List[Int](16, 16, 1)) # 2-layer neural network with 1 output
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
  
    #for k in range(100):
    var i = 20
    for k in range(i):
        try:
            var total_loss: Value
            var acc: Float64
            print("Forward pass")
            (total_loss, acc) = loss(model, X, y, PythonObject(None))

            print("Zeroing grads")
            #TODO: Implement this with trait 
            for out in model.parameters():
                out[].grad[] = 0
                """Needs to be reset because the grads are added. Not zeroing
                grads is one of the most common mistakes."""

            print("Doing backward")
            total_loss.backward()

            # update (sgd)
            print("Updating weigths")
            var learning_rate : Float64 = 1.0 - 0.9 * k/i 
            for p in model.parameters():
                p[].data[] -= learning_rate * p[].grad[]
                """This is what really makes the model 'learn'
                If the grad is positive, the neuron increaces the loss, hence we reduce it: - * + = -.
                If the grad is negative, the neuron decreses the loss (what we want) hence increce it: - * - = +"""
            
            #if k % 1 == 0:
            print("Step: ", k, " | loss data: ", total_loss.data[], " | loss grad: ", total_loss.grad[] , " | accuracy: ", acc*100)
            print("="*100)

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
    try:
        create_mlp_model()
    except e:
        print(e)

