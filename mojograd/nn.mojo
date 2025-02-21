""" """

from random import random_float64, rand
from memory import ArcPointer, UnsafePointer
from .engine import Value
from math import sqrt

struct Module:
    fn zero_grad(self):
        for p in self.parameters():
            p[][].grad[] = 0
    
    fn parameters(self) -> List[ArcPointer[Value]]:
        return List[ArcPointer[Value]]() 

#TODO: Implement inheritance
struct Neuron:
    """
    A set of weigths and a bias.
    
    Input
    ----------
    A list of Values.

    Returns
    ----------
    A weigthed Value.

    Parameters
    ----------
    nin:  
      Number of weigths of the neuron or how many values expects to receive.
    nonlin: 
      If non-linearity is applied to the neuron's activation.
    x:
      Input data.
      len(x) should be == len(w). Otherwise would require zero padding.

    Activation
    ----------
    The input data is weigthed (multiplied) against the neuron's and then sumed up (scalar product) to 
    get the neuron's influence on the data.
    len(a) == len(b)
    a . b = sum(a[i]*b[i]) for in range(len(a))

    Examples
    ----------    

    """
    var w : List[Value]
    var b : Value
    var nonlin : Bool

    fn __init__(out self, nin: Int, nonlin: Bool = True):
        self.w = List[Value]()
        
        for _ in range(nin):
            """Values between -1 and 1 for weigthing."""
            var rand = random_float64(-1.0, 1.0)
            """This gives better convergence than these
            bound = Float64(1 / sqrt(nin)); random_float64(-bound, bound)
            rand(foo, nin, min=-bound, max=bound)
            """
            self.w.append(Value(rand))

        self.b = Value(0)
        self.nonlin = nonlin

    #TODO: Check if this ArcPointer function is needed
    fn __call__(self, x : List[Value]) -> Value:
        #TODO: change to 0.0
        var act = Value(0)

        #TODO: Consider SIMD operations
        for i in range(len(self.w)):
            act += (self.w[i] * x[i])
            """Scalar product."""

        act += self.b

        if self.nonlin:
            return act.relu()
        else:
            return act

    fn __call__(self, x : List[Float64]) -> Value:
        var act = Value(0)

        #TODO: Consider SIMD operations
        for i in range(len(self.w)):
            act += (self.w[i] * x[i])
            """Scalar product."""

        act += self.b

        if self.nonlin:
            return act.relu()
        else:
            return act
    
    fn __moveinit__(out self, owned other: Neuron):
        #TODO: Check ^
        self.w = other.w
        self.b = other.b
        self.nonlin = other.nonlin
    
    fn __copyinit__(out self, other: Neuron):
        self.w = other.w
        self.b = other.b
        self.nonlin = other.nonlin
    
    fn parameters(self) -> List[Value]:
        #TODO: Make this more Pythonic
        #return self.w + self.b
        var params = self.w
        params.append(self.b)
        return params
      
    fn __repr__(self) -> String:
        var neuron_type = String("ReLU" if self.nonlin else "Linear")
        return neuron_type + " Neuron( w: " + str(len(self.w)) + ")"
      
    fn __repr__(self, full: Bool = False) -> String:
        var neuron_type = String("ReLU" if self.nonlin else "Linear")

        if full:
            neuron_type += "["
            for i in range(len(self.w)):
                neuron_type += ", " + repr(self.w[i])
            neuron_type += "]"

        return neuron_type + " Neuron(" + str(len(self.w)) + ")"

#TODO: Implement inheritance
struct Layer:
    """
    A fully connected Layer.
    
    Input
    ----------
    A list of Values.

    Returns
    ----------
    A list of weigthed Values.
    Note to self: A Layer has no weigths since is mostly an abstraction to interact with 
    many neurons in a uniform manner, the neuron itself has the weights.

    Parameters
    ----------
    nin:  
      Number of weigths per neuron or how many values will this layer receive.
    nout: 
      Number of neurons per layer or how many values will this layer output.
    kwargs: 
      If relu is applied to the output of every neuron.
    x:
      Input data.

    Activation
    ----------
    The input data is weigthed through all the layer's neurons.

    Examples
    ----------    

    """
    #TODO: Maybe this needs to be a pointer
    var neurons : List[ArcPointer[Neuron]]
    """The layer's neurons."""

    #TODO: Rename this kwargs to activation
    fn __init__(out self, nin: Int, nout: Int, kwargs: Bool = True):
        self.neurons = List[ArcPointer[Neuron]]()
        for _ in range(nout):
            self.neurons.append(Neuron(nin = nin, nonlin = kwargs))
    
    fn __call__(self, x: List[Value]) -> List[Value]:
        var out = List[Value]()
        for i in range(len(self.neurons)):
            out.append(self.neurons[i][](x = x))
        
        #return out[0] if len(out) == 1 else out
        return out
    
    fn parameters(self) -> List[ArcPointer[Value]]:
        var out = List[ArcPointer[Value]]()
        for n in self.neurons:
            for p in n[][].parameters():
                out.append(p[])

        return out

    fn __moveinit__(out self, owned other: Layer):
        #TODO: Validate ^
        self.neurons = other.neurons
    
    fn __repr__(self) -> String:
        var neurons_repr = String("Layer of [" )
        neurons_repr += "Input (weigths) " + str(len(self.neurons[0][].w)) + ' | '
        neurons_repr += ", Output (neurons): " + str(len(self.neurons)) + ' | '
        for i in range(len(self.neurons)):
            neurons_repr += ", " + repr(self.neurons[i][])
        neurons_repr += "]"

        return neurons_repr

#TODO: Implement inheritance
struct MLP:
    """
    Simple MLP model that implements fully connected layers.
    
    Input
    ----------
    A list of Values.

    Returns
    ----------
    A list of scores.

    Parameters
    ----------
    nin:  
      Number of weigths per neuron of the first layer or how many values will be passed 
      initially to the model.
    nouts: 
      Number of neurons per layer or how many values will each layer output.
    kwargs: 
      If relu is applied to the output of every neuron.
    x:
      Input data.

    Activation
    ----------
    The input data is weigthed through all the layer's neurons and the output
    is passed through the next layers.

    Examples
    ----------    
    A model the expect an input of x1, x2.
    Layer 1: Input 2, Output 16
    Layer 2: Input 16, Output 1
    model = MLP(2, List[Int](16, 16, 1)) # 2-layer neural network and 1 layer-output

    """
    #TODO: Maybe this needs to be pointer
    var layers : List[ArcPointer[Layer]]
    fn __init__(out self, nin: Int, nouts: List[Int]):
        #var sz = List[Int](nin) + nouts
        var sz = List[Int]()
        sz.append(nin)
        for n in nouts:
            sz.append(n[])
        self.layers = List[ArcPointer[Layer]]()

        for i in range(len(nouts)):
            self.layers.append(Layer(nin = sz[i], nout = sz[i + 1], kwargs = (i != len(nouts) - 1)))
            """This makes the number of neurons of the current layer equals to the number of weights of each neuron
            of the next layer."""

    fn __call__(self, mut x: List[Value]) -> List[Value]:
        for layer in self.layers:
            x = layer[][](x)
        
        #return x[0] if len(x) == 1 else x
        return x
        """The output can be a List of one value or multiple. This deppends on the last layer output"""

    fn __copyinit__(out self,  other: MLP):
        self.layers = other.layers

    fn __moveinit__(out self, owned other: MLP):
        #TODO: Validate ^
        self.layers = other.layers
    
    fn parameters(self) -> List[ArcPointer[Value]]:
        var out = List[ArcPointer[Value]]()
        for layer in self.layers:
            for p in layer[][].parameters():
                out.append(p[])

        return out
    
    fn __repr__(self) -> String:
        var mlp_repr = String("MLP of [" )
        for i in range(len(self.layers)):
            mlp_repr += ", " + repr(self.layers[i][])
        mlp_repr += "]"

        return mlp_repr

