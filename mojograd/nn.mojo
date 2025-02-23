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
    A list of linear concepts (Values).

    Returns
    ----------
    A transformed linear concept (Value).

    """
    var w : List[Value]
    """Linear representation of a concept."""
    var b : Value
    """Neuron's 'happyness'."""
    var nonlin : Bool
    """Non linear transformation."""

    fn __init__(out self, nin: Int, nonlin: Bool = True):
        self.w = List[Value]()
        
        for _ in range(nin): self.w.append(Value(random_float64(-1.0, 1.0)))
        """Values between -1 and 1 for weigthing."""
            
        self.b = Value(0)
        self.nonlin = nonlin

    fn __call__(self, x : List[Value]) -> Value:
        """Relation making operation: dot product between linear representations of concepts."""
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
        self.w = other.w^
        self.b = other.b^
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
    nonlin: 
      If relu is applied to the output of every neuron.
    x:
      Input data.

    Activation
    ----------
    The input data is weigthed through all the layer's neurons.

    Examples
    ----------    

    """
    var neurons : List[Neuron]
    """The layer's neurons."""

    fn __init__(out self, nin: Int, nout: Int, nonlin: Bool = True):
        self.neurons = List[Neuron]()
        for _ in range(nout):
            self.neurons.append(Neuron(nin = nin, nonlin = nonlin))
    
    fn __call__(self, x: List[Value]) -> List[Value]:
        var out = List[Value]()
        for i in range(len(self.neurons)):
            out.append(self.neurons[i](x = x))
        
        return out
    
    fn parameters(self) -> List[Value]:
        var out = List[Value]()
        for n in self.neurons:
            for p in n[].parameters():
                out.append(p[])

        return out

    fn __moveinit__(out self, owned other: Layer):
        self.neurons = other.neurons^
    
    fn __copyinit__(out self, other: Layer):
        self.neurons = other.neurons
    
    fn __repr__(self) -> String:
        var neurons_repr = String("Layer of [" )
        neurons_repr += "Input (weigths) " + str(len(self.neurons[0].w)) + ' | '
        neurons_repr += ", Output (neurons): " + str(len(self.neurons)) + ' | '
        for i in range(len(self.neurons)):
            neurons_repr += ", " + repr(self.neurons[i])
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
    nonlin: 
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
    var layers : List[Layer]
    fn __init__(out self, nin: Int, nouts: List[Int]):
        var sz = List[Int](nin) + nouts
        self.layers = List[Layer]()

        for i in range(len(nouts)):
            self.layers.append(Layer(nin = sz[i], nout = sz[i + 1], nonlin = (i < len(nouts) - 1)))
            """This makes the number of neurons of the current layer equals to the number of weights of each neuron
            of the next layer."""

    fn __call__(self, mut x: List[Value]) -> List[Value]:
        for layer in self.layers:
            x = layer[](x)
        
        return x
        """The output can be a List of one value or multiple. This deppends on the last layer output"""

    fn __copyinit__(out self,  other: MLP):
        self.layers = other.layers

    fn __moveinit__(out self, owned other: MLP):
        self.layers = other.layers^
    
    fn parameters(self) -> List[Value]:
        var out = List[Value]()
        for layer in self.layers:
            for p in layer[].parameters():
                out.append(p[])

        return out
    
    fn __repr__(self) -> String:
        var mlp_repr = String("MLP of [\n" )
        for i in range(len(self.layers)):
            mlp_repr += repr(self.layers[i]) + ",\n"
        mlp_repr += "]"

        return mlp_repr

