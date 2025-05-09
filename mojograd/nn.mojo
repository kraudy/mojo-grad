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
    A column or a row(when transpossed) of the matrix that represents the Layer.
    Transforms the linear representation of concept 'a' into concept 'c'
    ((a * w).sum + bias).activation() = c.
    
    Input
    ----------
    A linear concept representation(Values).

    Returns
    ----------
    A transformed linear concept representation(Value).

    """
    var w : List[Value]
    """Distributed representation of concepts."""
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
        """Relation making operation: dot product between the linear representation of a concept and
        the ditributed representation learned by the Neuron."""
        var act = Value(0)
        #TODO: Consider SIMD operations
        for i in range(len(self.w)): act += (self.w[i] * x[i])
        """Scalar product."""

        if self.nonlin: return (act + self.b).relu()
        return (act + self.b)

    fn __call__(self, x : List[Float64]) -> Value:
        var act = Value(0)
        #TODO: Consider SIMD operations
        for i in range(len(self.w)): act += (self.w[i] * x[i])
        """Scalar product."""

        if self.nonlin: return (act + self.b).relu()
        return (act + self.b)
    
    fn __moveinit__(out self, owned other: Neuron):
        self.w = other.w^
        self.b = other.b^
        self.nonlin = other.nonlin
    
    fn __copyinit__(out self, other: Neuron):
        self.w = other.w
        self.b = other.b
        self.nonlin = other.nonlin
    
    fn parameters(self) -> List[Value]:
        return self.w + List[Value](self.b)
      
    fn __repr__(self) -> String:
        return String("ReLU" if self.nonlin else "Linear") + " Neuron( w: " + str(len(self.w)) + ")"
      
    fn __repr__(self, full: Bool = False) -> String:
        var neuron_type = String("ReLU" if self.nonlin else "Linear")

        if full:
            neuron_type += "["
            for i in range(len(self.w)): neuron_type += ", " + repr(self.w[i])
            neuron_type += "]"

        return neuron_type + " Neuron(" + str(len(self.w)) + ")"

struct Linear:
  var weight : List[Neuron]
  var bias   : Optional[Neuron] 

  fn __init__(out self, in_features: Int, out_features: Int, bias: Bool=True):
    var bound = 1 / sqrt(in_features)
    self.weight = List[Neuron]()
    for _ in range(out_features): self.weight.append(Neuron(nin = in_features, nonlin = bias))

    self.bias = Optional[Neuron](Neuron(nin=in_features, nonlin=bias)) if bias else None
    #if bias:
    #  self.bias = Optional[Neuron](Neuron(nin=in_features, nonlin=bias))
    #else:
    #  self.bias = Optional[Neuron](None)
    

#TODO: Implement inheritance
struct Layer:
    """
    A Layer is basically a matrix of size (n,m) where n is the size of the input vector and m the number of neurons
    of the Layer.
    This matrix can also be viewed as a representing a linear transformation. The bias adds a translation and the activation
    affect the output vector space.

    Transforms a set of linear representation of concepts into a new vector space of related concepts.
    ((a @ W) + bias).activation()
    
    Input
    ----------
    A vector space of linear concepts representations(Values).

    Returns
    ----------
    A transformed new vector space of related linear concepts representations(Value). 
    """
    var neurons : List[Neuron]
    """The layer's neurons."""

    fn __init__(out self, nin: Int, nout: Int, nonlin: Bool = True):
        self.neurons = List[Neuron]()
        for _ in range(nout): self.neurons.append(Neuron(nin = nin, nonlin = nonlin))
    
    fn __call__(self, x: List[Value]) -> List[Value]:
        """Transforms a vector space of linear representations into a new vector space of related 
        linear representations."""
        var out = List[Value]()
        for i in range(len(self.neurons)): out.append(self.neurons[i](x = x))
        return out
    
    fn parameters(self) -> List[Value]:
        var out = List[Value]()
        for n in self.neurons:
            for p in n[].parameters(): out.append(p[])
        return out

    fn zero_grad(self):
        for p in self.parameters():
          p[].grad[] = 0

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
    
    Once the model has 'learn', the network becomes a distributed representation of concepts with
    their relations or a probability distribution. 
    
    Input
    ----------
    A vector space of linear concepts representations(Values).

    Returns
    ----------
    A transformed new vector space of related linear concepts representations(Value). 
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

    fn zero_grad(self):
        for p in self.parameters():
          p[].grad[] = 0 

    fn __repr__(self) -> String:
        var mlp_repr = String("MLP of [\n" )
        for i in range(len(self.layers)):
            mlp_repr += repr(self.layers[i]) + ",\n"
        mlp_repr += "]"

        return mlp_repr

