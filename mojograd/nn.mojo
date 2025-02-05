""" """

from random import random_float64
from memory import ArcPointer
from .engine import Value

struct Module:
    fn zero_grad(self):
        for p in self.parameters():
            p[][].grad[] = 0
    
    fn parameters(self) -> List[ArcPointer[Value]]:
        return List[ArcPointer[Value]]() 


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
    The input data is weigthed against the neuron's weigths plus bias and then sum up to 
    get the neuron's influence on the data.

    Examples
    ----------    

    """
    #TODO: Maybe this needs to be a pointer
    var w : List[ArcPointer[Value]]
    var b : ArcPointer[Value]
    var nonlin : ArcPointer[Bool]

    fn __init__(out self, nin: Int, nonlin: Bool = True):
        self.w = List[ArcPointer[Value]]()
        for _ in range(nin):
            """Values between -1 and 1 for weigthing."""
            var rand = random_float64(-1.0, 1.0)
            self.w.append(ArcPointer[Value](Value(rand)))

        self.b = Value(0)
        self.nonlin = nonlin

    fn __call__(self, x : List[ArcPointer[Value]]) -> Value:
        #var act = Value(data = self.b) #TODO: Evaluate this
        var act = Value(data = self.b[].data[])

        #TODO: Check vector operation
        #print("Neuron class =========")
        #print("len w : " + str(len(self.w)))
        for i in range(len(self.w)):
            #print(str(act.data[]))
            #act.data[] += (self.w[i][].data[] * x[i][].data[])
            act += (self.w[i][] * x[i][]) # weigth inputs and linear combination

        if self.nonlin[]:
            return act.relu()
        else:
            return act
    
    fn __moveinit__(out self, owned other: Neuron):
        self.w = other.w
        self.b = other.b
        self.nonlin = other.nonlin
    
    #TODO: Validate if this works fine or only the data should be coppied
    fn __copyinit__(out self, other: Neuron):
        self.w = other.w
        self.b = other.b
        self.nonlin = other.nonlin
    
    fn parameters(self) -> List[ArcPointer[Value]]:
        #TODO: Check this operation
        #return self.w + self.b
        var params = self.w
        params.append(self.b)
        return params
      
    fn __repr__(self) -> String:
        var neuron_type = String("ReLU" if self.nonlin[] else "Linear")
        return neuron_type + " Neuron( w: " + str(len(self.w)) + ")"
      
    fn __repr__(self, full: Bool = False) -> String:
        var neuron_type = String("ReLU" if self.nonlin[] else "Linear")

        if full:
            neuron_type += "["
            for i in range(len(self.w)):
                neuron_type += ", " + repr(self.w[i][])
            neuron_type += "]"

        return neuron_type + " Neuron(" + str(len(self.w)) + ")"

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

    fn __init__(out self, nin: Int, nout: Int, kwargs: Bool = True):
        self.neurons = List[ArcPointer[Neuron]]()
        for _ in range(nout):
            self.neurons.append(Neuron(nin = nin, nonlin = kwargs))
    
    fn __call__(self, x: List[ArcPointer[Value]]) -> List[ArcPointer[Value]]:
        var out = List[ArcPointer[Value]]()
        for i in range(len(self.neurons)):
            out.append(self.neurons[i][](x = x))
        
        return out
    
    fn parameters(self) -> List[ArcPointer[Value]]:
        var out = List[ArcPointer[Value]]()
        for n in self.neurons:
            for p in n[][].parameters():
                out.append(p[])

        return out

    fn __moveinit__(out self, owned other: Layer):
        self.neurons = other.neurons
    
    fn __repr__(self) -> String:
        var neurons_repr = String("Layer of [" )
        neurons_repr += "Input (weigths) " + str(len(self.neurons[0][].w)) + ' | '
        neurons_repr += ", Output (neurons): " + str(len(self.neurons)) + ' | '
        for i in range(len(self.neurons)):
            neurons_repr += ", " + repr(self.neurons[i][])
        neurons_repr += "]"

        return neurons_repr

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

    fn __call__(self, mut x: List[ArcPointer[Value]]) -> List[ArcPointer[Value]]:
        for layer in self.layers:
            x = layer[][](x)
        
        return x
        """The output can be a List of one value or multiple. This deppends on the last layer output"""

    fn __copyinit__(out self,  other: MLP):
        self.layers = other.layers

    fn __moveinit__(out self, owned other: MLP):
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

