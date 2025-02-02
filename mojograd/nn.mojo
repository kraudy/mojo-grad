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
    """The input a Neuron receives comes from the layer's activation. The input list
    gets progagated through the layer's Neurons and each one outputs a Value. So, the 
    output of a layer is a list of the neurons outpus."""
    #TODO: Maybe this needs to be a pointer
    var w : List[ArcPointer[Value]]
    """The weigths are used to assign importance to the inputs. That is why they are between 0 and 1 and
    that is also why they are called weights, because they 'weight' the inputs."""
    var b : ArcPointer[Value]
    var nonlin : ArcPointer[Bool]
    """W and b form the function that makes the model 'learn'."""

    fn __init__(out self, nin: Int, nonlin: Bool = True):
        self.w = List[ArcPointer[Value]]()
        for _ in range(nin):
            """Note how we are assigning values between -1 and 1.
            With Relu: 0 (Non probability), 1 (Certain)."""
            var rand = random_float64(-1.0, 1.0)
            self.w.append(ArcPointer[Value](Value(rand)))

        self.b = Value(0)
        self.nonlin = nonlin

    fn __call__(self, x : List[ArcPointer[Value]]) -> Value:
        """This is basically a relation making operation.
        len(x) should be >= len(w). Otherwise makes no sense."""
        var act = Value(data = self.b[].data[])

        #TODO: Check vector operation
        #print("Neuron class =========")
        #print("len w : " + str(len(self.w)))
        # W should have the same length as X
        for i in range(len(self.w)):
            """On each layer activation, the neuron takes the layer's input, multiplies it
            by it's weigth and sum it."""
            #print(str(act.data[]))
            #act.data[] += (self.w[i][].data[] * x[i][].data[])
            act += (self.w[i][] * x[i][]) # weigth inputs and linear combination
            """This sum represent all input's collective influence on the neuron."""

        if self.nonlin[]:
            """Activation function. Used to learn more complex patterns by including non-linearity"""
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
        # We can add Value repr fo w if the full detail is wanted
        return neuron_type + " Neuron(" + str(len(self.w)) + ")"
      
    fn __repr__(self, full: Bool = False) -> String:
        var neuron_type = String("ReLU" if self.nonlin[] else "Linear")

        if full:
            neuron_type += "["
            for i in range(len(self.w)):
                neuron_type += ", " + repr(self.w[i][])
            neuron_type += "]"

        # We can add Value repr fo w if the full detail is wanted
        return neuron_type + " Neuron(" + str(len(self.w)) + ")"

struct Layer:
    """A Layer receives a list of inputs and applies them to all the Neurons where each
    one returns a Value. The Layer itself returns a list of Value.
    Note to self: A Layer has no weigths, the Neuron itself has the weights."""
    #TODO: Maybe this needs to be a pointer
    var neurons : List[ArcPointer[Neuron]]
    """A layer is mostly an abstraction to interact with many neurons in a uniform manner."""

    fn __init__(out self, nin: Int, nout: Int, kwargs: Bool = True):
        """These Layers are assumed to be fully connected"""
        # nin:  How many values will this layer receive
        # nout: How many values will this layer output
        # kwargs: If relu is applied to the output of every neuron

        self.neurons = List[ArcPointer[Neuron]]()
        for _ in range(nout):
            self.neurons.append(Neuron(nin = nin, nonlin = kwargs))
    
    fn __call__(self, x: List[ArcPointer[Value]]) -> List[ArcPointer[Value]]:
        """When the layer is called, it activates all the layer's Neurons with the 
        input data."""
        var out = List[ArcPointer[Value]]()
        for i in range(len(self.neurons)):
            out.append(self.neurons[i][](x = x))
        
        return out
        """We pass Values between layer, not Neurons. The Neuron is needed to 
        transform the input into the output in the forward pass and update grads
        in the backprop."""
    
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
        for i in range(len(self.neurons)):
            neurons_repr += ", " + repr(self.neurons[i][])
        neurons_repr += "]"

        return neurons_repr

struct MLP:
    #TODO: Maybe this needs to be pointer
    var layers : List[ArcPointer[Layer]]
    fn __init__(out self, nin: Int, nouts: List[Int]):
        var sz = List[Int](nin) + nouts
        self.layers = List[ArcPointer[Layer]]()

        for i in range(len(nouts)):
            self.layers.append(Layer(nin = sz[i], nout = sz[i + 1], kwargs = (i != len(nouts) - 1)))

    fn __call__(self, mut x: List[ArcPointer[Value]]) -> List[ArcPointer[Value]]:
        for layer in self.layers:
            """Here we activate the layer and assign the output to the input of the next."""
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

