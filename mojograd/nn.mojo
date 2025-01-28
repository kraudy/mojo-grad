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
    var w : List[ArcPointer[Value]]
    var b : ArcPointer[Value]
    var nonlin : ArcPointer[Bool]

    fn __init__(out self, nin: Int, nonlin: Bool = True):
        # I think w and b should have the same length
        self.w = List[ArcPointer[Value]]()
        for _ in range(nin):
            var rand = random_float64(-1.0, 1.0)
            self.w.append(ArcPointer[Value](Value(rand)))

        self.b = Value(0)
        self.nonlin = nonlin

    fn __call__(self, x : List[ArcPointer[Value]]) -> Value:
        var act = Value(data = self.b[].data[])

        #TODO: Check vector operation
        print("Neuron class =========")
        print("len w : " + str(len(self.w)))
        for i in range(len(self.w)):
            print(str(act.data[]))
            act.data[] += (self.w[i][].data[] * x[i][].data[])

        if self.nonlin[]:
            return act.relu()
        else:
            return act
    
    fn __moveinit__(out self, owned other: Neuron):
        self.w = other.w
        self.b = other.b
        self.nonlin = other.nonlin
    
    fn parameters(self) -> List[ArcPointer[Value]]:
        #TODO: Check this operation
        return self.w + self.b
      
    fn __repr__(self) -> String:
        var neuron_type = String("ReLU" if self.nonlin[] else "Linear")
        # We can add Value repr fo w if the full detail is wanted
        return neuron_type + " Neuron(" + str(len(self.w)) + ")"

struct Layer:
    var neurons : List[ArcPointer[Neuron]]

    fn __init__(out self, nin: Int, nout: Int, kwargs: Bool):
        self.neurons = List[ArcPointer[Neuron]]()
        for _ in range(nout):
            self.neurons.append(Neuron(nin = nin, nonlin = kwargs))
    
    #TODO: Validate where we need a List of Value and a List of Neurons
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
        for i in range(len(self.neurons)):
            neurons_repr += ", " + repr(self.neurons[i][])
        neurons_repr += "]"

        return neurons_repr

struct MLP:
    var layers : List[ArcPointer[Layer]]
    fn __init__(out self, nin: Int, nouts: List[Int]):
        var sz = List[Int](nin) + nouts
        self.layers = List[ArcPointer[Layer]]()

        for i in range(len(nouts)):
            self.layers.append(Layer(nin = sz[i], nout = sz[i + 1], kwargs = (i != len(nouts) - 1)))

    fn __call__(self, mut x: List[ArcPointer[Value]]) -> List[ArcPointer[Value]]:
        for layer in self.layers:
            x = layer[][](x)
        
        return x

    fn __copyinit__(out self,  other: MLP):
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

