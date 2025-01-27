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

    fn __init__(out self, nin: Int, nonlin: Bool = True):
        self.w = List[ArcPointer[Value]]()
        for i in range(nin):
            var rand = random_float64(-1.0, 1.0)
            # Need to fix Value Float32 to Float64
            self.w.append(ArcPointer[Value](Value(rand)))
        self.b = Value(0)
