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
    var nonlin : Bool

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

        for i in range(len(self.w)):
            act.data[] += (self.w[i][].data[] * x[i][].data[])

        if self.nonlin:
            return act.relu()
        else:
            return act

