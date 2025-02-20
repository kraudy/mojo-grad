
""" 
"""

from collections import Optional, List, Dict, InlineList, Set
from memory import UnsafePointer, memset_zero, ArcPointer, pointer, Pointer

struct Value():
    #TODO: Validate if this ArcPointer is needed
    var data: ArcPointer[Float64]
    var grad :  ArcPointer[Float64]

    #TODO: Check if this scope function can be now implemented
    var _func  : UnsafePointer[fn() escaping -> None, alignment=1]
    # Validate UnsafePointer[Tuple[UnsafePointer[Value], UnsafePointer[Value]]]
    # var _prev :  Set[Value]
    #TODO: There has to be a better way to do this.
    var _prev : List[ArcPointer[Value]]

    var _op : String

    @always_inline
    fn __init__(inout self, data: Float64):
        self.data = data
        self.grad =  0.0

        self._func  = UnsafePointer[fn() escaping -> None, alignment=1]() 
        self._prev = List[ArcPointer[Value]]()

        self._op = String('') 

    fn __init__(inout self, data: Float64, prev1: Value, op: String):
        self.data = data
        self.grad = 0.0

        self._func  = UnsafePointer[fn() escaping -> None, alignment=1]() 

        self._prev = List[ArcPointer[Value]]()
        self._prev.append(ArcPointer[Value](prev1))

        self._op = op

    fn __init__(inout self, data: Float64, prev1: Value, prev2: Value, op: String):
        self.data = data
        self.grad = 0.0

        self._func  = UnsafePointer[fn() escaping -> None, alignment=1]() 

        self._prev = List[ArcPointer[Value]]()
        self._prev.append(ArcPointer[Value](prev1))
        self._prev.append(ArcPointer[Value](prev2))

        self._op = op

    fn __moveinit__(out self, owned existing: Self):
        self.data = existing.data^
        self.grad = existing.grad^
        # Validate
        self._func = existing._func#^
        self._prev = existing._prev^
        self._op = existing._op^
    
    fn __copyinit__(out self, existing: Self):
        self.data = existing.data
        self.grad = existing.grad
        self._func = existing._func
        self._prev = existing._prev
        self._op = existing._op
      
    @always_inline
    fn backward_add(mut self):
        self._prev[0][].grad[] += self.grad[]
        self._prev[1][].grad[] += self.grad[]

    @always_inline
    fn __add__(self, other: Value) -> Value:
        var out = Value(data = (self.data[] + other.data[]), prev1 = self, prev2 = other, op = '+')
        return out

    fn __add__(self, other: Float64) -> Value:
        var v = Value(other)
        return self + v
    
    fn __radd__(self, other: Float64) -> Value:
        # When adding, the order is indifferent
        return self + other

    @always_inline
    fn __neg__(self) -> Value:
        return self * (-1)

    fn __iadd__ (inout self, other: Value):
        var out = self + other
        self = out
        
    fn __iadd__ (inout self, other: Float64):
        var out = self + other
        self = out
    
    fn __sub__(self, other: Value) -> Value:
        return self + (- other)

    fn __sub__(self, other: Float64) -> Value:
        return self + (- other)

    fn __rsub__(self, other: Float64) -> Value:
        return other + (- self)

    fn __truediv__(self, other: Value) -> Value:
        return self * (other ** -1)

    fn __truediv__(self, other: Float64) -> Value:
        return self * (other ** -1)

    fn __rtruediv__(self, other: Float64) -> Value:
        return other * (self ** -1)
    
    @always_inline
    fn backward_mul(mut self):
        self._prev[0][].grad[] += self._prev[1][].data[] * self.grad[]
        self._prev[1][].grad[] += self._prev[0][].data[] * self.grad[]

    @always_inline
    fn __mul__(self, other: Value) -> Value:
        var out = Value(data = (self.data[] * other.data[]), prev1 = self, prev2 = other, op = '*')
        return out

    fn __mul__(self, other: Float64) -> Value:
        var v = Value(other)
        return self * v
    
    fn __rmul__(self, other: Float64) -> Value:
        # When multiply, the order is indifferent
        return self * other

    fn __imul__ (inout self, other: Value):
        var out = self * other
        self = out
        
    fn __imul__ (inout self, other: Float64):
        var out = self * other
        self = out
    
    @always_inline
    fn __eq__(self, other: Self) -> Bool:
        return self.data.__is__(other.data) 

    @always_inline
    fn backward_pow(mut self):
        self._prev[0][].grad[] += (self._prev[1][].data[] * self._prev[0][].data[] ** (self._prev[1][].data[] - 1)) * self.grad[]

    @always_inline
    fn __pow__(self, other : Value) -> Value:
        var out = Value(data = (self.data[] ** other.data[]), prev1 = self, prev2 = other, op = '**')

        return out
    
    fn __pow__(self, other: Float64) -> Value:
        var v = Value(other)
        return self ** v

    @always_inline
    fn backward_relu(mut self):
        if self.data[] > 0:
            self._prev[0][].grad[] += self.grad[]
        else:
            self._prev[0][].grad[] += 0


    fn relu(self) -> Value:
        var out = Value(data = (Float64(0) if self.data[] < 0 else self.data[]), prev1 = self, op = 'ReLu')
        
        return out

    fn build_topo(self, mut visited: List[ArcPointer[Value]], mut topo: List[ArcPointer[Value]]):

        #TODO: This should be optimized
        for vis in visited:
            if self == vis[][]:
                return

        visited.append(self)

        if self._op == "": return
        """
        We don't need to add the leaf nodes to the topo list since they have no other node to propagate
        the grad.
        """

        #for v in self._prev: Value.build_topo(v[][], visited, topo)
        for v in self._prev: v[][].build_topo(visited, topo)
        """
        All the non-leaf nodes are the op result of at least one previous node.
        Usual arithmetic operations output nodes have two previous nodes.
        """
        
        topo.append(self)
        """Nodes ordered from last non-leaf (first layer) to output (usually loss)."""

    fn backward(mut self):
        #TODO: Optimize this, maybe with a stack.
        var visited = List[ArcPointer[Value]](List[ArcPointer[Value]]())
        var topo = List[ArcPointer[Value]](List[ArcPointer[Value]]())

        print("previous topo")

        self.build_topo(visited, topo)

        self.grad[] = 1.0
        """If the first node's grad is 0, the chain rule will make all previous nodes 0."""

        for v in reversed(topo):
            """
            This reversed give us the order: 
            From output node (loss) to last non-leaf node (usually first layer's neurons).
            """
            # Note the double [] needed, the first for the iterator and the second for the pointer
            if v[][]._op == "+":
                v[][].backward_add()
                continue
            if v[][]._op == "*":
                v[][].backward_mul()
                continue
            if v[][]._op == "**":
                v[][].backward_pow()
                continue
            if v[][]._op == "ReLu":
                v[][].backward_relu()
                continue
    
    fn __repr__(self) -> String:
        return "data: " + str(self.data[]) + " | grad: " + str(self.grad[]) + " | Op: " + self._op