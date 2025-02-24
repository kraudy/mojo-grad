
""" 
"""

from collections import Optional, List, Dict, InlineList, Set
from memory import UnsafePointer, memset_zero, ArcPointer, pointer, Pointer

# Global counter for unique IDs
var global_id_counter: Int = 0

fn get_next_id() -> Int:
    global_id_counter += 1
    return global_id_counter

struct Value():
    var id: Int  # Unique sequential ID

    #TODO: Validate if this ArcPointer is needed
    var data: ArcPointer[Float64]
    var grad :  ArcPointer[Float64]

    var _backward  : fn (v: ArcPointer[Value]) -> None
    # Validate UnsafePointer[Tuple[UnsafePointer[Value], UnsafePointer[Value]]]
    # var _prev :  Set[Value]
    #TODO: There has to be a better way to do this.
    var _prev : List[ArcPointer[Value]]

    var _op : String

    @staticmethod
    fn no_backprop(v: ArcPointer[Value]) -> None: pass

    @always_inline
    fn __init__(out self, data: Float64):
        self.id = get_next_id()
        self.data = data
        self.grad =  0.0

        self._backward  = Value.no_backprop
        self._prev = List[ArcPointer[Value]]()

        self._op = String('') 

    fn __init__(out self, data: Float64, prev1: Value, op: String):
        self.id = get_next_id()
        self.data = data
        self.grad = 0.0

        self._backward  = Value.no_backprop

        self._prev = List[ArcPointer[Value]]()
        self._prev.append(ArcPointer[Value](prev1))

        self._op = op

    fn __init__(out self, data: Float64, prev1: Value, prev2: Value, op: String):
        self.id = get_next_id()
        self.data = data
        self.grad = 0.0

        self._backward  = Value.no_backprop

        self._prev = List[ArcPointer[Value]]()
        self._prev.append(ArcPointer[Value](prev1))
        self._prev.append(ArcPointer[Value](prev2))

        self._op = op

    fn __moveinit__(out self, owned existing: Self):
        self.id = existing.id
        self.data = existing.data^
        self.grad = existing.grad^
        # Validate
        self._backward = existing._backward#^
        self._prev = existing._prev^
        self._op = existing._op^
    
    fn __copyinit__(out self, existing: Self):
        self.id = existing.id
        self.data = existing.data
        self.grad = existing.grad
        self._backward = existing._backward
        self._prev = existing._prev
        self._op = existing._op
      
    #@always_inline
    #fn backward_add(mut self):
    #    self._prev[0][].grad[] += self.grad[]
    #    self._prev[1][].grad[] += self.grad[]

    @always_inline
    fn __add__(self, other: Value) -> Value:
        var out = Value(data = (self.data[] + other.data[]), prev1 = self, prev2 = other, op = '+')
        fn _backward(
            v: ArcPointer[Value]
        ) -> None:
            v[]._prev[0][].grad[] += v[].grad[]
            v[]._prev[1][].grad[] += v[].grad[]

        out._backward = _backward
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
    
    #@always_inline
    #fn backward_mul(mut self):
    #    self._prev[0][].grad[] += self._prev[1][].data[] * self.grad[]
    #    self._prev[1][].grad[] += self._prev[0][].data[] * self.grad[]

    @always_inline
    fn __mul__(self, other: Value) -> Value:
        var out = Value(data = (self.data[] * other.data[]), prev1 = self, prev2 = other, op = '*')
        fn _backward(
            v: ArcPointer[Value]
        ) -> None:
            v[]._prev[0][].grad[] += v[]._prev[1][].data[] * v[].grad[]
            v[]._prev[1][].grad[] += v[]._prev[0][].data[] * v[].grad[]

        out._backward = _backward

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
    
    #@always_inline
    #fn __eq__(self, other: Self) -> Bool:
    #    return self.data.__is__(other.data) 
        #return UnsafePointer.address_of(self) == UnsafePointer.address_of(other)

    #@always_inline
    #fn backward_pow(mut self):
    #    self._prev[0][].grad[] += (self._prev[1][].data[] * self._prev[0][].data[] ** (self._prev[1][].data[] - 1)) * self.grad[]

    @always_inline
    fn __pow__(self, other : Value) -> Value:
        var out = Value(data = (self.data[] ** other.data[]), prev1 = self, prev2 = other, op = '**')
        fn _backward(
            v: ArcPointer[Value]
        ) -> None:
            v[]._prev[0][].grad[] += (v[]._prev[1][].data[] * v[]._prev[0][].data[] ** (v[]._prev[1][].data[] - 1)) * v[].grad[]

        out._backward = _backward
        return out
    
    fn __pow__(self, other: Float64) -> Value:
        var v = Value(other)
        return self ** v

    #@always_inline
    #fn backward_relu(mut self):
    #    if self.data[] > 0:
    #        self._prev[0][].grad[] += self.grad[]
    #    else:
    #        self._prev[0][].grad[] += 0


    fn relu(self) -> Value:
        var out = Value(data = (Float64(0) if self.data[] < 0 else self.data[]), prev1 = self, op = 'ReLu')
        fn _backward(
            v: ArcPointer[Value]
        ) -> None:
            if v[].data[] > 0:
              v[]._prev[0][].grad[] += v[].grad[]
            else:
              v[]._prev[0][].grad[] += 0

        out._backward = _backward
        return out

    fn build_topo(self, mut visited: Set[Int], mut topo: List[Value]):
        if self.id in visited: return

        visited.add(self.id)

        if self._op == "": return
        """
        We don't need to add the leaf nodes to the topo list since they have no other node to propagate
        the grad.
        """

        for v in self._prev: v[][].build_topo(visited, topo)
        """
        All the non-leaf nodes are the op result of at least one previous node.
        Usual arithmetic operations output nodes have two previous nodes.
        """
        
        topo.append(self)
        """Nodes ordered from last non-leaf (first layer) to output (usually loss)."""

    fn back_prop(self, mut topo: List[Value]):
        self.grad[] = 1.0
        """If the first node's grad is 0, the chain rule will affect badly all previous nodes."""

        for v in reversed(topo):
            """
            This reversed give us the order: 
            From output node (loss) to last non-leaf node (usually first layer's neurons).
            """
            if v[]._op == "+":
                v[]._backward(v[])
                continue
            if v[]._op == "*":
                #v[].backward_mul()
                v[]._backward(v[])
                continue
            if v[]._op == "**":
                #v[].backward_pow()
                v[]._backward(v[])
                continue
            if v[]._op == "ReLu":
                #v[].backward_relu()
                v[]._backward(v[])
                continue

    fn backward(mut self):
        #TODO: Optimize this, maybe with a stack.
        var visited = Set[Int](Set[Int]())
        var topo = List[Value](List[Value]())

        self.build_topo(visited, topo)

        self.back_prop(topo)
    
    fn __repr__(self) -> String:
        return "data: " + str(self.data[]) + " | grad: " + str(self.grad[]) + " | Op: " + self._op