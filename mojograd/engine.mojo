
""" 
"""

from collections import Optional, List, Dict, InlineList, Set
from memory import UnsafePointer, memset_zero, ArcPointer, pointer, Pointer


# Validate alias : fn() escaping -> None, alignment=1

struct Value():
    var data: ArcPointer[Float64]
    var grad :  ArcPointer[Float64]

    #TODO: Check if this scope function can be now implemented
    var _func  : UnsafePointer[fn() escaping -> None, alignment=1]
    # Validate UnsafePointer[Tuple[UnsafePointer[Value], UnsafePointer[Value]]]
    # var _prev :  Set[Value]
    #TODO: There has to be a better way to do this.
    var _prev : List[ArcPointer[Value]]

    var _op : String

    #TODO: Validate if this ArcPointer[Float64] is needed or just Float64 since data is already ArcPointer
    #TODO: Consider changin inout to out
    #fn __init__(inout self, data: ArcPointer[Float64]):
    #    
    #    self.data = data
    #    self.grad =  ArcPointer[Float64](0)

    #    self._func  = UnsafePointer[fn() escaping -> None, alignment=1]() 
    #    self._prev = List[ArcPointer[Value]]()


    #    self._op = String('') 

    fn __init__(inout self, data: Float64):
        
        #self.data = ArcPointer[Float64](data)
        self.data = data
        #self.grad =  ArcPointer[Float64](0)
        self.grad =  0.0

        self._func  = UnsafePointer[fn() escaping -> None, alignment=1]() 
        self._prev = List[ArcPointer[Value]]()


        self._op = String('') 

    #fn __init__(inout self, data: ArcPointer[Float64], prev1: Value, op: String):
    #    
    #    self.data = data
    #    self.grad =  ArcPointer[Float64](0)

    #    self._func  = UnsafePointer[fn() escaping -> None, alignment=1]() 

    #    self._prev = List[ArcPointer[Value]]()
    #    self._prev.append(ArcPointer[Value](prev1))

    #    self._op = op

    fn __init__(inout self, data: Float64, prev1: Value, op: String):
        
        self.data = data
        self.grad = 0.0

        self._func  = UnsafePointer[fn() escaping -> None, alignment=1]() 

        self._prev = List[ArcPointer[Value]]()
        self._prev.append(ArcPointer[Value](prev1))

        self._op = op

    #fn __init__(inout self, data: ArcPointer[Float64], prev1: Value, prev2: Value, op: String):
    #    self.data = data
    #    self.grad =  ArcPointer[Float64](0)

    #    self._func  = UnsafePointer[fn() escaping -> None, alignment=1]() 

    #    self._prev = List[ArcPointer[Value]]()
    #    self._prev.append(ArcPointer[Value](prev1))
    #    self._prev.append(ArcPointer[Value](prev2))

    #    self._op = op

    fn __init__(inout self, data: Float64, prev1: Value, prev2: Value, op: String):
        self.data = data
        self.grad = 0.0

        self._func  = UnsafePointer[fn() escaping -> None, alignment=1]() 

        self._prev = List[ArcPointer[Value]]()
        self._prev.append(ArcPointer[Value](prev1))
        self._prev.append(ArcPointer[Value](prev2))

        self._op = op

    fn __moveinit__(out self, owned existing: Self):
        # Validate ^
        self.data = existing.data
        self.grad = existing.grad
        # Validate
        self._func = existing._func
        self._prev = existing._prev
        self._op = existing._op
    
    fn __copyinit__(out self, existing: Self):
        self.data = existing.data
        self.grad = existing.grad
        # Validate pointee copy
        self._func = existing._func
        self._prev = existing._prev
        self._op = existing._op
        #self =  existing
      
    fn backward_add(mut self):
        self._prev[0][].grad[] += self.grad[]
        self._prev[1][].grad[] += self.grad[]

    fn __add__(self, other: Value) -> Value:
        #var out = Value(data = (ArcPointer[Float64](self.data[] + other.data[])), prev1 = self, prev2 = other, op = '+')
        var out = Value(data = (self.data[] + other.data[]), prev1 = self, prev2 = other, op = '+')

        return out

    fn __add__(self, other: Float64) -> Value:
        # We are only making the conversion and reusing the value __add__ logic
        var v = Value(other)
        return self + v
    
    fn __radd__(self, other: Float64) -> Value:
        # When adding the order is indifferent
        return self.__add__(other)

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
    
    fn backward_mul(mut self):
        self._prev[0][].grad[] += self._prev[1][].data[] * self.grad[]
        self._prev[1][].grad[] += self._prev[0][].data[] * self.grad[]

    fn __mul__(self, other: Value) -> Value:
        #var out = Value(data = (ArcPointer[Float64](self.data[] * other.data[])), prev1 = self, prev2 = other, op = '*')
        var out = Value(data = (self.data[] * other.data[]), prev1 = self, prev2 = other, op = '*')

        return out

    fn __mul__(self, other: Float64) -> Value:
        # We are only making the conversion and reusing the value __mul__ logic
        var v = Value(other)
        return self * v
    
    fn __rmul__(self, other: Float64) -> Value:
        # When adding the order is indifferent
        #TODO: Change to self * other
        return self.__mul__(other)

    fn __imul__ (inout self, other: Value):
        var out = self * other
        self = out
        
    fn __imul__ (inout self, other: Float64):
        var out = self * other
        self = out
    
    fn __eq__(self, other: Self) -> Bool:
        return self.data.__is__(other.data) 

    fn backward_pow(mut self):
        self._prev[0][].grad[] += (self._prev[1][].data[] * self._prev[0][].data[] ** (self._prev[1][].data[] - 1)) * self.grad[]

    fn __pow__(self, other : Value) -> Value:
        var out = Value(data = (self.data[] ** other.data[]), prev1 = self, prev2 = other, op = '**')

        return out
    
    fn __pow__(self, other: Float64) -> Value:
        var v = Value(other)
        return self ** v

    fn backward_relu(mut self):
        if self.data[] > 0:
            self._prev[0][].grad[] += self.grad[]
        else:
            self._prev[0][].grad[] += 0


    fn relu(self) -> Value:
        var out = Value(data = (Float64(0) if self.data[] < 0 else self.data[]), prev1 = self, op = 'ReLu')
        
        return out

    @staticmethod
    #TODO: Change to mut self here and remove static
    fn build_topo(self_ptr: ArcPointer[Value], mut visited: List[ArcPointer[Value]], mut topo: List[ArcPointer[Value]]):

        #TODO: This should be optimized
        for vis in visited:
            if self_ptr[] == vis[][]:
                return

        visited.append(self_ptr)

        if self_ptr[]._op == "": return
        """
        We don't need to add the leaf nodes to the topo list since they have no other node to propagate
        the grad.
        """

        Value.build_topo(self_ptr[]._prev[0], visited, topo)
        """All the non-leaf nodes are the op result of at least one previous node."""

        if len(self_ptr[]._prev) == 2: Value.build_topo(self_ptr[]._prev[1], visited, topo)
        """Nodes tha are the result of usual arithmetic operations have two previous nodes."""
        
        topo.append(self_ptr)
        """Nodes ordered from last non-leaf (first layer) node to output node (usually loss node)."""

    fn backward(mut self):
        #TODO: Optimize this, maybe with a stack.
        var visited = List[ArcPointer[Value]](List[ArcPointer[Value]]())
        var topo = List[ArcPointer[Value]](List[ArcPointer[Value]]())

        print("previous topo")
        print(len(topo))
        print(len(visited))

        #TODO: Validate if this pointer is needed
        var self_ref = ArcPointer[Value](self)

        #TODO: Validate just passing self
        Value.build_topo(self_ref, visited, topo)

        self.grad[] = 1.0

        print(repr(self))
        print(repr(topo[-1][]))
        print("================")

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