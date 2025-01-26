"""  """

from collections import Optional, List, Dict, InlineList, Set
from memory import UnsafePointer, memset_zero, ArcPointer, pointer


# Validate alias : fn() escaping -> None, alignment=1

struct Value():
    var data: Float32
    var grad : Float32

    var _func  : UnsafePointer[fn() escaping -> None, alignment=1]
    # Validate UnsafePointer[Tuple[UnsafePointer[Value], UnsafePointer[Value]]]
    # var _prev :  Set[Value]
    var _prev : List[ArcPointer[Value]]

    var _op : String

    fn __init__(inout self, data: Float32):
        
        self.data = Float32(data)
        self.grad = Float32(0)

        self._func  = UnsafePointer[fn() escaping -> None, alignment=1]() 
        self._prev = List[ArcPointer[Value]]()


        self._op = String('') 

    fn __init__(inout self, data: Float32, prev1: Value, op: String):
        
        self.data = Float32(data)
        self.grad = Float32(0)

        self._func  = UnsafePointer[fn() escaping -> None, alignment=1]() 

        self._prev = List[ArcPointer[Value]]()
        self._prev.append(ArcPointer[Value](prev1))

        self._op = op

    fn __init__(inout self, data: Float32, prev1: Value, prev2: Value, op: String):
        
        self.data = Float32(data)
        self.grad = Float32(0)

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
      
    fn backward_add(mut self):
        self._prev[0][].grad += self.grad
        self._prev[1][].grad += self.grad

    fn __add__(self, other: Value) -> Value:
        var out = Value(data = (self.data + other.data), prev1 = self, prev2 = other, op = '+')

        return out

    fn __add__(self, other: Float32) -> Value:
        # We are only making the conversion and reusing the value __add__ logic
        var v = Value(other)
        return self + v
    
    fn __radd__(self, other: Float32) -> Value:
        # When adding the order is indifferent
        return self.__add__(other)

    fn __neg__(self) -> Value:
        return self * (-1)

    fn __iadd__ (inout self, other: Value):
        var out = self + other
        self = out
        
    fn __iadd__ (inout self, other: Float32):
        var out = self + other
        self = out
    
    fn __sub__(self, other: Value) -> Value:
        return self + (- other)

    fn __sub__(self, other: Float32) -> Value:
        return self + (- other)

    fn __rsub__(self, other: Float32) -> Value:
        return other + (- self)

    fn __truediv__(self, other: Value) -> Value:
        return self * (other ** -1)

    fn __truediv__(self, other: Float32) -> Value:
        return self * (other ** -1)

    fn __rtruediv__(self, other: Float32) -> Value:
        return other * (self ** -1)
    
    fn backward_mul(mut self):
        self._prev[0][].grad += self._prev[1][].data * self.grad
        self._prev[1][].grad += self._prev[0][].data * self.grad

    fn __mul__(self, other: Value) -> Value:
        var out = Value(data = (self.data * other.data), prev1 = self, prev2 = other, op = '*')

        return out

    fn __mul__(self, other: Float32) -> Value:
        # We are only making the conversion and reusing the value __mul__ logic
        var v = Value(other)
        return self * v
    
    fn __rmul__(self, other: Float32) -> Value:
        # When adding the order is indifferent
        return self.__mul__(other)
    
    fn __eq__(self, other: Self) -> Bool:
        return UnsafePointer[Value].address_of(self) == UnsafePointer[Value].address_of(other)

    fn backward_pow(mut self):
        self._prev[0][].grad += (self._prev[1][].data * self._prev[0][].data ** (self._prev[1][].data - 1)) * self.grad

    fn __pow__(self, other : Value) -> Value:
        var out = Value(data = (self.data ** other.data), prev1 = self, prev2 = other, op = '**')

        return out
    
    fn __pow__(self, other: Float32) -> Value:
        var v = Value(other)
        return self ** v

    fn backward_relu(mut self):
        self._prev[0][].grad += (Float32(0) if self.data < 0 else self.grad) 


    fn relu(self) -> Value:
        var out = Value(data = (Float32(0) if self.data < 0 else self.data), prev1 = self, op = 'ReLu')
        
        return out

    @staticmethod
    # Validate UnsafePointer[List[UnsafePointer[Value]]]
    # mut makes the changes visible to the calle
    fn build_topo(self_ptr: ArcPointer[Value], mut visited: List[ArcPointer[Value]], mut topo: List[ArcPointer[Value]]):

        print("Build topo")

        #if UnsafePointer[Value].address_of(self_ptr) in visited:
        #    return
        #if ArcPointer[Value].address_of(self_ptr[]) in visited:
        for vis in visited:
            if self_ptr.__is__(vis[]):
                return
            
        print("Entering not visited")
        #visited.append(UnsafePointer.address_of(self_ptr))
        visited.append(self_ptr)
        print(len(visited))
        if len(self_ptr[]._prev) > 0:
            print("Entered _prev1 != UnsafePointer[Value]()")
            Value.build_topo(self_ptr[]._prev[0][], visited, topo)

        if len(self_ptr[]._prev) == 2:
            print("Entered _prev2 != UnsafePointer[Value]()")
            Value.build_topo(self_ptr[]._prev[1][], visited, topo)
        
        #topo.append(UnsafePointer[Value].address_of(self_ptr))
        topo.append(self_ptr)
        print(len(topo))

    fn backward(mut self):
        # Maybe this needs to be a pointer, we'll see
        var visited = List[ArcPointer[Value]]()
        var topo = List[ArcPointer[Value]]()

        print("previous topo")
        print(len(topo))
        print(len(visited))

        Value.build_topo(self, visited, topo)

        self.grad = Float32(1.0)
        topo[-1][].grad = Float32(1.0)

        print(repr(self))
        print(repr(topo[-1][]))
        print("================")

        for v_ptr in reversed(topo):
            print("for reversed")
            # Note the double [] needed, the first for the iterator and the second for the pointer
            var v = v_ptr[][]
            print(repr(v))
            if v._op == "+":
                print(repr(v._prev[0][]))
                print(repr(v._prev[1][]))
                v.backward_add()
                print(repr(v._prev[0][]))
                print(repr(v._prev[1][]))
            elif v._op == "*":
                print(repr(v._prev[0][]))
                print(repr(v._prev[1][]))
                v.backward_mul()
                print(repr(v._prev[0][]))
                print(repr(v._prev[1][]))
            elif v._op == "**":
                print(repr(v._prev[0][]))
                v.backward_pow()
                print(repr(v._prev[0][]))
            elif v._op == "ReLu":
                print(repr(v._prev[0][]))
                v.backward_relu()
                print(repr(v._prev[0][]))

          
            #if v._func != UnsafePointer[fn() escaping -> None, alignment=1]():
            #    v._func[]()
            
            #print(repr(v._prev1[]))
            #print(repr(v._prev2[]))

    
    fn __print(self):
        print("data: ", self.data, "grad: ", self.grad, "Op: ", self._op)
    
    fn __repr__(self) -> String:
        return "data: " + str(self.data) + " | grad: " + str(self.grad) + " | Op: " + self._op
    
    fn destroy(owned self):
        """Owned assures we get the unique ownership of the value, so we can free it."""
        #if self._prev1 != UnsafePointer[Value]():
        #    self._prev1.destroy_pointee()
        #    self._prev1.free()
            
        #if self._prev2 != UnsafePointer[Value]():
        #    self._prev2.destroy_pointee()
        #    self._prev2.free()
        pass

    
