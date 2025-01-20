"""  """

from collections import Optional, List, Dict, InlineList
from memory import UnsafePointer, memset_zero, ArcPointer
from memory import pointer

# Validate alias : fn() escaping -> None, alignment=1

struct Value():
    var data: Float32
    var grad : Float32

    var _func  : UnsafePointer[fn() escaping -> None, alignment=1]
    # Validate UnsafePointer[Tuple[UnsafePointer[Value], UnsafePointer[Value]]]
    var _prev1 : UnsafePointer[Value]
    var _prev2 : UnsafePointer[Value]
    var _op : String

    fn __init__(inout self, data: Float32):
        
        self.data = Float32(data)
        self.grad = Float32(0)

        self._func  = UnsafePointer[fn() escaping -> None, alignment=1]() 
        self._prev1 = UnsafePointer[Value]() 
        self._prev2 = UnsafePointer[Value]() 

        self._op = String('') 

    fn __moveinit__(out self, owned existing: Self):
        # Validate ^
        self.data = existing.data
        self.grad = existing.grad
        self._func = existing._func
        self._prev1 = existing._prev1
        self._prev2 = existing._prev2
        self._op = existing._op
    
    fn __copyinit__(out self, existing: Self):
        self.data = existing.data
        self.grad = existing.grad
        self._func = existing._func
        self._prev1 = existing._prev1
        self._prev2 = existing._prev2
        self._op = existing._op

    fn __add__(self, other: Value) -> Value:
        var out = Value(data = (self.data + other.data))

        out._prev1 = UnsafePointer[Value].alloc(1)
        out._prev1.init_pointee_copy(self)

        out._prev2 = UnsafePointer[Value].alloc(1)
        out._prev2.init_pointee_copy(other)
        
        out._op = String('+')

        fn _backward() -> None:
            print("Trying _backward")
            var out_grad = out.grad
            print("out_grad: ", out_grad)
            var _self = UnsafePointer[Value].address_of(self) 
            var _other = UnsafePointer[Value].address_of(other)
            _self[].grad += out_grad
            _other[].grad += out_grad
        
        out._func = UnsafePointer[fn() escaping -> None, alignment=1].alloc(1)

        # Validate ^
        out._func.init_pointee_move(_backward)

        return out

    fn __add__(self, other: Float32) -> Value:
        # We are only making the conversion and reusing the value __add__ logic
        var v = Value(other)
        return self.__add__(v)
    
    fn __eq__(self, other: Self) -> Bool:
        return UnsafePointer[Value].address_of(self) == UnsafePointer[Value].address_of(other)

    
    fn __pow__(self, other : Value) -> Value:
        var out = Value(data = (self.data ** other.data)) 
         # We need to add the previous nodes
        out._prev1 = UnsafePointer[Value].alloc(1)
        out._prev1.init_pointee_copy(self)

        out._prev2 = UnsafePointer[Value].alloc(1)
        out._prev2.init_pointee_copy(other) 

        out._op = String('**')

        return out
    
    fn __pow__(self, other: Float32) -> Value:
        var v = Value(other)
        return self.__pow__(v)
    
    @staticmethod
    fn backward_pow(mut v: Value):
        print("backward_pow")
        v.__print()

        if v._prev1 == UnsafePointer[Value]() or v._prev2 == UnsafePointer[Value]():
            return
        
        v._prev1[].grad = (v._prev1[].grad 
                    + (v._prev2[].data * v._prev1[].data ** (v._prev2[].data - 1) * v._prev2[].grad ))
        
    
    @staticmethod
    fn backward_add(mut v: Value):
        print("backward_add")
        v.__print()

        if v._prev1 != UnsafePointer[Value]():
            var _children1 = v._prev1[]
            print("_children1.grad = ", _children1.grad, "v.grad = ",v.grad)
            v._prev1[].grad = _children1.grad + v.grad
        
        if v._prev2 != UnsafePointer[Value]():
            var _children2 = v._prev2[]
            print("_children2.grad = ", _children2.grad, "v.grad = ",v.grad)
            v._prev2[].grad = _children2.grad + v.grad
    
    @staticmethod
    fn _backward(mut v: Value):
        print("op")
        print(v._op)

        print("_backward")
        v.__print()

        if v._op == '+':
            print("Option +")
            Value.backward_add(v)
            return
        if v._op == '**':
            print("Option **")
            Value.backward_pow(v)
            return
        
        print("OP not suported")

    @staticmethod
    # Validate UnsafePointer[List[UnsafePointer[Value]]]
    fn build_topo(self, mut visited: List[UnsafePointer[Value]], mut topo: List[UnsafePointer[Value]]):
        if UnsafePointer[Value].address_of(self) == UnsafePointer[Value]():
            return

        var is_visited = Bool(False)

        var size = Int(len(visited))

        print("Build topo")

        for i in range(size):
            if self == visited[i][]:
                is_visited = True
        
        if is_visited:
            return
            
        print("Entering not visited")
        visited.append(UnsafePointer.address_of(self))
        print(len(visited))
        if self._prev1 != UnsafePointer[Value]():
            print("Entered _prev1 != UnsafePointer[Value]()")
            var _children1 = self._prev1[]
            print(_children1.data)
            if _children1._prev1 != UnsafePointer[Value]():
                Value.build_topo(_children1, visited, topo)

        if self._prev2 != UnsafePointer[Value]():
            print("Entered _prev2 != UnsafePointer[Value]()")
            var _children2 = self._prev2[]
            print(_children2.data)
            if _children2._prev2 != UnsafePointer[Value]():
                Value.build_topo(_children2, visited, topo)
        
        topo.append(UnsafePointer[Value].address_of(self))
        print(len(topo))

    fn backward(mut self):
        # Maybe this needs to be a pointer, we'll see
        var visited = List[UnsafePointer[Value]]()
        var topo = List[UnsafePointer[Value]]()

        print("previo a topo")
        print(len(topo))
        print(len(visited))

        Value.build_topo(self, visited, topo)

        self.grad = Float32(1.0)

        for v in reversed(topo):
            print("for reversed")
            print(len(topo))
            # Note the double [] needed, the first for the iterator and the second for the pointer
            v[][].__print()
            Value._backward(v[][])
    
    fn __print(self):
        print("data: ", self.data, "grad: ", self.grad, "Op: ", self._op)
    
        
fn main():
    var a = Value(data = 2.0)
    var b = Value(data = 3.0)
    var c = Float32(2.0)
    var d = b ** c
    var e = a + c
    
    e.backward()

    a.__print()
    b.__print()
    d.__print()
    e.__print()

    try:
        e._func[]()
    finally:
        if a._prev1 != UnsafePointer[Value]():
            a._prev1.destroy_pointee()
            a._prev1.free()
        
        if a._prev2 != UnsafePointer[Value]():
            a._prev2.destroy_pointee()
            a._prev2.free()

        if b._prev1 != UnsafePointer[Value]():
            b._prev1.destroy_pointee()
            b._prev1.free()
        
        if b._prev2 != UnsafePointer[Value]():
            b._prev2.destroy_pointee()
            b._prev2.free()

        if d._prev1 != UnsafePointer[Value]():
            d._prev1.destroy_pointee()
            d._prev1.free()
        
        if d._prev2 != UnsafePointer[Value]():
            d._prev2.destroy_pointee()
            d._prev2.free()

        if e._prev1 != UnsafePointer[Value]():
            e._prev1.destroy_pointee()
            e._prev1.free()
        
        if e._prev2 != UnsafePointer[Value]():
            e._prev2.destroy_pointee()
            e._prev2.free() 

    # May god help us
    #var f = Value(data = 3.0)
    #var g = Value(data = 4.0)
    #var h = f + g
    #h.backward()

    if a._prev1 != UnsafePointer[Value]():
        a._prev1.destroy_pointee()
        a._prev1.free()
    
    if a._prev2 != UnsafePointer[Value]():
        a._prev2.destroy_pointee()
        a._prev2.free()

    if b._prev1 != UnsafePointer[Value]():
        b._prev1.destroy_pointee()
        b._prev1.free()
    
    if b._prev2 != UnsafePointer[Value]():
        b._prev2.destroy_pointee()
        b._prev2.free()

    if d._prev1 != UnsafePointer[Value]():
        d._prev1.destroy_pointee()
        d._prev1.free()
    
    if d._prev2 != UnsafePointer[Value]():
        d._prev2.destroy_pointee()
        d._prev2.free()

    if e._prev1 != UnsafePointer[Value]():
        e._prev1.destroy_pointee()
        e._prev1.free()
    
    if e._prev2 != UnsafePointer[Value]():
        e._prev2.destroy_pointee()
        e._prev2.free()

    #if h._prev1 != UnsafePointer[Value]():
    #    h._prev1.destroy_pointee()
    #    h._prev1.free()
    #
    #if h._prev2 != UnsafePointer[Value]():
    #    h._prev2.destroy_pointee()
    #    h._prev2.free()