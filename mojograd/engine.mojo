"""  """

from collections import Optional, List, Dict, InlineList, Set
from memory import UnsafePointer, memset_zero, ArcPointer
from memory import pointer

# Validate alias : fn() escaping -> None, alignment=1

struct Value():
    var data: Float32
    var grad : Float32

    var _func  : UnsafePointer[fn() escaping -> None, alignment=1]
    # Validate UnsafePointer[Tuple[UnsafePointer[Value], UnsafePointer[Value]]]
    # var _prev :  Set[Value]
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

    fn __init__(inout self, data: Float32, prev1: Value, op: String):
        
        self.data = Float32(data)
        self.grad = Float32(0)

        self._func  = UnsafePointer[fn() escaping -> None, alignment=1]() 

        self._prev1 = UnsafePointer[Value].alloc(1)
        self._prev1.init_pointee_copy(prev1)

        self._prev2 = UnsafePointer[Value]() 

        self._op = op

    fn __init__(inout self, data: Float32, prev1: Value, prev2: Value, op: String):
        
        self.data = Float32(data)
        self.grad = Float32(0)

        self._func  = UnsafePointer[fn() escaping -> None, alignment=1]() 

        self._prev1 = UnsafePointer[Value].alloc(1)
        self._prev1.init_pointee_copy(prev1)

        self._prev2 = UnsafePointer[Value].alloc(1)
        self._prev2.init_pointee_copy(prev2)

        self._op = op

    fn __moveinit__(out self, owned existing: Self):
        # Validate ^
        self.data = existing.data
        self.grad = existing.grad
        # Validate
        self._func = existing._func
        self._prev1 = existing._prev1
        self._prev2 = existing._prev2
        self._op = existing._op
    
    fn __copyinit__(out self, existing: Self):
        self.data = existing.data
        self.grad = existing.grad
        # Validate
        self._func = existing._func
        self._prev1 = existing._prev1
        self._prev2 = existing._prev2
        self._op = existing._op

    fn __add__(self, other: Value) -> Value:
        var out = Value(data = (self.data + other.data), prev1 = self, prev2 = other, op = '+')

        fn _backward() -> None:
            print("Trying _backward add")
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
    
    fn __iadd__ (inout self, other: Value):
        self.data += other.data
        
    fn __iadd__ (inout self, other: Float32):
        # check if prev1 needs to be assigned
        self.data += other 

    fn __mul__(self, other: Value) -> Value:
        var out = Value(data = (self.data * other.data), prev1 = self, prev2 = other, op = '*')

        fn _backward() -> None:
            print("Trying _backward mul")
            var out_grad = out.grad
            print("out_grad: ", out_grad)
            var _self = UnsafePointer[Value].address_of(self) 
            var _other = UnsafePointer[Value].address_of(other)
            _self[].grad += _other[].data * out_grad
            _other[].grad += _self[].data * out_grad
        
        out._func = UnsafePointer[fn() escaping -> None, alignment=1].alloc(1)

        # Validate ^
        out._func.init_pointee_move(_backward)

        return out

    fn __mul__(self, other: Float32) -> Value:
        # We are only making the conversion and reusing the value __mul__ logic
        var v = Value(other)
        return self.__mul__(v)
    
    fn __eq__(self, other: Self) -> Bool:
        return UnsafePointer[Value].address_of(self) == UnsafePointer[Value].address_of(other)

    
    fn __pow__(self, other : Value) -> Value:
        var out = Value(data = (self.data ** other.data), prev1 = self, prev2 = other, op = '**')

        fn _backward() -> None:
            print("Trying _backward pow")
            var out_grad = out.grad
            print("out_grad: ", out_grad)
            var _self = UnsafePointer[Value].address_of(self) 
            var _other = UnsafePointer[Value].address_of(other)
            _self[].grad += (_other[].data * _self[].data ** (_other[].data - 1)) *  out_grad # Validate this out_grad
        
        out._func = UnsafePointer[fn() escaping -> None, alignment=1].alloc(1)

        # Validate ^
        out._func.init_pointee_move(_backward)

        return out
    
    fn __pow__(self, other: Float32) -> Value:
        var v = Value(other)
        return self.__pow__(v)

    fn relu(self) -> Value:
        var out = Value(data = (Float32(0) if self.data < 0 else self.data), prev1 = self, op = 'ReLu')

        fn _backward():
            var _out = UnsafePointer[Value].address_of(out)
            var _self = UnsafePointer[Value].address_of(self)
            _self[].grad += (Float32(0) if _self[].data < 0 else _self[].data) * _out[].grad
        
        out._func = UnsafePointer[fn() escaping -> None, alignment=1].alloc(1)
      
        out._func.init_pointee_move(_backward)
        
        return out

    @staticmethod
    # Validate UnsafePointer[List[UnsafePointer[Value]]]
    fn build_topo(self, mut visited: List[UnsafePointer[Value]], mut topo: List[UnsafePointer[Value]]):

        if UnsafePointer[Value].address_of(self) == UnsafePointer[Value]():
            return

        print("Build topo")

        if UnsafePointer[Value].address_of(self) in visited:
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

        print("previous topo")
        print(len(topo))
        print(len(visited))

        Value.build_topo(self, visited, topo)

        self.grad = Float32(1.0)

        for v in reversed(topo):
            print("for reversed")
            print(len(topo))
            # Note the double [] needed, the first for the iterator and the second for the pointer
            v[][]._func[]()
            v[][].__print()

    
    fn __print(self):
        print("data: ", self.data, "grad: ", self.grad, "Op: ", self._op)
    
        
fn main():
    fn test1():
        var a = Value(data = 2.0)
        var b = Value(data = 3.0)
        var c = Float32(2.0)
        var d = b ** c
        var e = a + c
        
        try:
            e.backward()

            a.__print()
            b.__print()
            d.__print()
            e.__print()
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
    
    fn test2():
        a2 = Value(4.0)
        b2 = Value(2.0)
        c2 = a2 + b2
        d2 = a2 * b2 + b2**3
        c2 += c2 + 1

        #d = a * b + b**3

        if a2._prev1 != UnsafePointer[Value]():
            a2._prev1.destroy_pointee()
            a2._prev1.free()
        
        if a2._prev2 != UnsafePointer[Value]():
            a2._prev2.destroy_pointee()
            a2._prev2.free()

        if b2._prev1 != UnsafePointer[Value]():
            b2._prev1.destroy_pointee()
            b2._prev1.free()
        
        if b2._prev2 != UnsafePointer[Value]():
            b2._prev2.destroy_pointee()
            b2._prev2.free()

    #test1()
    test2()
