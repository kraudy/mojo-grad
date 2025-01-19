"""  """

from collections import Optional, List, Dict, InlineList
from memory import UnsafePointer, memset_zero, ArcPointer
from memory import pointer


# Understand what this does
#struct Value(CollectionElement, Writable, Stringable): # These two gives error
#struct Value(CollectionElement): # Understand what this does
# Validates if AnyTypes behaves correlty
struct Value():
    var data: Float32
    var grad : Float32

    var _prev1 : UnsafePointer[Value]
    var _prev2 : UnsafePointer[Value]
    var _op : String

    fn __init__(inout self, data: Float32):
        
        #self.data = UnsafePointer[Scalar[DType.float32]].alloc(1)
        self.data = Float32(data)
        self.grad = Float32(0)


        self._prev1 = UnsafePointer[Value]() 
        self._prev2 = UnsafePointer[Value]() 

        self._op = String('') 

    fn __moveinit__(out self, owned existing: Self):
        self.data = existing.data
        self.grad = existing.grad
        self._prev1 = existing._prev1
        self._prev2 = existing._prev2
        self._op = existing._op
    
    fn __copyinit__(out self, existing: Self):
        self.data = existing.data
        self.grad = existing.grad
        self._prev1 = existing._prev1
        self._prev2 = existing._prev2
        self._op = existing._op

    fn __add__(self, other: Value) -> Value:
        var out = Value(data = (self.data + other.data))
        # Maybe i can just append this
        out._prev1 = UnsafePointer[Value].alloc(1)
        out._prev1.init_pointee_move(self)

        out._prev2 = UnsafePointer[Value].alloc(1)
        out._prev2.init_pointee_move(other)
        
        out._op = String('+')

        return out

    fn __add__(self, other: Float32) -> Value:
        # If the value passed is not Value
        # This isa can be useful to accept multiples types on a paramete
        var v = Value(other)
        # We are only making the conversion and reusing the value logic
        return self.__add__(v)
    
    fn __eq__(self, other: Self) -> Bool:
        #return self is other
        return UnsafePointer[Value].address_of(self) == UnsafePointer[Value].address_of(other)

    
    fn __pow__(self, other : Value) -> Value:
        var out = Value(data = (self.data ** other.data)) 
         # We need to add the previous nodes
        out._prev1 = UnsafePointer[Value].alloc(1)
        out._prev1.init_pointee_move(self)

        out._prev2 = UnsafePointer[Value].alloc(1)
        out._prev2.init_pointee_move(other) 

        out._op = String[]('**')

        return out
    
    fn __pow__(self, other: Float32) -> Value:
        var v = Value(other)
        return self.__pow__(v)
    
    fn _backward(mut v: Value):
        print("op")
        print(v._op)

        print("_backward")
        v.__print()

        if v._op == '+':
            print("Option +")
            #Value.backward_add(v)
            return
        if v._op == '**':
            print("Option **")
            #Value.backward_pow(v)
            return
        
        print("OP not suported")

    @staticmethod
    fn build_topo(self, mut visited: List[UnsafePointer[Value]], mut topo: List[UnsafePointer[Value]]):
        var is_visited = Bool(False)

        var size = Int(len(visited))

        print("Build topo")

        for i in range(size):
            if self == visited[i][]:
                is_visited = True
        
        #if not is_visited:
        if is_visited:
            return
            
        print("Entering visited")
        #visited.append(UnsafePointer.address_of(self))
        visited.append(UnsafePointer.address_of(self))
        print(len(visited))
        if self._prev1 != UnsafePointer[Value]():
            print("Entered _prev1 != UnsafePointer[Value]()")
            var _children1 = self._prev1[]
            print(_children1.data)
            if _children1._prev1 != UnsafePointer[Value]():
                Value.build_topo(_children1, visited, topo)
            #else:
            #    return

        if self._prev2 != UnsafePointer[Value]():
            print("Entered _prev2 != UnsafePointer[Value]()")
            var _children2 = self._prev2[]
            print(_children2.data)
            if _children2._prev2 != UnsafePointer[Value]():
                Value.build_topo(_children2, visited, topo)
            #else:
            #    return
        
        topo.append(UnsafePointer[Value].address_of(self))
        print(len(topo))

    fn backward(mut self):
        # Maybe this needs to be a pointer, we'll see
        var visited = List[UnsafePointer[Value]]()
        var topo = List[UnsafePointer[Value]]()

        print("previo a topo")
        print(len(topo))
        print(len(visited))


        # Maybe this fn can be defined here
        Value.build_topo(self, visited, topo)

        self.grad = Float32(1.0)

        for v in reversed(topo):
            print("for reversed")
            # Note the double [] needed, the first for the iterator and the second for the pointer
            print(v[][].data) 
            Value._backward(v[][])
    
    fn __print(self):
        print("data: ", self.data, "grad: ", self.grad)
    
        
fn main():
    var a = Value(data = 1.0)
    var b = Value(data = 2.0)
    var c = a + b
    
    # May god help us
    
    c.backward()

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

    if c._prev1 != UnsafePointer[Value]():
        c._prev1.destroy_pointee()
        c._prev1.free()
    
    if c._prev2 != UnsafePointer[Value]():
        c._prev2.destroy_pointee()
        c._prev2.free()