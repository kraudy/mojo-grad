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
    

    
            
fn main():
    var a = Value(data = 1.0)
    var b = Value(data = 2.0)
    var c = a + b
    
    # May god help us
    #c.backward()

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