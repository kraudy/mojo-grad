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
    var _op : String

    fn __init__(inout self, data: Float32):
        
        #self.data = UnsafePointer[Scalar[DType.float32]].alloc(1)
        self.data = Float32(data)
        self.grad = Float32(0)


        #self._prev1 = UnsafePointer[Value]() 
        self._prev1 = UnsafePointer[Value]() 

        self._op = String('') 
    

    
            
fn main():
    var a = Value(data = 1.0)

    if a._prev1 != UnsafePointer[Value]():
        a._prev1.destroy_pointee()
        a._prev1.free()
