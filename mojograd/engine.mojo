"""  """

from collections import Optional, List, Dict, InlineList
from memory import UnsafePointer, memset_zero, ArcPointer
from memory import pointer

from utils import Variant
# This can be useful for acceptinNone
alias ValueOrFloat = Variant[Value, Float32]
alias ValueOrNone  = Variant[Value, NoneType]
alias FunOrNone  = Variant[fn(mut Value), NoneType]
alias SelfOrNone  = Variant[Value, NoneType]

fn otro_fun ():
    print("Hello from fun")


@value # Understand what this does
#struct Value(CollectionElement, Writable, Stringable): # These two gives error
#struct Value(CollectionElement): # Understand what this does
struct Value():
    #TODO: This does not needs to be a pointer
    var data: ArcPointer[Float32]
    var grad : ArcPointer[Float32]
    #var _backward : fn() -> None
    #var _backward : List[fn() -> None]
    #var _backward : List[fn(mut self: Self) -> None]
    #var _backward : fn(mut self: Self) -> None
    #var _backward : ArcPointer[fn()]
    #var _backward : ArcPointer[fn()]
    #var _backward : ArcPointer[Int]

    var _prev1 : ValueOrNone
    var _prev2 : ValueOrNone
    #var _prev  : Tuple[Self, Self]
    var _op : ArcPointer[String]

    fn __init__(out self, data: Float32):
        
        #self.data = UnsafePointer[Scalar[DType.float32]].alloc(1)
        self.data = ArcPointer[Float32](data)
        self.grad = ArcPointer[Float32](0)


        self._prev1 = None
        self._prev2 = None

        self._op = ArcPointer[String]('') 
    

    fn __moveinit__(out self, owned existing: Self):
        self.data = existing.data
        self.grad = existing.grad
        self._prev1 = existing._prev1
        self._prev2 = existing._prev2
        self._op = existing._op
    
    fn __add__(self, other: Value) -> Value:
        var out = Value(data = (self.data[] + other.data[]))
        out._prev1 = self
        out._prev2 = other

        return out

    fn __add__(self, other: Float32) -> Value:
        # If the value passed is not Value
        # This isa can be useful to accept multiples types on a paramete

        return out
            
    


def main():
    var d = Value(data = 3.0)
    var a = Value(data = 3.0)
    var b = Value(data = 1.0)
    print(a.data.load())
    #a._backward()
    # Get first element
    #a._backward[0]()


    #var c = Value(data = 1.0, _backward = otro_fun, _children1 = a, _children2 = b)
    # Maybe i can add another function to the class to do this thing
    var c = Value(data = 1.0, 
                  _children1 = List[ArcPointer[Value]](a),
                  _children2 = List[ArcPointer[Value]](b))
    print(c.data.load())
    print(c._prev1.data)
    #print(c._prev1[0][].data.load()) # Seg fault error
    #print(c._prev1[0][].data)
    print(c._prev2.data)
    #c._backward[0]()
    #var c = Value(data = 1.0, _backward = otro_fun, _children1 = List[a])
