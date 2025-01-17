"""  """

from collections import Optional, List, Dict, InlineList
from memory import UnsafePointer, memset_zero, ArcPointer
from memory import pointer

from utils import Variant
# This can be useful for acceptinNone
alias ValueOrFloat = Variant[Value, Float32]
alias ValueOrNone  = Variant[Value, NoneType]

fn otro_fun ():
    print("Hello from fun")

@value # Understand what this does
#struct Value(CollectionElement, Writable, Stringable): # These two gives error
#struct Value(CollectionElement): # Understand what this does
struct Value():
    #TODO: This does not needs to be a pointer
    var data: UnsafePointer[Float32]
    var grad : Float32
    #var _backward : fn() -> None
    #var _backward : List[fn() -> None]
    var _backward : List[fn(mut self: Self) -> None]
    # The compiler does not like an array of pointers of more than 1 element, so  1 element array it is
    #var _prev : List[Self] # Maybe this can be implemented instead of ArcPointer?
    var _prev1 : List[ArcPointer[Self]]
    var _prev2 : List[ArcPointer[Self]]
    #var _prev  : Tuple[Self, Self]
    var _op : String

    fn __init__(out self, data: Float32,
                        # This needs to be optional also
                        # for when we only pass a number
                        #_backward : fn(),
                        _backward : List[fn(mut self: Self)] = List[fn(mut self: Self)](),
                        #_backward: fn() -> None = fn() -> None { },
                        _children1: List[ArcPointer[Self]] = List[ArcPointer[Self]](),
                        _children2: List[ArcPointer[Self]] = List[ArcPointer[Self]](),
                        _op : Optional[String] = None) # Note how optionl arguments must be at the end
                        :
        
        self.data = UnsafePointer[Scalar[DType.float32]].alloc(1)
        self.data.store(data)
        self.grad = 0

        if _backward:
            self._backward = _backward
        else:
            self._backward = List[fn(mut self: Self) -> None]()

        # if len(_chiledren) > 1
        if _children1:
            # There is no alloc por ArcPointer
            self._prev1 = List[ArcPointer[Value]](capacity=1)
        else:
            self._prev1 = List[ArcPointer[Value]](capacity=0)

        if _children2:
            self._prev2 = List[ArcPointer[Value]](capacity=1)
        else:
            self._prev2 = List[ArcPointer[Value]](capacity=0)
        
        if _op is not None:
            self._op = _op.value()
        else:
            self._op = ''
    
    fn __moveinit__(out self, owned existing: Self):
        self.data = existing.data
        self.grad = existing.grad
        self._backward = existing._backward
        self._prev1 = existing._prev1
        self._prev2 = existing._prev2
        self._op = existing._op
    
    #fn __add__(self, other: Self):
    fn __add__(self, other: ValueOrFloat) -> Value:
        # If the value passed is not Value
        # This isa can be useful to accept multiples types on a parameter
        var other2: Value

        if other.isa[Float32]():
            #var other2 = Value(data = other.take[Float32]())
            other2 = Value(data = other[Float32])
        else:
            other2 = other[Value]
         
        var out = Value(data = (self.data.load() + other2.data.load()), 
                                _children1 = List[ArcPointer[Value]](self),
                                _children2 = List[ArcPointer[Value]](other2))

        fn _backward(mut self: Value) -> None:
            self.grad += out.grad
            other2.grad += out.grad

        #out._backward = _backward
        out._backward.append(_backward)
        #out._backward[0] = _backward()
        #out._backward = List[_backward]()

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
