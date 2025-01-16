"""  """

from collections import Optional, Dict, InlineList
from memory import UnsafePointer, memset_zero, ArcPointer
from memory import pointer

fn otro_fun ():
    print("Hello from fun")

@value # Understand what this does
#struct Value(CollectionElement, Writable, Stringable): # These two gives error
#struct Value(CollectionElement): # Understand what this does
struct Value():
    var data: UnsafePointer[Float32]
    var grad : Float32
    var _backward : fn() -> None
    # The compiler does not like an array of pointers of more than 1 element, so  1 element array it is
    var _prev1 : List[ArcPointer[Self]]
    var _prev2 : List[ArcPointer[Self]]
    var _op : String

    fn __init__(out self, data: Float32,
                        _backward : fn(),
                        _children1: List[ArcPointer[Self]] = List[ArcPointer[Self]](),
                        _children2: List[ArcPointer[Self]] = List[ArcPointer[Self]](),
                        _op : Optional[String] = None) # Note how optionl arguments must be at the end
                        :
        
        self.data = UnsafePointer[Scalar[DType.float32]].alloc(1)
        self.data.store(data)
        self.grad = 0

        self._backward = _backward

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
    


def main():
    var a = Value(data = 3.0, _backward = otro_fun)
    var b = Value(data = 1.0, _backward = otro_fun)
    print(a.data.load())
    a._backward()

    #var c = Value(data = 1.0, _backward = otro_fun, _children1 = a, _children2 = b)
    var c = Value(data = 1.0, _backward = otro_fun, 
                  _children1 = List[ArcPointer[Value]](a),
                  _children2 = List[ArcPointer[Value]](b))
    print(c.data.load())
    print(c._prev1.data)
    #print(c._prev1[0][].data.load()) # Seg fault error
    #print(c._prev1[0][].data)
    print(c._prev2.data)
    c._backward()
    #var c = Value(data = 1.0, _backward = otro_fun, _children1 = List[a])
