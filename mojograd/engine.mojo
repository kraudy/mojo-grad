"""  """

from collections import Optional, Dict, InlineList
from memory import UnsafePointer, memset_zero, ArcPointer
from memory import pointer

fn otro_fun ():
    print("Hello from fun")

@value
#struct Value(CollectionElement, Writable, Stringable): # These two gives error
struct Value(CollectionElement):
    var data: UnsafePointer[Float32]
    var grad : Float32
    var _backward : fn() -> None
    # Using a list of more than one pointer gives compiler error. So one pointer it is
    var _prev1 : List[ArcPointer[Self]]
    var _prev2 : List[ArcPointer[Self]]
    var _op : String

    fn __init__(out self, data: Float32,
                        #_children: List[Float32], # Maybe i can use pointers to Value object 
                        #_children2: List[UnsafePointer[Value]] , 
                        _backward : fn(),
                        _children1: List[ArcPointer[Self]] = List[ArcPointer[Self]](),
                        _children2: List[ArcPointer[Self]] = List[ArcPointer[Self]](),
                        _op : Optional[String] = None) # Note how optionl arguments must be at the end
                        :
        
        self.data = UnsafePointer[Scalar[DType.float32]].alloc(1)
        self.data.store(data)
        self.grad = 0

        self._backward = _backward

        if _children1:
            # Note that you can not directly call len on an optional argument
            # as you would on a normal argument
            #self._prev3 = ArcPointer[Value].alloc(len(_children2.value()))
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
    #var midata : List[Float64]
    #var midata : List[SIMD[Float64, 1]] = []
    #midata.append(1.0)
    #midata.append(2.0)
    # var midata : UnsafePointer[Scalar[DType.float32]]
    # midata.store(1.0)
    # midata.store(2.0)
    v = Value(data = 3.0, _backward = otro_fun)
    print(v.data.load())
    v._backward()
