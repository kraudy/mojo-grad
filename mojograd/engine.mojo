"""  """

from collections import Optional
from memory import UnsafePointer, memset_zero
from memory import pointer

fn otro_fun ():
    print("Hello from fun")

struct Value:
    # We still don't know what data is getting pass
    #var data : List[Float64]
    #var data : UnsafePointer[Scalar[DType.float32]]
    var data: UnsafePointer[Float32]
    #var _children : object
    var grad : Float32
    var _backward : fn() -> None
    #var _prev : List[Value]
    var _prev : List[UnsafePointer[Float32]]
    var _prev2 : List[UnsafePointer[UnsafePointer[Value]]]
    var _prev3 : List[UnsafePointer[Value]]
    var _op : String

    fn __init__(out self, data: Float32,
                        #_children: List[Float32], # Maybe i can use pointers to Value object 
                        #_children2: List[UnsafePointer[Value]] , 
                        _backward : fn(),
                        _children2: Optional[List[UnsafePointer[Value]]] = None,
                        _op : Optional[String] = None) # Note how optionl arguments must be at the end
                        :
        
        self.data = UnsafePointer[Scalar[DType.float32]].alloc(1)
        self.data.store(data)
        self.grad = 0

        self._backward = _backward

        self._prev = UnsafePointer[Scalar[DType.float32]].alloc(1)
        self._prev2 = UnsafePointer[UnsafePointer[Value]].alloc(1)

        if _children2 is not None:
            # Note that you can not directly call len on an optional argument
            # as you would on a normal argument
            self._prev3 = UnsafePointer[Value].alloc(len(_children2.value()))

            for i in range(len(_children2.value())):
                self._prev3[i] = _children2.value()[i]
        else:
            self._prev3 = UnsafePointer[Value].alloc(0)
        
        if _op is not None:
            self._op = _op.value()
        else:
            self._op = ''

      
      
def main():
    #var midata : List[Float64]
    #var midata : List[SIMD[Float64, 1]] = []
    #midata.append(1.0)
    #midata.append(2.0)
    # var midata : UnsafePointer[Scalar[DType.float32]]
    # midata.store(1.0)
    # midata.store(2.0)
    v = Value(3.0, otro_fun, None, None)
    print(v.data.load())
    v._backward()
