"""  """

from collections import Optional, List, Dict, InlineList
from memory import UnsafePointer, memset_zero, ArcPointer
from memory import pointer

from utils import Variant
# This can be useful for acceptinNone
alias ValueOrFloat = Variant[Value, Float32]
alias ValueOrNone  = Variant[Value, NoneType]
alias SelfOrNone  = Variant[Value, NoneType]

fn otro_fun ():
    print("Hello from fun")


@value # Understand what this does
#struct Value(CollectionElement, Writable, Stringable): # These two gives error
#struct Value(CollectionElement): # Understand what this does
struct Value():
    var data: ArcPointer[Float32]
    var grad : ArcPointer[Float32]

    var _prev1 : List[ArcPointer[Value]]
    var _prev2 : List[ArcPointer[Value]]
    #var _prev  : Tuple[Self, Self]
    var _op : ArcPointer[String]

    fn __init__(out self, data: Float32):
        
        #self.data = UnsafePointer[Scalar[DType.float32]].alloc(1)
        self.data = ArcPointer[Float32](data)
        self.grad = ArcPointer[Float32](0)


        self._prev1 = List[ArcPointer[Value]]() 
        self._prev2 = List[ArcPointer[Value]]() 

        self._op = ArcPointer[String]('') 
    

    fn __moveinit__(out self, owned existing: Self):
        self.data = existing.data
        self.grad = existing.grad
        self._prev1 = existing._prev1
        self._prev2 = existing._prev2
        self._op = existing._op
    
    fn __add__(self, other: Value) -> Value:
        var out = Value(data = (self.data[] + other.data[]))
        out._prev1 = List[ArcPointer[Value]](self) 
        out._prev2 = List[ArcPointer[Value]](other) 
        out._op = ArcPointer[String]('+')

        return out

    fn __add__(self, other: Float32) -> Value:
        # If the value passed is not Value
        # This isa can be useful to accept multiples types on a paramete
        var v = Value(other)
        # We are only making the conversion and reusing the value logic
        return self.__add__(v)
    
    fn __eq__(self, other: Self) -> Bool:
        #return self is other
        return self.data.__is__(other.data) and
               self.grad.__is__(other.data) and
               self._op.__is__(other._op)
    
    fn build_topo(self, mut visited: List[ArcPointer[Value]], topo: List[ArcPointer[Value]]):
        var is_visited = Bool[](False)
        var size = Int[](len(visited))

        for i in range(size):
            if self == visited[i][]:
                is_visited = True
        
        if not is_visited:
              #visited.append(ArcPointer[Value](self))
              visited.append(self)
              if len(self._prev1) == 1:
                  var _children1 = self._prev1
                  # Value.build_topo(_children1, visited, topo)

    
    fn backward(self):
        var visited = List[ArcPointer[Value]]()
        var topo = List[ArcPointer[Value]]()

        # Maybe this fn can be defined here
        Value.build_topo(self, visited, topo)
    
    fn __print(self):
        print(self.data[])
    
            
    


def main():
    pass
    var a = Value(data = 1.0)
    var b = Value(data = 2.0)
    a.__print()
    #print(a.data[])

    #var c = Value(data = 1.0, _backward = otro_fun, _children1 = a, _children2 = b)
    # Maybe i can add another function to the class to do this thing
    var c = a + b
    print(c.data[])
    var d = c + Float32(3.0)
    print(d.data[])
    #print(c._prev1.data[])
    c._prev1[0][].__print()

    # May god help me
    c.backward()
