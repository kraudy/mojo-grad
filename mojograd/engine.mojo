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
# Validates if AnyTypes behaves correlty
struct Value(AnyType):
    var data: Float32[]
    var grad : Float32[]

    var _prev1 : UnsafePointer[Value]
    var _prev2 : UnsafePointer[Value]
    var _op : String[]

    fn __init__(out self, data: Float32):
        
        #self.data = UnsafePointer[Scalar[DType.float32]].alloc(1)
        self.data = Float32[](data)
        self.grad = Float32[](0)


        self._prev1 = UnsafePointer[Value]() 
        self._prev2 = UnsafePointer[Value]() 

        self._op = String[]('') 
    

    fn __moveinit__(out self, owned existing: Self):
        self.data = existing.data
        self.grad = existing.grad
        self._prev1 = existing._prev1
        self._prev2 = existing._prev2
        self._op = existing._op
    
    fn __add__(self, other: Value) -> Value:
        var out = Value(data = (self.data + other.data))
        # Maybe i can just append this
        out._prev1.init_pointee_move(self)
        out._prev2.init_pointee_move(other)
        out._op = String[]('+')

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
        out._prev1.init_pointee_move(self)
        out._prev2.init_pointee_move(other) 
        out._op = String[]('**')

        return out
    
    fn __pow__(self, other: Float32) -> Value:
        var v = Value(other)
        return self.__pow__(v)

    fn backward_pow(mut v: Value):
        # This basically updates the grad
        var vv = v
        print("backward_pow")
        vv.__print()

        if v._prev1 != UnsafePointer[Value]() and v._prev2 != UnsafePointer[Value]() :
            var l = v._prev1[].grad
            var r = v._prev2[].grad

            v._prev1[0].grad = Float32[](l + (r * l ** (r - 1) * vv.grad) )
    
    fn backward_add(mut v: Value):
        var vv = v
        print("backward_add")
        vv.__print()

        if v._prev1 != UnsafePointer[Value]():
            var _children1 = v._prev1[]
            print("_children1.grad = ", _children1.grad, "vv.grad = ",vv.grad)
            #TODO: Valide which works
            #_children1.grad = ArcPointer[Float32](_children1.grad[] + vv.grad[])
            v._prev1[].grad = Float32[](_children1.grad + vv.grad)
            #v.grad = _children1.grad
        
        if v._prev2 != UnsafePointer[Value]():
            var _children2 = v._prev2[]
            print("_children2.grad = ", _children2.grad, "vv.grad = ",vv.grad)
            v._prev2[].grad = Float32[](_children2.grad + vv.grad)
    
    fn _backward(mut v: Value):
        var op = String[](v._op)
        print("op")
        print(op)

        print("_backward")
        v.__print()

        if op == '+':
            print("Option found")
            Value.backward_add(v)
        if op == '**':
            print("Option **")
            Value.backward_pow(v)
        else:
            print("OP not suported")
    
    fn build_topo(self, mut visited: List[ArcPointer[Value]], mut topo: List[ArcPointer[Value]]):
        var is_visited = Bool[](False)
        var size = Int[](len(visited))

        print("Build topo")

        for i in range(size):
            if self == visited[i][]:
                is_visited = True
        
        if not is_visited:
            #visited.append(ArcPointer[Value](self))
            print("Entering visited")
            visited.append(self)
            print(len(visited))
            if self._prev1 != UnsafePointer[Value]():
                print("Entered _prev1 != UnsafePointer[Value]()")
                var _children1 = self._prev1[]
                if _children1._prev1 != UnsafePointer[Value]():
                    Value.build_topo(_children1, visited, topo)
                #else:
                #    return

            if self._prev2 != UnsafePointer[Value]():
                print("Entered _prev2 != UnsafePointer[Value]()")
                var _children2 = self._prev2[]
                if _children2._prev2 != UnsafePointer[Value]():
                    Value.build_topo(_children2, visited, topo)
                #else:
                #    return
            
            topo.append(self)
            print(len(topo))

    
    fn backward(mut self):
        # Maybe this needs to be a pointer, we'll see
        var visited = List[ArcPointer[Value]]()
        var topo = List[ArcPointer[Value]]()

        # Maybe this fn can be defined here
        Value.build_topo(self, visited, topo)

        self.grad = Float32(1.0)
        #var reversed = List[ArcPointer[Value]](reversed(topo))
        var reversed_topo = reversed(topo) # this returns an iterator

        # Lets pray this thing works
        #for ref in reversed_topo:
        for i in range(len(topo), -1, -1):
            #var v = ref[]
            print("i: ", i)
            # If there is no elements, this gives error, maybe use try catch
            var v = topo[i][]
            print("Previous _backward: ")
            v.__print()
            Value._backward(v)
    
    fn __print(self):
        print("data: ", self.data, "grad: ", self.grad)
    
            
def main():
    var a = Value(data = 1.0)
    var b = Value(data = 2.0)
    var c = a + b
    try:
        a.__print()
        #print(a.data[])

        #var c = Value(data = 1.0, _backward = otro_fun, _children1 = a, _children2 = b)
        # Maybe i can add another function to the class to do this thing
        print(c.data)
        var d = c + Float32(3.0)
        print(d.data)
        #print(c._prev1.data[])
        c._prev1[].__print()

        # May god help us
        c.backward()
        print("Resultado ====")
        c.__print()
        c._prev1[].__print()
        c._prev2[].__print()
    finally:
        a._prev1.destroy_pointee()
        a._prev1.free()
        a._prev2.destroy_pointee()
        a._prev2.free()

        b._prev1.destroy_pointee()
        b._prev1.free()
        b._prev2.destroy_pointee()
        b._prev2.free()

        c._prev1.destroy_pointee()
        c._prev1.free()
        c._prev2.destroy_pointee()
        c._prev2.free()
