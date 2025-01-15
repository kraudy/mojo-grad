"""  """

struct Value:
    # We still don't know what data is getting pass
    var data : object
    var _children : object
    var grad : Float64
    var _backward : fn()

    fn __init__(out self, data: object, _children: object):
        self.data = data
        self.grad = 0

        #self._backward = fn()