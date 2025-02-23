# 🔥 mojo-grad
Simple implementation of Karpathy's micrograd (https://github.com/karpathy/micrograd/tree/master) in mojo and update on automata's previous pointer implementation (https://github.com/automata/mojograd/tree/main).

This is a scalar (1 element tensor) autodiff engine intended for learning purposes. The goal is to see how far this engine can go while keeping the scalar approach and implementing mojo's performance.
It could also be used as the building block for a tensor-based autodiff engine or an MLIR AST ops implementations.

There is still a lot to do.

### Example usage

Example showing Karpathy's and automata's test operations:

```mojo
from mojograd.engine import Value 

fn automatas_test():
    var a = Value(data = 2.0)
    var b = Value(data = 3.0)
    var c = Float64(2.0)
    var d = b ** c
    var e = a + c
    e.backward()

fn karpathys_test():
    a = Value(-4.0)
    b = Value(2.0)
    c = a + b 
    d = a * b + b**3 
    c += c + 1 
    c += 1 + c + (-a) 
    d += d * 2 + (b + a).relu() 
    d += 3 * d + (b - a).relu() 
    e = c - d 
    f = e**2 
    g = f / 2.0
    g += 10.0 / f
    g.backward()
```

### Training a neural net

Run the training test:

```bash
mojo train_mlp.mojo
```

![image](https://github.com/user-attachments/assets/3af2dc09-d238-437a-bcb8-94cdee42bd21)


### Running tests

To run the unit tests simply do:

```bash
mojo test_engine.mojo 
```

### Benchmarks

Really fast! See it for yourself in the forward and backward pass.

### License

MIT
