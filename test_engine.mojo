from mojograd.engine import Value 
from testing import assert_almost_equal, assert_true, assert_equal
from python import Python

fn main():
    fn test1() raises:
        var a = Value(data = 2.0)
        var b = Value(data = 3.0)
        var c = Float64(2.0)

        var d = b ** c
        assert_equal(d.data[], 9.0, "d should be 9.0")

        var e = a + c
        assert_equal(e.data[], 4.0, "e should be 4.0")
        
        e.backward()
        print("Results =============")
        assert_equal(b.grad[], 0.0, "b grad should be 0.0")
        assert_equal(a.grad[], 1.0, "a grad should be 1.0")

        print(repr(a))
        print(repr(b))

    
    fn test2() raises:
        a = Value(-4.0)
        b = Value(2.0)
        c = a + b 
        assert_equal(c.data[], -2.0, "c2 should be -2.0")

        d = a * b + b**3 
        assert_equal(d.data[], 0.0, "d should be 0.0")

        c += c + 1 
        assert_equal(c.data[], -3.0, "c should be -3.0")

        c += 1 + c + (-a) 
        assert_equal(c.data[], -1.0, "c should be -1.0")

        d += d * 2 + (b + a).relu() 
        assert_equal(d.data[], 0.0, "d should b 0.0")

        d += 3 * d + (b - a).relu() 
        assert_equal(d.data[], 6.0, "d should b 6.0")

        e = c - d 
        assert_equal(e.data[], -7.0, "e should be -7.0")

        f = e**2 
        assert_equal(f.data[], 49.0, "f should be 49.0")

        g = f / 2.0
        assert_equal(g.data[], 24.5, "g should be 24.5")

        g += 10.0 / f
        assert_equal(g.data[], 24.70408163265306, "g should be almost 24.70408163265306")

        g.backward()
        print("Results ===============================")
        assert_equal(b.grad[], 645.5772594752186, "b grad should be almost 645.5772594752186")
        assert_equal(a.grad[], 138.83381924198252, "a grad should be almost 138.83381924198252")
        print(repr(b))
        print(repr(a))

    
    fn test3() raises:
        """Karpathy sanity check"""
        var x = Value(data = -4.0)
        var z = 2 * x + 2 + x
        var q = z.relu() + z * x
        var h = (z * z).relu()
        var y = h + q + q * x
        assert_equal(y.data[], -20.0, "y data should be -20")
        
        y.backward()
        print("Results =============")
        assert_equal(x.grad[], 46.0, "x grad should be 46.0")
        print(repr(x))


    try:
        test1()
        test2()
        test3()
    except e:
        print(e)
