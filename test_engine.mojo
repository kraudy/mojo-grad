from mojograd.engine import Value 
from testing import assert_almost_equal, assert_true, assert_equal
from python import Python

fn main():
    fn test1() raises:
        var a = Value(data = 2.0)
        var b = Value(data = 3.0)
        var c = Float32(2.0)

        var d = b ** c
        assert_equal(d.data[], 9.0, "d should be 9.0")

        var e = a + c
        assert_equal(e.data[], 4.0, "e should be 4.0")
        
        try:
            e.backward()
            print(repr(a))
            print(repr(b))

        finally:
            a.destroy()
            b.destroy()
            d.destroy()
            e.destroy()

        a.destroy()
        b.destroy()
        d.destroy()
        e.destroy()
    
    fn test2() raises:
        a2 = Value(-4.0)
        b2 = Value(2.0)
        c2 = a2 + b2 
        assert_equal(c2.data[], -2.0, "c2 should be -2.0")

        d2 = a2 * b2 + b2**3 
        assert_equal(d2.data[], 0.0, "d2 should be 0.0")

        c2 += c2 + 1 
        assert_equal(c2.data[], -3.0, "c2 should be -3.0")

        c2 += 1 + c2 + (-a2) 
        assert_equal(c2.data[], -1.0, "c2 should be -1.0")
        
        d2 += d2 * 2 + (b2 + a2).relu() 
        assert_equal(d2.data[], 0.0, "d2 should be 0.0") # 0 because of relu

        d2 += 3 * d2 + (b2 - a2).relu() 
        assert_equal(d2.data[], 6.0, "d2 should be 6.0")

        e2 = c2 - d2 
        assert_equal(e2.data[], -7.0, "e2 should be -7.0")

        f2 = e2**2 
        assert_equal(f2.data[], 49.0, "f2 should be 49.0")

        g2 = f2 / 2.0
        assert_equal(g2.data[], 24.5, "g2 should be 24.5")

        g2 += 10.0 / f2
        assert_equal(g2.data[], 24.704082, "g2 should be almost 24.7041")
        # We got the same output as micrograd

        # Now comes the backward
        try:
            g2.backward()
            print("After backward")
            print(repr(a2))
            print(repr(b2))

        finally:
            a2.destroy()
            b2.destroy() 
            c2.destroy()
            d2.destroy()
            e2.destroy()
            f2.destroy()
            g2.destroy()

        a2.destroy()
        b2.destroy() 
        c2.destroy()
        d2.destroy()
        e2.destroy()
        f2.destroy()
        g2.destroy()

    fn test3() raises:
        a3 = Value(-4.0)
        b3 = Value(2.0)
        c3 = a3 + b3 
        assert_equal(c3.data[], -2.0, "c2 should be -2.0")

        d3 = a3 * b3 + b3**3 
        assert_equal(d3.data[], 0.0, "d3 should be 0.0")

        c3 += c3 + 1 
        assert_equal(c3.data[], -3.0, "c3 should be -3.0")

        c3 += 1 + c3 + (-a3) 
        assert_equal(c3.data[], -1.0, "c3 should be -1.0")

        d3 += d3 * 2 + (b3 + a3).relu() 
        assert_equal(d3.data[], 0.0, "d3 should b3 0.0")

        d3 += 3 * d3 + (b3 - a3).relu() 
        assert_equal(d3.data[], 6.0, "d3 should b3 6.0")

        e3 = c3 - d3 
        assert_equal(e3.data[], -7.0, "e3 should be -7.0")

        f3 = e3**2 
        assert_equal(f3.data[], 49.0, "f3 should be 49.0")

        g3 = f3 / 2.0
        assert_equal(g3.data[], 24.5, "g3 should be 24.5")

        g3 += 10.0 / f3
        assert_equal(g3.data[], 24.704082, "g3 should be almost 24.7041")

        try:
            g3.backward()
            print("Results")
            assert_equal(b3.grad[], 645.5773, "b3 grad should be almost 645.5773")
            assert_equal(a3.grad[], 138.83382, "a3 grad should be almost 138.83382")
            print(repr(b3))
            print(repr(a3))
        finally:
            a3.destroy()
            b3.destroy() 
            c3.destroy()
            d3.destroy()

        a3.destroy()
        b3.destroy() 
        c3.destroy()
        d3.destroy()

    try:
        #test1()
        #test2()
        test3()
    except e:
        print(e)
