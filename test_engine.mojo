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
            print("Results =============")
            assert_equal(b.grad[], 0.0, "b grad should be 0.0")
            assert_equal(a.grad[], 1.0, "a grad should be 1.0")

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
        assert_equal(g.data[], 24.704082, "g should be almost 24.7041")

        try:
            g.backward()
            print("Results ===============================")
            assert_equal(b.grad[], 645.5773, "b grad should be almost 645.5773")
            assert_equal(a.grad[], 138.83382, "a grad should be almost 138.83382")
            print(repr(b))
            print(repr(a))
        finally:
            a.destroy()
            b.destroy() 
            c.destroy()
            d.destroy()

        a.destroy()
        b.destroy() 
        c.destroy()
        d.destroy()

    try:
        test1()
        test2()
    except e:
        print(e)
