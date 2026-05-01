from src.math.autodiff import Value

def test_simple_expression():
    # a = 2, b = -3, c = 10
    # d = a * b + c
    # d = -6 + 10 = 4
    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')
    c = Value(10.0, label='c')
    e = a * b; e.label = 'e'
    d = e + c; d.label = 'd'
    f = Value(-2.0, label='f')
    L = d * f; L.label = 'L'
    
    L.backward()
    
    # L = (a*b + c) * f
    # dL/df = d = 4.0
    # dL/dd = f = -2.0
    # dL/dc = dL/dd * dd/dc = -2.0 * 1 = -2.0
    # dL/de = dL/dd * dd/de = -2.0 * 1 = -2.0
    # dL/da = dL/de * de/da = -2.0 * b = -2.0 * -3.0 = 6.0
    # dL/db = dL/de * de/db = -2.0 * a = -2.0 * 2.0 = -4.0
    
    print(f"L data: {L.data} (expected -8.0)")
    print(f"a grad: {a.grad} (expected 6.0)")
    print(f"b grad: {b.grad} (expected -4.0)")
    print(f"c grad: {c.grad} (expected -2.0)")
    print(f"f grad: {f.grad} (expected 4.0)")

    assert L.data == -8.0
    assert a.grad == 6.0
    assert b.grad == -4.0
    assert c.grad == -2.0
    assert f.grad == 4.0
    print("Simple expression test passed!")

def test_activation():
    x = Value(0.5)
    y = x.tanh()
    y.backward()
    
    # dtanh(x)/dx = 1 - tanh(x)^2
    expected_grad = 1 - y.data**2
    print(f"x grad: {x.grad}, expected: {expected_grad}")
    assert abs(x.grad - expected_grad) < 1e-7
    print("Activation test passed!")

if __name__ == "__main__":
    test_simple_expression()
    test_activation()
