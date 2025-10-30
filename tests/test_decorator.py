import pytest
from reml.utils.decorators import auto_repr, check_fitter


def test_auto_repr():
    @auto_repr
    class Foo:
        def __init__(self, x=1, y=2): self.x, self.y = x, y

    assert "x=1" in repr(Foo())
    assert "y=2" in repr(Foo())

def test_check_fitter():    
    class Foo:
        def __init__(self, x=1, y=2): self.x, self.y = x, y

        def fit(self):
            self.is_fitted = True
            return self
        
        @check_fitter
        def predict(self):
            return "predicted"
    foo = Foo()
    with pytest.raises(ValueError):
        foo.predict()

    foo.fit()
    assert foo.predict() == "predicted"
