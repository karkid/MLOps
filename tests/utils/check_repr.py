import inspect

def check_repr(cls):
    """Generic helper to verify repr includes all __init__ parameters."""
    sig = inspect.signature(cls.__init__)
    kwargs = {
        k: v.default for k, v in sig.parameters.items()
        if k != "self" and v.default is not inspect._empty
    }

    model = cls(**kwargs)
    text = repr(model)

    for k, v in kwargs.items():
        assert f"{k}=" in text, f"{cls.__name__}: missing '{k}=' in repr"
        assert str(v) in text, f"{cls.__name__}: missing value '{v}' in repr"
