import inspect
from functools import wraps


def check_fitter(func):
    """Decorator to check if the model has been fitted before prediction or scoring."""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, "is_fitted") or not self.is_fitted:
            raise ValueError(
                f"Model {self.__class__.__name__} must be fitted "
                f"before calling `{func.__name__}`."
            )
        return func(self, *args, **kwargs)

    return wrapper


def auto_repr(cls):
    """Class decorator that auto-generates __repr__ from __init__ parameters."""
    sig = inspect.signature(cls.__init__)
    params = [p for p in sig.parameters if p != "self"]

    def __repr__(self):
        args_str = ", ".join(f"{name}={getattr(self, name, None)!r}" for name in params)
        return f"{self.__class__.__name__}({args_str})"

    cls.__repr__ = __repr__
    return cls
