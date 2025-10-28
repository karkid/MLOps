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
