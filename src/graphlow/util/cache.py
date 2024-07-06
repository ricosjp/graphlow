from collections.abc import Callable


def cache(method: Callable):
    """Cache decorator for an object's method with no arguments.
    Use this instead of functool.cache to avoid memory leakage.
    """
    def _wrapper(obj):
        variable_name: str = method.__name__.replace('compute_', '')

        if hasattr(obj, variable_name):
            return getattr(obj, variable_name)
        else:
            ret = method(obj)
            setattr(obj, variable_name, ret)
            return ret

    return _wrapper
