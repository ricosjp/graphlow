from collections.abc import Callable


class _HashedSeq(list):
    """This implementation is identical to _HashedSeq from functools"""

    __slots__ = "hashvalue"

    def __init__(self, tup: tuple, hash: Callable = hash):
        self[:] = tup
        self.hashvalue = hash(tup)

    def __hash__(self):
        return self.hashvalue


def make_key(
    method_name: str, args: tuple, kwds: dict, kwd_mark: tuple = (object(),)
) -> _HashedSeq:
    key = (method_name,)
    key += args
    if kwds:
        key += kwd_mark
        for item in kwds.items():
            key += item
    return _HashedSeq(key)
