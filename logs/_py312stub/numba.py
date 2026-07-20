def jit(*a, **k):
    if a and callable(a[0]): return a[0]
    def deco(f): return f
    return deco
njit=jit
def prange(*a): return range(*a)
def __getattr__(n):
    def _x(*a,**k): raise RuntimeError("numba."+n)
    return _x
