class _Any:
    def __init__(self,*a,**k): pass
    def __call__(self,*a,**k): return self
    def __getattr__(self,n): return _Any()
def __getattr__(name): return _Any
class _Cuda:
    @staticmethod
    def is_available(): return False
cuda=_Cuda(); float32=int; device=lambda *a,**k:"cpu"
no_grad=lambda *a,**k: __import__('contextlib').nullcontext()
