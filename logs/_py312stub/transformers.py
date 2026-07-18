class _Any:
    def __init__(self,*a,**k): pass
    def __call__(self,*a,**k): return self
    def __getattr__(self,n): return _Any()
def __getattr__(n): return _Any
