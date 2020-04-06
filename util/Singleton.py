class Singleton:
    _instance = None
    @classmethod
    def getInstance(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance=cls(*args,**kwargs)
        return cls._instance
