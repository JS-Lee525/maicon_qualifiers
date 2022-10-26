

class ComposeList:
    def __init__(self, list_compose):
        self.list_compose = list_compose
    
    def __call__(self, **kwargs):
        for x in self.list_compose:
            kwargs = x(**kwargs)
        return kwargs