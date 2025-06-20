from typing import Any


class Layout:
    def __call__(self, *args, **kwargs) -> Any:
        return self.layout(*args, **kwargs)

    def layout(self, *args, **kwargs):
        pass
