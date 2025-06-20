from typing import Any


class LayoutBlock:
    def __init__(self, tracker_prefix=""):
        self.tracker_prefix = tracker_prefix

    def layout(self, tracker, root):
        pass

    def __call__(self, tracker, root) -> Any:
        self.layout(tracker, root)