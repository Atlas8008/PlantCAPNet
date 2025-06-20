class ComponentValueTracker:
    def __init__(self) -> None:
        self._tracker = {}

    def __getitem__(self, index):
        return self._tracker[index]

    def __setitem__(self, index, item):
        self._tracker[index] = item

    def dump_list(self):
        return list(self._tracker.values())

    def __len__(self):
        return len(self._tracker)

    @staticmethod
    def tracked_list_to_dict(tracker, list):
        assert len(list) == len(tracker), f"The number of items in the component tracker and the list do not match. ({len(list), len(tracker)})"

        return dict(zip(tracker._tracker.keys(), list))