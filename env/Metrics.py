from copy import deepcopy


class Metrics:
    def __init__(self, info) -> None:
        self.metrics = {}
        for i in info:
            self.metrics[i] = 0
        self.total_metrics = deepcopy(self.metrics)

    def add_scale(self, key, value):
        self.metrics[key] += value

    def minus_scale(self, key, value):
        self.metrics[key] -= value

    def get(self, key):
        return self.total_metrics.get(key)

    def get_metrics(self):
        return self.metrics

    def num_check(self, agent_num):
        num = 0
        for k in self.total_metrics.keys():
            self.total_metrics[k] += self.metrics[k]
            num += self.metrics[k]
            self.metrics[k] = 0
        assert num == agent_num, f"Metrics num error: {num}"
