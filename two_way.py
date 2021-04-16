class TwoWay:
    def __init__(self):
        self.d = {}

    def add(self, k, v):
        self.d[k] = v
        self.d[v] = k

    def remove(self, k):
        self.d.pop(self.d.pop(k))

    def get(self, k):
        return self.d[k]
