class Node: 
    def __init__(self) -> None:
        self.neighbors = []
    
class Edge:
    def __init__(self, l, r, l_admit, r_admit) -> None:
        self.l = l
        self.r = r
        self.l_admit = l_admit
        self.r_admit = r_admit

class Source(Edge):
    def __init__(self, l, r, Vs) -> None:
        super().__init__(l, r, Vs, -Vs)

class Resistor(Edge):
    def __init__(self, l, r, R) -> None:
        super().__init__(l, r, R, R)

# is this right????
class ReLu(Edge):
    def __init__(self, l, r, dir) -> None:
        super().__init__(l, r, dir, not dir)

