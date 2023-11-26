from typing import List

class Node:
    # create class Node to store the data and the next node
    # A node is a record of (id, vector, list of node pointers)
    def __init__(self, id: int, vector: List[float], neighbours = set()):
        self.id = id
        self.vector = vector
        self.neighbours = neighbours