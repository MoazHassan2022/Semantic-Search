from typing import List
import numpy as np

class Node:
    # create class Node to store the data and the next node
    # A node is a record of (id, vector, list of node pointers)
    def __init__(self, id: int, pq_code):
        self.id = id
        self.pq_code = pq_code
        self.neighbours = set()