import numpy as np
import random
from typing import Dict, List, Annotated, Optional
from node import Node

class HNSW:

    def __init__(self, M: int, num_layers: int,file_path = "saved_db.csv", new_db = True) -> None:
        self.M = M
        self.m_L = 1/np.log(M)
        self.num_layers = num_layers
        self.layers = [{} for _ in range(num_layers)]
        self.file_path = file_path
        if new_db:
            # just open new file to delete the old one
            with open(self.file_path, "w") as fout:
                # if you need to add any head to the file
                pass

    # Calculate the distance between two nodes by calculating the cosine similarity between their vectors.
    def _calculate_distance(self, node1: Node, node2: Node) -> float:
        dot_product = np.dot(node1.vector, node2.vector)
        norm_vec1 = np.linalg.norm(node1.vector)
        norm_vec2 = np.linalg.norm(node2.vector)
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
        return cosine_similarity

    # Find the nearest node in a layer
    def _find_nearest(self, layer: Dict[int, Node], node: Node) -> Node:
        nearest_node = min(layer.values(), key=lambda n: self._calculate_distance(n, node))
        return nearest_node

    # This method creates a connection between two nodes by adding each node to the other's list of neighbours.
    def _create_connection(self, node1: Node, node2: Node):
        node1.neighbours.add(node2)
        # node2.neighbours.add(node1)

    def insert(self, node: Node):
        probabilities = [np.exp(-i / self.m_L) * (1 - np.exp(-1 / self.m_L)) for i in range(len(self.layers))]
        for i in range(len(self.layers)):  # start from the bottom layer
            self.layers[i][node.id] = node
            if random.random() - sum(probabilities[:i]) < probabilities[i] :
                break

    def connect_nodes(self):
        for i,layer in enumerate(self.layers):
            nodes = list(layer.values())
            with open(f'layer_{i}', "w") as fout:
                for node in layer.values():
                    # we need to sort the nodes by their distance to the current node
                    nodes.sort(key=lambda n: self._calculate_distance(n, node))
                    for neighbor in nodes[1:self.M+1]:
                        self._create_connection(node, neighbor)
                    row_str = f"{node.id}," +",".join([str(n.id) for n in node.neighbours])
                    fout.write(f"{row_str}\n")
    
    def insert_records(self, rows: List[Dict[int, Annotated[List[float], 70]]]):
        with open(self.file_path, "a+") as fout:
            for row in rows:
                id, embed = row["id"], row["embed"]
                node = Node(id, embed)
                self.insert(node)
        self.connect_nodes()
