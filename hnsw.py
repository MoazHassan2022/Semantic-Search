import numpy as np
import random
from typing import Dict, List, Annotated, Optional
from node import Node
import linecache as lc
from helpers import *

class HNSW:

    def __init__(self, M: int, num_layers: int,file_path = "saved_db.csv", new_db = True, records_per_file: int = 100) -> None:
        self.M = M
        self.m_L = 1/np.log(M)
        self.num_layers = num_layers
        self.layers = [{} for _ in range(num_layers)]
        self.file_path = file_path
        self.records_per_file = records_per_file
        self.layer_sizes = []
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

    def insert(self, node: Node):
        probabilities = [np.exp(-i / self.m_L) * (1 - np.exp(-1 / self.m_L)) for i in range(len(self.layers))]
        for i in range(len(self.layers)):  # start from the bottom layer
            self.layers[i][node.id] = Node(node.id, node.vector)
            if random.random() - sum(probabilities[:i]) < probabilities[i] :
                break

    def connect_nodes(self):
        for i,layer in enumerate(self.layers):
            nodes = list(layer.values())
            with open(f'layer_{i}', "w") as fout:
                for node in layer.values():
                    # we need to sort the nodes by their distance to the current node
                    nodes.sort(key=lambda n: self._calculate_distance(n, node), reverse=True)
                    for neighbor in nodes[1:self.M+1]:
                        self._create_connection(node, neighbor)
                    row_str = f"{node.id}," +",".join([str(n.id) for n in node.neighbours])
                    fout.write(f"{row_str}\n")

    def insert_records(self, rows: List[Dict[int, Annotated[List[float], 70]]]):
        # insert records to the file such that each file has self.records_per_file row of records and name the file as data_0, data_1, etc.
        for i in range(len(rows) // self.records_per_file):
            with open(f'data/data_{i}', "w") as fout:
                for row in rows[i*self.records_per_file:(i+1)*self.records_per_file]:
                    id, embed = row["id"], row["embed"]
                    row_str = f"{id}," + ",".join([str(e) for e in embed])
                    fout.write(f"{row_str}\n")
        with open(self.file_path, "a+") as fout:
            for row in rows:
                id, embed = row["id"], row["embed"]
                node = Node(id, embed)
                self.insert(node)
        self.layer_sizes = [len(layer) for layer in self.layers]
        self.connect_nodes()
        # delete the layers from the memory
        self.layers = [{} for _ in range(self.num_layers)]

    def retrive(self, query: Annotated[List[float], 70], top_k = 1):
        '''
        This method returns the nearest neighbour of the query node.
        1. select the first line in the top layer as entry point 
        2. search over these neighbours and get the nearest neighbour to the query node by :
            a. get the data of the neighbours from the files of the data
            b. get the id of nearest neighbour to the query node by calculating the distance between the query node and the neighbours and select the minimum
            c. get the record of the nearest neighbour from the file of the data using binary search and linecache module
            d. repeat step a and b until reaching the local minimum in this layer
            e. after that go to the next layer
        3. down the layers, search over the neighbours of the nearest neighbour and get the nearest neighbour to the query node until reaching the bottom layer
        4. return the nearest neighbour to the query node
        note : we are using linecache module to read lines from the files 
        '''
        query = query.T
        # select the first line in the top layer as entry point
        # read the top layer file and get the first line using linecache module
        # get the largest layer number that contain data i.e. the self.layer_sizes[layer_number] > 0
        curr_layer = 0
        for i in range(len(self.layer_sizes)-1, 0, -1):
            if self.layer_sizes[i] > 0:
                curr_layer = i
                break
        curr_node_record = lc.getline(f'layer_{curr_layer}', 1)
        # get the id of the entry point
        curr_id = int(curr_node_record.split(',')[0])
        # get the record of the entry point from the file of the data
        curr_data = lc.getline(f'data/data_{curr_id // self.records_per_file}', curr_id % self.records_per_file + 1)
        while True:
            curr_node = Node(curr_id, [float(e) for e in curr_data.split(',')[1:]])
            # get the data of the neighbours of the entry point from the files of the data
            curr_neighbours = [lc.getline(f'data/data_{int(n) // self.records_per_file}', int(n) % self.records_per_file + 1) for n in curr_node_record.split(',')[1:]]
            # Split the neighbor ID and vector
            split_neighbour = [n.split(',') for n in curr_neighbours]
            # Create Node instances for each neighbor
            neighbour_nodes = [Node(int(n[0]), [float(e) for e in n[1:]]) for n in split_neighbour]
            # Calculate the distance from each neighbor to the query
            distances = [self._calculate_distance(n, Node(-1, query)) for n in neighbour_nodes]
            distances.append(self._calculate_distance(curr_node, Node(-1, query)))
            # Find the ID of the neighbor with the smallest distance
            min_index = distances.index(min(distances))
            # Get the ID of the nearest neighbor
            if(min_index == len(distances)-1):
                # we reach the local minimum in this layer
                curr_layer -= 1
                if curr_layer < 0:
                    return [curr_id]
                # get the record of the current node from the next layer
                curr_node_record = file_binary_search(f'layer_{curr_layer}', curr_id, self.layer_sizes[curr_layer]) if curr_layer != 0 else lc.getline(f'layer_0', curr_id)
                continue
            # get the record of the nearest neighbour from the file of the data using binary search and linecache module
            curr_node_record = file_binary_search(f'layer_{curr_layer}', neighbour_nodes[min_index].id, self.layer_sizes[curr_layer])
            curr_id = int(curr_node_record.split(',')[0])
            curr_data = curr_neighbours[min_index]