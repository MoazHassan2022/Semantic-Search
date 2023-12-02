import numpy as np
import random
from typing import Dict, List, Annotated, Optional
from node import Node
import linecache as lc
from helpers import *

class HNSW:

    def __init__(self, M: int, num_layers: int, records_per_file: int = 100) -> None:
        self.M = M
        self.m_L = 1/np.log(M)
        self.num_layers = num_layers
        self.layers = [{} for _ in range(num_layers)]
        self.records_per_file = records_per_file
        self.layer_sizes = []

    # Calculate the distance between two nodes by calculating the cosine similarity between their vectors.
    def calculate_similarity(self, node1: Node, node2: Node) -> float:
        dot_product = np.dot(node1.vector, node2.vector)
        norm_vec1 = np.linalg.norm(node1.vector)
        norm_vec2 = np.linalg.norm(node2.vector)
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
        return cosine_similarity

    # Find the nearest node in a layer
    def _find_nearest(self, layer: Dict[int, Node], node: Node) -> Node:
        nearest_node = max(layer.values(), key=lambda n: self.calculate_similarity(n, node))
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
            with open(f'layers/layer_{i}', "w") as fout:
                for node in layer.values():
                    # we need to sort the nodes by their distance to the current node
                    nodes.sort(key=lambda n: self.calculate_similarity(n, node), reverse=True)
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
            b. get the id of nearest neighbour to the query node by calculating the distance between the query node and the neighbours and select the maximum
            c. get the record of the nearest neighbour from the file of the data using binary search and linecache module
            d. repeat step a and b until reaching the local maximum in this layer
            e. after that go to the next layer
        3. down the layers, search over the neighbours of the nearest neighbour and get the nearest neighbour to the query node until reaching the bottom layer in this step consider if we will down with one node or more than one based on the top_k and M 
        4. return the nearest neighbour to the query node
        note : we are using linecache module to read lines from the files 
        '''
        query = query.T
        
        # determine the number of nodes we will down with in each layer
        num_nodes_down = np.ceil(top_k/self.M).astype(int)
        
        # select the first line in the top layer as entry point
        # read the top layer file and get the first line using linecache module
        # get the largest layer number that contain data i.e. the self.layer_sizes[layer_number] > 0
        curr_layer = 0
        for i in range(len(self.layer_sizes)-1, 0, -1):
            if self.layer_sizes[i] > 1:
                curr_layer = i
                break
            
        # generate unique random initial points in the top layer between 0 and min(self.layer_sizes[curr_layer], num_nodes_down)
        random_lines = set(random.sample(range(0, self.layer_sizes[curr_layer] - 1), min(num_nodes_down,self.layer_sizes[curr_layer])))
        
        # if the random_lines has less lines than min(num_nodes_down,self.layer_sizes[curr_layer]) we need to add more lines to it
        while len(random_lines) < min(num_nodes_down,self.layer_sizes[curr_layer]):
            random_lines.add(random.randint(0, self.layer_sizes[curr_layer] - 1))
        
        print("random_lines len: ", len(random_lines))
        
        # get the records of the entry points from the top layer file
        curr_node_records = [lc.getline(f'layers/layer_{curr_layer}', random_line + 1) for random_line in random_lines]
        
        # get the ids of the entry points
        curr_ids = [int(curr_node_record.split(',')[0]) for curr_node_record in curr_node_records]
        
        # get the records of the entry points from the file of the data
        curr_data = [lc.getline(f'data/data_{curr_id // self.records_per_file}', curr_id % self.records_per_file + 1) for curr_id in curr_ids]
        
        # convert to nodes
        curr_nodes = [Node(curr_node_id, [float(e) for e in curr_node_data.split(',')[1:]]) for curr_node_id, curr_node_data in zip(curr_ids, curr_data)]
        while True:
            # get the data of all neighbours of all curr nodes from the files of the data
            # make sure that the neighbours are unique
            curr_neighbours_nodes = {}
            for curr_node_record in curr_node_records:
                if curr_node_record == '' or curr_node_record == '\n' or not curr_node_record:
                    continue
                # remove '\n' from the record, get the ids of the neigbours
                curr_node_record = curr_node_record[:-1]
                if curr_node_record == '':
                    continue
                record_ids = curr_node_record.split(',')[1:]
                for n in record_ids:
                    n = int(n)
                    if not curr_neighbours_nodes.get(n):
                        data_line_splitted = lc.getline(f'data/data_{n // self.records_per_file}', n % self.records_per_file + 1).split(',')
                        curr_neighbours_nodes[n] = Node(n, [float(e) for e in data_line_splitted[1:]])
                        
            # add current data to the neighbours set
            for curr_node in curr_nodes:
                if not curr_neighbours_nodes.get(curr_node.id):
                    curr_neighbours_nodes[curr_node.id] = curr_node
        
            # Calculate the distance from each neighbor to the query
            query_node = Node(-1, query)
            # calculate the similarity between the query node and the neighbours
            neighbours_similarity = [(neighbour_id, self.calculate_similarity(curr_neighbours_nodes[neighbour_id], query_node)) for neighbour_id in curr_neighbours_nodes]
        
            # Find the IDs of the num_nodes_down neighbours with the max distances
            max_similarity = sorted(neighbours_similarity, key=lambda neighbour_similarity: neighbour_similarity[1])

            # get new ids of the neighbours
            max_ids = [neighbour_similarity[0] for neighbour_similarity in max_similarity[-num_nodes_down:]]
            
            # Get the IDs of the nearest neighbors
            if(set(max_ids) == set(curr_ids)):
                # we reach the local maximum in this layer
                curr_layer -= 1
                if curr_layer < 0:
                    # largen the nodes to be returned to the size of top_k
                    # we want to return the top_k results from max_ids
                    # we need to sort the nodes by their distance to the query node
                    if len(max_similarity) >= top_k:
                        return [i[0] for i in max_similarity[-top_k:]]
                    
                    max_similarity_ids = [i[0] for i in max_similarity]
                    difference = top_k - len(max_similarity_ids) # number of nodes to be added to the result
                    
                    # we need to add more nodes to the result
                    max_similarity_index = 0
                    curr_max_similarity = max_similarity[max_similarity_index]
                    while difference > 0:
                        # get the record of this node from layer_0 file
                        curr_max_similarity_record = lc.getline(f'layers/layer_0', curr_max_similarity[0] + 1)
                        
                        # remove '\n' from the record
                        curr_max_similarity_record = curr_max_similarity_record[:-1].split(',')[1:]
                        
                        # remove ids from the record that already exist in the result
                        curr_max_similarity_record = [int(i) for i in curr_max_similarity_record if int(i) not in max_similarity_ids]
                        number_of_neighbours = len(curr_max_similarity_record)
                        new_difference = difference - number_of_neighbours
                        if new_difference < 0:
                            # we need to add only the first difference neighbours
                            curr_max_similarity_record = curr_max_similarity_record[:difference]
                            difference = 0
                        max_similarity_ids.extend(curr_max_similarity_record)
                        
                        # move to the next node    
                        max_similarity_index += 1
                        curr_max_similarity = max_similarity[max_similarity_index]
        
                # get the records of the current nodes from the next layer
                curr_node_records = [file_binary_search(f'layers/layer_{curr_layer}', curr_id, self.layer_sizes[curr_layer]) if curr_layer != 0 else lc.getline(f'layers/layer_0', curr_id + 1) for curr_id in curr_ids]
                continue
        
            # get the records of the nearest neighbours from the file of the data using binary search and linecache module
            curr_node_records = [file_binary_search(f'layers/layer_{curr_layer}', max_id, self.layer_sizes[curr_layer]) for max_id in max_ids]
            curr_ids = max_ids
            curr_nodes = [curr_neighbours_nodes[max_id] for max_id in max_ids]
            