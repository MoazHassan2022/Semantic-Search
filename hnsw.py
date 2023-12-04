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
        self.repeat_retrives_num = 5

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

    def retrive_n(self, query: Annotated[List[float], 70], top_n = 1):
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
        num_nodes_down = np.ceil(top_n/self.M).astype(int)
        
        # select the first line in the top layer as entry point
        # read the top layer file and get the first line using linecache module
        # get the largest layer number that contain data i.e. the self.layer_sizes[layer_number] > 0
        curr_layer = 0
        for i in range(len(self.layer_sizes)-1, 0, -1):
            if self.layer_sizes[i] > 1:
                curr_layer = i
                break
            
        # get the initial lines of the entry points from the top layer file based on the similarity between the query node and the all points of the top layer
        top_layer_lines = [lc.getline(f'layers/layer_{curr_layer}', i + 1) for i in range(self.layer_sizes[curr_layer])]
        top_layer_ids = [int(top_layer_line.split(',')[0]) for top_layer_line in top_layer_lines]
        top_layer_data = [lc.getline(f'data/data_{top_layer_id // self.records_per_file}', top_layer_id % self.records_per_file + 1) for top_layer_id in top_layer_ids]
        top_layer_nodes = [Node(top_layer_id, [float(e) for e in top_layer_data.split(',')[1:]]) for top_layer_id, top_layer_data in zip(top_layer_ids, top_layer_data)]
        sorted(top_layer_nodes,key=lambda n: self.calculate_similarity(n, Node(-1, query)), reverse=True)
        curr_nodes = top_layer_nodes[0 : num_nodes_down]
        # get the records of the entry points from the top layer file
        curr_node_records = top_layer_lines[0 : num_nodes_down]
        
        # get the ids of the entry points
        curr_ids = top_layer_ids[0 : num_nodes_down]
        
        # convert to nodes
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
            
            # TODO: sort nodes by similarity (put larger similarity first), return these nodes 
            
            # calculate the similarity between the query node and the neighbours
            curr_neighbours_nodes_list = [curr_neighbours_nodes[neighbour_id] for neighbour_id in curr_neighbours_nodes]
        
            # sort the neighbours by their similarity to the query node
            max_similarity = sorted(curr_neighbours_nodes_list, key=lambda curr_neighbours_node: self.calculate_similarity(curr_neighbours_node, query_node), reverse=True)

            # get new ids of the neighbours
            max_ids = [neighbour_similarity.id for neighbour_similarity in max_similarity[0 : num_nodes_down]]
            
            # Get the IDs of the nearest neighbors
            if(set(max_ids) == set(curr_ids)):
                # we reach the local maximum in this layer
                curr_layer -= 1
                if curr_layer < 0:
                    return max_similarity[0: top_n]
        
                # get the records of the current nodes from the next layer
                curr_node_records = [file_binary_search(f'layers/layer_{curr_layer}', curr_id, self.layer_sizes[curr_layer]) if curr_layer != 0 else lc.getline(f'layers/layer_0', curr_id + 1) for curr_id in curr_ids]
                continue
        
            # get the records of the nearest neighbours from the file of the data using binary search and linecache module
            curr_node_records = [file_binary_search(f'layers/layer_{curr_layer}', max_id, self.layer_sizes[curr_layer]) for max_id in max_ids]
            curr_ids = max_ids
            curr_nodes = [curr_neighbours_nodes[max_id] for max_id in max_ids]
    def retrive(self, query: Annotated[List[float], 70], top_k = 1):
        # retrieve only connections length + 1 nodes (list of Node sorted from larger similarity to smaller similarity)
        # why self.M + 1? because in layer file, every node has itself + M neighbours
        # and sometimes we will have closed groups of such nodes, each group has M + 1 nodes
        top_nodes = self.retrive_n(query, min(self.M + 1, top_k))
        if len(top_nodes) == top_k:
            top_nodes_ids = [i.id for i in top_nodes]
            return top_nodes_ids
        
        top_nodes_dict = {node.id: node for node in top_nodes}
        
        previous_len = 0
        new_len = len(top_nodes_dict)
        # we need to add more nodes to the result
        while new_len != previous_len and new_len != top_k:
            # retive more nodes and update set
            top_nodes = self.retrive_n(np.array(top_nodes[-1].vector), min(self.M + 1, top_k - new_len))
            
            for node in top_nodes:
                if not top_nodes_dict.get(node.id):
                    top_nodes_dict[node.id] = node
            
            previous_len = new_len
            new_len = len(top_nodes_dict)
        
        top_nodes = [top_nodes_dict[node_id] for node_id in top_nodes_dict]
        
        # sort nodes by similarity (put larger similarity first), return these nodes
        sorted(top_nodes, key=lambda n: self.calculate_similarity(n, Node(-1, query.T)), reverse=True)
        top_nodes_ids = [i.id for i in top_nodes]
        if new_len < top_k:
            # add random nodes
            top_nodes_ids.extend([random.randint(0, self.layer_sizes[0] - 1) for _ in range(top_k - new_len)])
            
        return top_nodes_ids