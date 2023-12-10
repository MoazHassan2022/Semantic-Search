import numpy as np
from typing import Dict, List, Annotated
from node import Node
from helpers import *
from sklearn.cluster import KMeans

class PQ:

    def __init__(self, records_per_file: int = 1000, num_subvectors = 14, num_centroids = 256) -> None:
        self.records_per_file = records_per_file
        self.num_subvectors = num_subvectors
        self.num_centroids = num_centroids


    # Calculate the distance between two nodes by calculating the cosine similarity between their vectors.
    def calculate_similarity(self, node1: Node, node2: Node) -> float:
        dot_product = np.dot(node1.pq_code, node2.pq_code)
        norm_vec1 = np.linalg.norm(node1.pq_code)
        norm_vec2 = np.linalg.norm(node2.pq_code)
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
        return cosine_similarity

    def insert_records(self, rows: List[Dict[int, Annotated[List[float], 70]]]):
        # Perform Product Quantization encoding for each subvector
        codebooks = []
        kmeans_models = []
        for i in range(self.num_subvectors):
            subvectors = np.array([row["embed"][i *  70//self.num_subvectors:(i + 1) * 70// self.num_subvectors] for row in rows])
            kmeans = KMeans(n_clusters=self.num_centroids, n_init=10)
            kmeans.fit(subvectors)
            codebooks.append(kmeans.cluster_centers_)
            kmeans_models.append(kmeans)
        self.codebooks = codebooks
        self.kmeans_models = kmeans_models
        self.data_size = len(rows)

        result_records = np.zeros((len(rows), self.num_subvectors), dtype=np.uint16)
        for i in range(self.num_subvectors):
            subvectors = np.array([row["embed"][i *  70//self.num_subvectors:(i + 1) * 70// self.num_subvectors] for row in rows])
            result_records[:, i] = kmeans_models[i].predict(subvectors)
        
        # insert records to the file such that each file has self.records_per_file row of records and name the file as data_0, data_1, etc.
        for i in range(len(rows) // self.records_per_file):
            with open(f'data/data_{i}', "w") as fout:
                for row in rows[i*self.records_per_file:(i+1)*self.records_per_file]:
                    id = row["id"]
                    pq_codes = result_records[id]
                    row_str = f"{id}," + ",".join([str(e) for e in pq_codes])
                    fout.write(f"{row_str}\n")
                    
        """ # insert records to the file such that each file has self.records_per_file row of records and name the file as data_0, data_1, etc.
        for i in range(len(rows) // self.records_per_file):
            with open(f'data/data_{i}', "w") as fout:
                for row in rows[i*self.records_per_file:(i+1)*self.records_per_file]:
                    id, embed = row["id"], row["embed"]
                    embed = np.array(embed)
                    subvectors = np.array_split(embed, self.num_subvectors)
                    pq_codes = [kmeans.predict(subvector.reshape(1, -1))[0] for subvector, kmeans in zip(subvectors, kmeans_models)]
                    row_str = f"{id}," + ",".join([str(e) for e in pq_codes])
                    fout.write(f"{row_str}\n") """

    def retrive(self, query: Annotated[List[float], 70], top_k = 1):
        '''
        steps: 
        loop over files and :
            1. get top_k records from each file based on cosine similarity
        append all the records to a list and sort them based on cosine similarity
        return the top_k records
        '''
        # Encode the query vector using Product Quantization
        sub_vectors = np.array_split(query[0], self.num_subvectors)
        query = [kmeans.predict(subvector.reshape(1, -1))[0] for subvector, kmeans in zip(sub_vectors, self.kmeans_models)]
        query_node = Node(-1, query)
        # in for loop over the files and get the top_k records from each file
        all_top_k_records = []
        for i in range(self.data_size // self.records_per_file):
            with open(f'data/data_{i}', "r") as fin:
                lines = fin.readlines()
                records = []
                for line in lines:
                    splitted_line = line.split(",") # id, pq_code
                    id, pq_code = int(splitted_line[0]), splitted_line[1:]
                    pq_code = np.array([np.uint16(e) for e in pq_code])
                    records.append(Node(id, pq_code))
                # get the top_k records from the current file
                sorted_records = sorted(records, key=lambda x: self.calculate_similarity(x, query_node), reverse=True)
                top_k_records = sorted_records[:top_k]
                # append the top_k_records to a list
                all_top_k_records.extend(top_k_records)
        # sort the list based on cosine similarity
        sorted_all_top_k_records = sorted(all_top_k_records, key=lambda x: self.calculate_similarity(x, query_node), reverse=True)
        # return the top_k records
        return [record.id for record in sorted_all_top_k_records[:top_k]]