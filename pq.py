import numpy as np
from typing import Dict, List, Annotated
from sklearn.cluster import KMeans

class PQ:

    def __init__(self, records_per_file: int = 1000, num_subvectors = 14, num_centroids = 4096) -> None:
        self.records_per_file = records_per_file
        self.num_subvectors = num_subvectors
        self.num_centroids = num_centroids

    def insert_records(self, rows: List[Dict[int, Annotated[List[float], 70]]]):
        # Perform Product Quantization encoding for each subvector
        codebooks = []
        kmeans_models = []
        subvector_size = 70 // self.num_subvectors
        for i in range(self.num_subvectors):
            subvectors = np.array([row["embed"][i * subvector_size : (i + 1) * subvector_size] for row in rows])
            kmeans = KMeans(n_clusters=self.num_centroids, n_init=10)
            kmeans.fit(subvectors)
            codebooks.append(kmeans.cluster_centers_)
            kmeans_models.append(kmeans)
        self.codebooks = codebooks
        self.data_size = len(rows)

        # Insert records to the file such that each file has self.records_per_file row of records and name the file as data_0, data_1, etc.
        for i in range(len(rows) // self.records_per_file):
            with open(f'data/data_{i}', "w") as fout:
                for row in rows[i*self.records_per_file:(i+1)*self.records_per_file]:
                    pq_codes = np.zeros((self.num_subvectors), dtype=np.uint16)
                    for j in range(self.num_subvectors):
                        # Predict the centroid of each subvector for each record
                        pq_codes[j] = kmeans_models[j].predict(row["embed"][j * subvector_size : (j + 1) * subvector_size])
                    # Print the pq_codes to the file
                    fout.write(",".join([str(e) for e in pq_codes]) + "\n")

    def retrive(self, query: Annotated[List[float], 70], top_k = 1):
        query = query[0]
        subvector_size = 70 // self.num_subvectors
        
        # Encode the query vector using Product Quantization
        # Caclulate eculidean distance between all the query vector subvectors and each of the centroids of that subvector, and store it in distances matrix
        # Every element of query_centroids_distances is a vector of size num_centroids, which contains the eculidean distance between the query vector subvector and each of the centroids of that subvector
        query_centroids_distances = np.zeros((self.num_subvectors, self.num_centroids))
        # codebooks is num_subvectors * num_centroids * ds, ds = 70 / num_subvectors
        for i in range(self.num_subvectors):
            query_centroids_distances[i] = np.linalg.norm(self.codebooks[i] - query[i * subvector_size : (i + 1) * subvector_size], axis=1)
        
        # Read all records from the files
        # For loop over the files and get the top_k records from each file
        records = np.empty((self.data_size, self.num_subvectors), dtype=np.uint16)
        records_counter = 0
        for i in range(self.data_size // self.records_per_file):
            with open(f'data/data_{i}', "r") as fin:
                lines = fin.readlines()
                for line in lines:
                    pq_code = line.split(",") # pq_code
                    pq_code = np.array([np.uint16(e) for e in pq_code])
                    records[records_counter] = pq_code
                    records_counter += 1
                    
        # Now, we have the distances between the query vector subvectors and each of the centroids of that subvector
        # We need to find the distances between the query vector and each of the records in the database, based on ids of centroids in each record
        query_records_distances = np.zeros((self.data_size))
        for i in range(self.num_subvectors):
            query_records_distances += query_centroids_distances[i, records[:, i]]
            
        # Argsort the records based on the distances between the query vector and each of the records in the database
        # Note that indices are the records ids
        sorted_records_indices = np.argsort(query_records_distances)[:top_k]
        
        # Return the top_k records ids
        return sorted_records_indices