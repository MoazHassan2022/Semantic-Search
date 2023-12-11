import numpy as np
from typing import Dict, List, Annotated
from sklearn.cluster import KMeans

class PQ:

    def __init__(self, records_per_file: int = 1000, num_subvectors = 14) -> None:
        self.records_per_file = records_per_file
        self.num_subvectors = num_subvectors
        self.num_centroids_dict = {
            1000:256,
            10000:1800,
            100000:4096,
            1000000:16384,
            2000000: 32768,
            5000000:65536,
            10000000:131072,
            20000000:262144
        }

    def insert_records(self, rows: List[Dict[int, Annotated[List[float], 70]]]):
        self.num_centroids = self.num_centroids_dict[len(rows)]
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
                        pq_codes[j] = kmeans_models[j].predict(np.array(row["embed"][j * subvector_size : (j + 1) * subvector_size]).reshape(1, -1))[0]
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

        # create top_k_records array of size (self.k) that contain tuples of (record_id, distance) and initialize the distance with max int
        # for each file, read all records and calculate the distance between the query vector and each of the records in the database
        # extract the top_k records from both prevoius top_k_records and the current file
        # sort the top_k_records based on the distance
        # return the top_k records ids
        top_k_records = np.zeros((top_k), dtype=[('record_id', np.int32), ('distance', np.float32)])
        # Set distance to the maximum finite float32 value
        top_k_records['distance'] = np.finfo(np.float32).max
        records = np.empty((self.records_per_file, self.num_subvectors), dtype=np.uint16)
        
        for i in range(self.data_size // self.records_per_file):
            with open(f'data/data_{i}', "r") as fin:
                lines = fin.readlines()
                for j in range(len(lines)):
                    pq_code = lines[j].split(",") # pq_code
                    pq_code = np.array([np.uint16(e) for e in pq_code])
                    records[j] = pq_code
            # Now, we have the distances between the query vector subvectors and each of the centroids of that subvector
            # We need to find the distances between the query vector and each of the records in the database, based on ids of centroids in each record
            query_records_distances = np.zeros((self.records_per_file))
            for k in range(self.num_subvectors):
                query_records_distances += query_centroids_distances[k, records[:, k]]
            # merge distances with previous top_k_records in one array and sort them
            distances = np.concatenate(( query_records_distances,top_k_records['distance']))
            indices = np.argsort(distances)
            curr_top_k_records = np.zeros((top_k), dtype=[('record_id', np.int32), ('distance', np.float32)])
            for k in range(top_k):
                if indices[k] < self.records_per_file:
                    curr_top_k_records[k] = (i * self.records_per_file + indices[k], distances[indices[k]])
                else: 
                    curr_top_k_records[k] = top_k_records[indices[k] - self.records_per_file]
            top_k_records = curr_top_k_records

        # Return the top_k records ids
        return top_k_records['record_id']