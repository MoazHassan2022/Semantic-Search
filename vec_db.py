import numpy as np
from typing import Dict, List, Annotated
from sklearn.cluster import KMeans

class VecDB:

    def __init__(self, num_subvectors = 14, file_path = None, new_db = True) -> None:
        self.num_subvectors = num_subvectors
        if file_path != None:
            self.file_path = 'saved_db' + file_path
            self.data_size = int(file_path)
        else:
            self.file_path = 'saved_db'
        if new_db == False:
            # Load the self.codebooks from the codebooks file
            self.codebooks_file_path = 'codebooks' + file_path # file_path maybe 1000000
            self.load_codebooks()

    def select_parameters(self):
        if self.data_size <= 1000:
            self.num_centroids = 256
        else:
            self.num_centroids = 1024
            
        if self.data_size <= 10000:
            self.records_per_read = 1000
        else:
            self.records_per_read = 10000
            
    def load_codebooks(self):
        self.select_parameters()
        codebooks = []
        for i in range(self.num_subvectors):
            with open(self.codebooks_file_path, "r") as fin:
                codebooks.append(np.loadtxt(fin, delimiter=",", dtype=np.float32, skiprows=i * self.num_centroids, max_rows=self.num_centroids))
        self.codebooks = codebooks
    
    def insert_records(self, rows: List[Dict[int, Annotated[List[float], 70]]]):
        rows = np.array([row["embed"] for row in rows])
        
        self.data_size = len(rows)
        
        self.select_parameters()
        
        # Train only on 1 million, if data is more than 1 million, else, train on all data
        training_data = None
        if self.data_size > 1000000:
            training_data = rows[:1000000]
        else:
            training_data = rows
        # Perform Product Quantization encoding for each subvector
        codebooks = []
        kmeans_models = []
        subvector_size = 70 // self.num_subvectors
        for i in range(self.num_subvectors):
            kmeans = KMeans(n_clusters=self.num_centroids, n_init=10, max_iter=10, init='random')
            kmeans.fit(training_data[:, i * subvector_size : (i + 1) * subvector_size])
            codebooks.append(kmeans.cluster_centers_)
            kmeans_models.append(kmeans)
            print(f"Finished training model {i}")
        self.codebooks = codebooks
        # Save the codebooks to the codebooks file
        with open(f'codebooks', "w") as fout:
            for i in range(self.num_subvectors):
                np.savetxt(fout, codebooks[i], delimiter=",", fmt="%.8f")
        # write to only 1 file
        with open(self.file_path, "w") as fout:
            pq_codes = np.zeros((self.data_size, self.num_subvectors), dtype=np.uint16)
            for i in range(self.num_subvectors):
                # Predict the centroid of each subvector for each record
                pq_codes[:, i] = kmeans_models[i].predict(rows[:, i * subvector_size : (i + 1) * subvector_size])
            # Save the pq_codes array to the file
            np.savetxt(fout, pq_codes, delimiter=",", fmt="%d")

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
        # create top_k_records array of size (self.k) that contain tuples of (record_id, distance) and initialize the distance with max int
        # for each file, read all records and calculate the distance between the query vector and each of the records in the database
        # extract the top_k records from both prevoius top_k_records and the current file
        # return the top_k records ids
        top_k_records = np.zeros((top_k), dtype=[('record_id', np.int32), ('distance', np.float32)])
        # Set distance to the maximum finite float32 value
        top_k_records['distance'] = np.finfo(np.float32).max
        
        for i in range(self.data_size // self.records_per_read):
            # Open the database file
            with open(self.file_path, 'r') as fin:
                # Read the records from the file, paginated
                records = np.loadtxt(fin, delimiter=",", dtype=np.uint16, skiprows=i * self.records_per_read, max_rows=self.records_per_read)
                
                # Now, we have the distances between the query vector subvectors and each of the centroids of that subvector
                # We need to find the distances between the query vector and each of the records in the database, based on ids of centroids in each record
                query_records_distances = np.zeros((self.records_per_read))
                for k in range(self.num_subvectors):
                    query_records_distances += query_centroids_distances[k, records[:, k]]
                # merge distances with previous top_k_records in one array and sort them
                distances = np.concatenate(( query_records_distances,top_k_records['distance']))
                indices = np.argsort(distances)
                curr_top_k_records = np.zeros((top_k), dtype=[('record_id', np.int32), ('distance', np.float32)])
                for k in range(top_k):
                    if indices[k] < self.records_per_read:
                        curr_top_k_records[k] = (i * self.records_per_read + indices[k], distances[indices[k]])
                    else: 
                        curr_top_k_records[k] = top_k_records[indices[k] - self.records_per_read]
                top_k_records = curr_top_k_records

        # Return the top_k records ids
        return top_k_records['record_id']