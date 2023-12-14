import numpy as np
from typing import Dict, List, Annotated
from sklearn.cluster import KMeans
from collections import defaultdict
from linecache import getline
import os
class VecDB:

    def __init__(self, num_subvectors = 14, file_path = None, new_db = True) -> None:
        self.num_subvectors = num_subvectors
        if file_path != None:
            self.file_path = 'data' + file_path
            self.data_size = int(file_path)
        else:
            self.file_path = 'data'
        # Initialize the inverted index
        self.inverted_index = defaultdict(list)
        if new_db == False:
            # Load the self.codebooks from the codebooks file
            self.codebooks_file_path = 'codebooks' + file_path # file_path maybe 1000000
            self.inverted_index_path = 'inverted_index' + file_path
            self.load_codebooks()
            self.load_index()
    
    def calculate_similarity(self, node1, node2) -> float:
        dot_product = np.dot(node1, node2)
        norm_vec1 = np.linalg.norm(node1)
        norm_vec2 = np.linalg.norm(node2)
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
        return cosine_similarity

    def select_parameters(self):
        if self.data_size <= 1000:
            self.num_centroids = 256
        else:
            self.num_centroids = 1024
            
        self.records_per_read = 1000
        
        self.clusters_uncertainty = 15
        
        self.kmeans_iterations = 3
            
    def load_codebooks(self):
        self.select_parameters()
        codebooks = []
        for i in range(self.num_subvectors):
            with open(self.codebooks_file_path, "r") as fin:
                codebooks.append(np.loadtxt(fin, delimiter=",", dtype=np.float32, skiprows=i * self.num_centroids, max_rows=self.num_centroids))
        self.codebooks = codebooks

    def load_index(self):
        self.inverted_index = defaultdict(list)
        sub_space,centroid_id = 0,0
        with open(self.inverted_index_path, "r") as fin:
            for line in fin:
                self.inverted_index[(sub_space,centroid_id)] = [np.uint16(i) for i in line.split(',')]
                centroid_id += 1
                if(centroid_id == self.num_centroids):
                    sub_space += 1
                    centroid_id = 0
    
    def insert_records(self, rows: List[Dict[int, Annotated[List[float], 70]]]):
        if isinstance(rows, np.ndarray) and rows.shape[1] == 70:
            convert_rows = False
        else:
            convert_rows = True
        if convert_rows:
            rows = np.array([row["embed"] for row in rows])
        self.data_size = len(rows)
        
        self.select_parameters()
        # Ensure that self.file_path directory exists
        if not os.path.exists(self.file_path):
            os.makedirs(self.file_path)
            
        # Save the data in files each file has records_per_read records
        for i in range(0, self.data_size, self.records_per_read):
            with open(f'{self.file_path}/{i}', "w") as fout:
                np.savetxt(fout, rows[i:i+self.records_per_read], delimiter=",", fmt='%.8f')
        
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
            kmeans = KMeans(n_clusters=self.num_centroids, n_init=10, max_iter=self.kmeans_iterations, init='random')
            kmeans.fit(training_data[:, i * subvector_size : (i + 1) * subvector_size])
            codebooks.append(kmeans.cluster_centers_)
            kmeans_models.append(kmeans)
            print(f"Finished training model {i}")
        self.codebooks = codebooks
        # Save the codebooks to the codebooks file
        with open(f'codebooks', "w") as fout:
            for i in range(self.num_subvectors):
                np.savetxt(fout, codebooks[i], delimiter=",")

        pq_codes = np.zeros((self.data_size), dtype=np.uint16)
            
        # Update the inverted index during insertion
        for i in range(self.num_subvectors):
            # Predict the centroid of each subvector for each record
            pq_codes = kmeans_models[i].predict(rows[:, i * subvector_size : (i + 1) * subvector_size])
            for j in range(self.num_centroids):
                records_with_centroid_j = np.where(pq_codes == j)[0]
                self.inverted_index[(i, j)].extend(records_with_centroid_j.tolist())
            print(f"Finished predicting subvector {i}")
        
        # Save the inverted index to the inverted index file
        with open(f'inverted_index', "w") as fout:
            for key in self.inverted_index:
                fout.write(','.join([str(id) for id in self.inverted_index[key]])+"\n")
                
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
        
        # Use the inverted index to filter potential records
        potential_records = set()
        for k in range(self.num_subvectors):
            indexes = np.argsort(query_centroids_distances[k])[:self.clusters_uncertainty]
            for index in indexes:
                potential_records.update(self.inverted_index[(k, index)])
        
        records = np.zeros((len(potential_records), 71),dtype=np.float32)
        
        # Read only the potential records from the files
        for i,record_id in enumerate(potential_records):
            # read the line and append it to records list with linecache module
            file_num = (record_id // self.records_per_read) * self.records_per_read
            record = getline(f'{self.file_path}/{file_num}', (record_id % self.records_per_read) + 1).split(',')
            records[i] = np.array([record_id] + [np.float32(i) for i in record])
        # sort the records based on cosine similarity between each record and query vector 
        records = sorted(records, key=lambda x: self.calculate_similarity(x[1:], query), reverse=True)
        # Return the top_k records ids
        return [np.int32(record[0]) for record in records[:top_k]]
