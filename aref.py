import numpy as np
from helpers import *
from sklearn.cluster import KMeans
import os
import csv
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances

BITS2DTYPE = {
    8: np.uint8,
    10: np.uint16,
}

class CustomIndexPQ:
    def _init_(
        self,
        d: int,
        m: int,
        nbits: int,
        **estimator_kwargs: str | int,
    ) -> None:
        if d % m != 0:
            raise ValueError("d needs to be a multiple of m")

        if nbits not in BITS2DTYPE:
            raise ValueError(f"Unsupported number of bits {nbits}")

        self.m = m
        self.k = 2**nbits
        self.d = d
        self.ds = d // m

        self.estimators = [
            KMeans(n_clusters=self.k, **estimator_kwargs, n_init=10) for _ in range(m)
        ]

        self.is_trained = False

        self.dtype = BITS2DTYPE[nbits]
        self.dtype_orig = np.float32

        # delete codes.csv if it exists
        if os.path.exists("codes.csv"):
            os.remove("codes.csv")

    # @description: train the model
    # @param {type} X: np array of shape (n, d) and dtype float32
    def train(self, X: np.ndarray) -> None:
        if self.is_trained:
            raise ValueError("Training multiple times is not allowed")

        for i in range(self.m):
            estimator = self.estimators[i]

            X_i = X[:, i * self.ds : (i + 1) * self.ds]

            estimator.fit(X_i)
            print(f"Finished training estimator {i}")

        self.is_trained = True

    # @description: encode the data
    # @param {type} X: np array of shape (n, d) and dtype float32
    # @return {type} result: np array of shape (n, m)
    def encode(self, X: np.ndarray) -> np.ndarray:
        n = len(X)
        result = np.empty((n, self.m), dtype=self.dtype)

        for i in range(self.m):
            estimator = self.estimators[i]
            X_i = X[:, i * self.ds : (i + 1) * self.ds]
            result[:, i] = estimator.predict(X_i)

        return result

    # @description: add the data to the model
    # @param {type} X: np array of shape (n, d) and dtype float32
    def add(self, X: np.ndarray) -> None:
        if not self.is_trained:
            raise ValueError("The quantizer needs to be trained first.")

        csv.writer(open(f"codes.csv", "a")).writerows(self.encode(X))

    # compute asymmetric distances to all database codes
    # @param {type} X: np array of shape (n, d) and dtype float32
    # @return {type} distances: np array of shape (n, n_codes) and dtype float32
    def compute_asymmetric_distances(self, X: np.ndarray, codes) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("The quantizer needs to be trained first.")

        n_queries = len(X)
        n_codes = len(codes)

        distance_table = np.empty(
            (n_queries, self.m, self.k), dtype=self.dtype_orig
        )  # (n_queries, m, k)

        for i in range(self.m):
            X_i = X[:, i * self.ds : (i + 1) * self.ds]  # (n_queries, ds)
            centers = self.estimators[i].cluster_centers_  # (k, ds)
            distance_table[:, i, :] = euclidean_distances(X_i, centers, squared=True)

        distances = np.zeros((n_queries, n_codes), dtype=self.dtype_orig)

        for i in range(self.m):
            distances += distance_table[:, i, codes[:, i]]

        return distances

    # @description: search the model
    # @param {type} X: np array of shape (n, d) and dtype float32
    # @param {type} k: int
    # @return {type} distances: np array of shape (n, k) and dtype float32
    # @return {type} indices: np array of shape (n, k) and dtype int64
    def search(self, X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        n_queries = len(X)
        
        # Read codes from csv file
        codes = np.array(pd.read_csv("codes.csv", header=None))

        distances_all = self.compute_asymmetric_distances(X, codes)

        indices = np.argsort(distances_all, axis=1)[:, :k]

        distances = np.empty((n_queries, k), dtype=np.float32)

        for i in range(n_queries):
            distances[i] = distances_all[i][indices[i]]

        return distances, indices