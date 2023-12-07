import linecache as lc
from node import Node
import numpy as np

def file_binary_search(file_path, record_id, num_lines):
    '''
    This method is used to search for a record in a file using binary search
    '''
    # initialize the start and end of the binary search
    start = 1
    end = num_lines
    while start <= end:
        # get the middle line
        mid = (start + end) // 2
        # get the record of the middle line
        mid_record = lc.getline(file_path, mid)
        # get the id of the middle record
        mid_id = int(mid_record.split(',')[0])
        # check if the middle record is the record we are searching for
        if mid_id == record_id:
            return mid_record
        # if the middle record is greater than the record we are searching for, then the record we are searching for is in the first half of the file
        elif mid_id > record_id:
            end = mid - 1
        # if the middle record is less than the record we are searching for, then the record we are searching for is in the second half of the file
        else:
            start = mid + 1
    # if the record is not found, return None
    return None
    
# Calculate the distance between two nodes by calculating the cosine similarity between their vectors.
def calculate_similarity(node1: Node, node2: Node) -> float:
    dot_product = np.dot(node1.vector, node2.vector)
    norm_vec1 = np.linalg.norm(node1.vector)
    norm_vec2 = np.linalg.norm(node2.vector)
    cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
    return cosine_similarity
    
def calculate_pairwise_similarity(args):
    node_id1, nodes = args
    results = {}
    for node_id2 in range(node_id1 + 1, len(nodes)):
        node1, node2 = nodes[node_id1], nodes[node_id2]
        similarity = calculate_similarity(node1, node2)
        results[(node_id1, node_id2)] = similarity
    return results
