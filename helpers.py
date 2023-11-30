import linecache as lc

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
    
