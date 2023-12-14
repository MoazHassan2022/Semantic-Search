import numpy as np
import time
from dataclasses import dataclass
from memory_profiler import memory_usage
from typing import List
from vec_db import VecDB

AVG_OVERX_ROWS = 10

@dataclass
class Result:
    run_time: float
    top_k: int
    db_ids: List[int]
    actual_ids: List[int]

def run_queries(db, np_rows, top_k, num_runs):
    results = []
    for _ in range(num_runs):
        query = np.random.random((1,70))
        
        tic = time.time()
        actual_ids = np.argsort(np_rows.dot(query.T).T / (np.linalg.norm(np_rows, axis=1) * np.linalg.norm(query)), axis= 1).squeeze().tolist()[::-1]
        toc = time.time()
        
        tic = time.time()
        # mem_before = max(memory_usage())
        # mem = memory_usage(proc=(db.retrive, (query,top_k), {}), interval = 1e-3)
        # print("Memory used: %s MB" % (max(mem) - mem_before))
        db_ids = db.retrive(query,top_k)
        toc = time.time()
        run_time = toc - tic
        
        np_run_time = toc - tic
        
        results.append(Result(run_time, top_k, db_ids, actual_ids))
    return results

def eval(results: List[Result]):
    # scores are negative. So getting 0 is the best score.
    scores = []
    run_time = []
    for res in results:
        run_time.append(res.run_time)
        # case for retireving number not equal to top_k, socre will be the lowest
        if len(set(res.db_ids)) != res.top_k or len(res.db_ids) != res.top_k:
            scores.append( -1 * len(res.actual_ids) * res.top_k)
            continue
        score = 0
        for id in res.db_ids:
            try:
                ind = res.actual_ids.index(id)
                if ind > res.top_k * 3:
                    score -= ind
            except:
                score -= len(res.actual_ids)
        scores.append(score)

    return sum(scores) / len(scores), sum(run_time) / len(run_time)


if __name__ == "__main__":
    # without insertion
    """db = VecDB(new_db=False, file_path='1000000')
    res = run_queries(db, [], top_k=5, num_runs=1)
    print(eval(res))"""
    
    # with insertion
    
    # db = VecDB(file_path='10000',new_db=False)
    # with open('saved_db10000', "r") as fin:
    #     records_np = np.loadtxt(fin, delimiter=",", dtype=np.float32)
    # res = run_queries(db, records_np, top_k=5, num_runs=5)
    # print(eval(res))
    
    db = VecDB()
    records_np = np.random.random((1000000, 70))
    records_dict = [{"id": i, "embed": list(row)} for i, row in enumerate(records_np)]
    tic = time.time()
    db.insert_records(records_dict)
    toc = time.time()
    print(f'Index craeted! time = {toc-tic}')
    res = run_queries(db, records_np, top_k=5, num_runs=10)
    print(eval(res))
    
    # TEST
    """print(records_np[5])
    tic = time.time()
    db_ids = db.retrive(records_np[5], 1)
    print(db_ids)
    toc = time.time()
    run_time = toc - tic"""
    
    # records_np = np.concatenate([records_np, np.random.random((90000, 70))])
    # records_dict = [{"id": i + _len, "embed": list(row)} for i, row in enumerate(records_np[_len:])]
    # _len = len(records_np)
    # db.insert_records(records_dict)
    # res = run_queries(db, records_np, 5, 10)
    # print(eval(res))

    # records_np = np.concatenate([records_np, np.random.random((900000, 70))])
    # records_dict = [{"id": i + _len, "embed": list(row)} for i, row in enumerate(records_np[_len:])]
    # _len = len(records_np)
    # db.insert_records(records_dict)
    # res = run_queries(db, records_np, 5, 10)
    # eval(res)

    # records_np = np.concatenate([records_np, np.random.random((4000000, 70))])
    # records_dict = [{"id": i + _len, "embed": list(row)} for i, row in enumerate(records_np[_len:])]
    # _len = len(records_np)
    # db.insert_records(records_dict)
    # res = run_queries(db, records_np, 5, 10)
    # eval(res)

    # records_np = np.concatenate([records_np, np.random.random((5000000, 70))])
    # records_dict = [{"id": i + _len, "embed": list(row)} for i, row in enumerate(records_np[_len:])]
    # _len = len(records_np)
    # db.insert_records(records_dict)
    # res = run_queries(db, records_np, 5, 10)
    # eval(res)

    # records_np = np.concatenate([records_np, np.random.random((5000000, 70))])
    # records_dict = [{"id": i +  _len, "embed": list(row)} for i, row in enumerate(records_np[_len:])]
    # _len = len(records_np)
    # db.insert_records(records_dict)
    # res = run_queries(db, records_np, 5, 10)
    # eval(res)

    