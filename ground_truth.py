import pickle
import tqdm
from global_info import execute_sql_with_args_explain_returned, convert_to_list

def calc_groundtruth(cursor, query_path, gt_path):
    query_sql_list = pickle.load(open(query_path, 'rb'))
    gt_list = []
    for sql, _ in tqdm.tqdm(query_sql_list):
        res, t, explain_returned = execute_sql_with_args_explain_returned(cursor, sql, ['enable_indexscan = off'], 'explain ')
        res = convert_to_list(res)
        gt_list.append((res, t, explain_returned))
    pickle.dump(gt_list, open(gt_path, 'wb'))

if __name__ == '__main__':
   pass