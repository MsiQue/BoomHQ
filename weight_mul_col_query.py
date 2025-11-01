import random
import pickle
from collections import defaultdict
from gen_mul_col_query import get_query_vector, extract_nth_distance_query
from global_info import get_database_cursor, execute_sql_with_args_explain_returned, convert_to_list, calc

def weight_mul_col_query_topk_sql(column_name, table_name, w1, vector_column_1, vector_1, w2, vector_column_2, vector_2, limit, explain = False, opt = '<->'):
    prefix = 'EXPLAIN ' if explain else ''
    return prefix + f"SELECT {column_name} FROM {table_name} ORDER BY {w1} * ({vector_column_1} {opt} '%s') + {w2} * ({vector_column_2} {opt} '%s') LIMIT {limit} ;" % (vector_1, vector_2)

def gen_mul_col_query_sql(cursor, table_name, vec_column_name_1, vec_column_name_2, N, limitk, file_name_suffix = ''):
    query_sql_list = []
    for i in range(N):
        v1 = get_query_vector(cursor, table_name, vec_column_name_1)
        v2 = get_query_vector(cursor, table_name, vec_column_name_2)
        w1 = random.random()
        w2 = 1.0 - w1
        target_sql = weight_mul_col_query_topk_sql('id', table_name, w1, vec_column_name_1, v1, w2, vec_column_name_2, v2, limitk)

        print(i)
        print(target_sql)
        query_sql_list.append((target_sql, 1.0))

    filepath = f'query/{N}_weight_mul_col_query_sql_[{table_name}]_limit{limitk}{file_name_suffix}.pickle'
    pickle.dump(query_sql_list, open(filepath, 'wb'))

def cte_mul_col(ori_sql, pre_sql, large_k):
    pre_sql = pre_sql.replace('id', 'id, p_name_vector, p_comment_vector').replace(';', '').replace('100', str(large_k))
    new_sql = f"with filtered as MATERIALIZED ({pre_sql}) {ori_sql.replace('part', 'filtered')}"
    return new_sql

def do_sql(cursor, ori_sql, pre_sql, large_k, gt):
    final_sql = cte_mul_col(ori_sql, pre_sql, large_k)
    res, t, explain_returned = execute_sql_with_args_explain_returned(cursor, final_sql, [], 'explain ')
    res = convert_to_list(res)
    acc = calc(res, gt)
    print('-'*50)
    for _ in explain_returned:
        print(_)
    return acc, t, explain_returned

def run_sql(cursor, query_path, gt_path, save_path):
    query_sql_list = pickle.load(open(query_path, 'rb'))
    gt_list = pickle.load(open(gt_path, 'rb'))
    result_dict = defaultdict(dict)
    for large_k in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]:
        for i, ((sql, _), (gt, __, ___)) in enumerate(zip(query_sql_list, gt_list)):
            sql_1 = extract_nth_distance_query(sql, 1, False)
            sql_2, components = extract_nth_distance_query(sql, 2, True)
            sub_vec = [eval(eval(c.split('<->')[1][:-1])) for c in components]
            cat_vec = sub_vec[0] + sub_vec[1]
            sql_12 = sql_1.replace(str(sub_vec[0]), str(cat_vec)).replace('p_comment_vector', 'vec_2_combine')
            sql_dict = {}
            sql_dict['sql_1'] = do_sql(cursor, sql, sql_1, large_k, gt)
            sql_dict['sql_2'] = do_sql(cursor, sql, sql_2, large_k, gt)
            sql_dict['sql_12'] = do_sql(cursor, sql, sql_12, large_k, gt)
            sql_dict['table_scan'] = (1.0, __)
            result_dict[large_k][i] = sql_dict
            print('='*50)

    pickle.dump(result_dict, open(save_path, 'wb'))

if __name__ == '__main__':
    cursor = get_database_cursor('tpch')
    # gen_mul_col_query_sql(cursor, 'part', 'p_comment_vector', 'p_name_vector', 10000, 100)

    N = 10000
    table_name = 'part'
    query_path = f'query/{N}_weight_mul_col_query_sql_[{table_name}]_limit100.pickle'
    gt_path = f'ground_truth/{N}_weight_mul_col_query_ground_truth_[{table_name}]_limit100.pickle'
    save_path = f'result/weight_mul_col/{N}_weight_mul_col_query_result_[{table_name}]_limit100.pickle'
    run_sql(cursor, query_path, gt_path, save_path)