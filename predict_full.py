import pickle
import time
import numpy as np
import torch
import re
import os
from runtime_predictor import get_sql_info, Predictor
from binary_classification import Binary_Predictor
from global_info import table_name_to_vector_column_name, get_database_cursor, get_number_column_list, column_diff_value_cnt, get_range_list, ef_search_list, iterative_scan_list

scalar_position_dim_dict = {
    'aka_title' : 66,
    'aka_name' : 20,
    'title' : 56,
    'supplier': 10,
    'part' : 40,
    'partsupp' : 50,
    'orders' : 41,
    'lineitem' : 86
}
scalar_magnitude_dim_dict = {
    'aka_title' : 11,
    'aka_name' : 11,
    'title' : 11,
    'supplier': 10,
    'part' : 11,
    'partsupp' : 11,
    'orders' : 11,
    'lineitem' : 11
}

def get_info_dict_table(database, table_name):
    info_dict_table = {}
    cursor = get_database_cursor(database)
    info_dict_table['cursor'] = cursor
    number_column_list = get_number_column_list(cursor, table_name)
    number_column_range_list = []
    for c in number_column_list:
        number_column_range_list.append((c, column_diff_value_cnt(cursor, table_name, c), get_range_list(cursor, table_name, c)))
    info_dict_table['number_column_range_list'] = number_column_range_list
    info_dict_table['scalar_position_dim'] = scalar_position_dim_dict[table_name]
    info_dict_table['scalar_magnitude_dim'] = scalar_magnitude_dim_dict[table_name]
    return info_dict_table

def get_info_dict():
    info_dict = {}
    for table_name, vector_column_name in table_name_to_vector_column_name.items():
        database = 'tpch' if 'comment' in vector_column_name else 'imdb'
        info_dict[table_name] = get_info_dict_table(database, table_name)
    return info_dict

def extract_table_name(sql_query):
    match = re.search(r'from\s+([a-zA-Z0-9_]+)(?:\s+where|\s|$)', sql_query, re.IGNORECASE)
    if match:
        return match.group(1)
    return None

def predict(sql, threshold, info_dict):
    table_name = extract_table_name(sql)
    _v, _s1, _s2, _pre = get_sql_info(sql, {'interval_cnt': 10, 'number_column_range_list': info_dict[table_name]['number_column_range_list'], 'cursor': info_dict[table_name]['cursor']})
    s1 = torch.tensor(_s1).float()
    # s2 = torch.tensor(_s2).float()
    s2 = torch.tensor([_s2[0]] * 11).float()
    pre = torch.tensor(_pre)
    # pre = torch.tensor([_, _, _])
    index_info = torch.tensor([16, 64]).float()

    binary_model = Binary_Predictor(hidden_dim=256, scalar_position_dim=info_dict[table_name]['scalar_position_dim'], scalar_magnitude_dim=info_dict[table_name]['scalar_magnitude_dim'], output_dim=2)
    binary_model.load_state_dict(torch.load(f'model/model_binary_{table_name}.pth'))
    binary_model = binary_model.to('cpu')
    is_scalar_first = torch.argmax(binary_model(s1.unsqueeze(0), s2.unsqueeze(0), pre.unsqueeze(0), index_info.unsqueeze(0))).item()

    if is_scalar_first == 1:
        return is_scalar_first, None, None

    VAE = torch.load(f'VAE_model/VAE_{table_name}_[data_sample_cnt_1000]_[interval_cnt_10].pth')
    a_model = Predictor(VAE, hidden_dim=256, vae_output_dim=512, output_dim=1)
    a_model.load_state_dict(torch.load(f'model/model_accuracy_predictor_{table_name}.pth'))
    a_model = a_model.to('cpu')

    t_model = Predictor(VAE, hidden_dim=256, vae_output_dim=512, output_dim=1)
    t_model.load_state_dict(torch.load(f'model/model_runtime_predictor_{table_name}.pth'))
    t_model = t_model.to('cpu')

    v = torch.tensor([_v for _ in range(30)])
    s1 = torch.tensor([_s1 for _ in range(30)]).float()
    pre = torch.tensor([_pre for _ in range(30)])
    index_info = torch.tensor([[16, 64] for _ in range(30)]).float()
    query = torch.tensor([[e, i] for e in range(10) for i in range(3)]).float()

    a_pred = a_model(v, s1, pre, index_info, query)
    t_pred = t_model(v, s1, pre, index_info, query)

    max_a = torch.max(a_pred).item()
    if max_a >= threshold:
        indices = torch.where(a_pred > threshold)[0]
        valid_t_pred = t_pred[indices]
        min_rel_idx = torch.argmin(valid_t_pred)
        pred = indices[min_rel_idx].item()
    else:
        pred = torch.argmax(a_pred).item()

    iterative_scan = iterative_scan_list[pred % len(iterative_scan_list)]
    ef_search = ef_search_list[pred // len(iterative_scan_list)]

    return is_scalar_first, ef_search, iterative_scan

def predict_full(all_query_path, threshold, info_dict):
    all_query = pickle.load(open(all_query_path, 'rb'))
    right_cnt = 0
    pred_total_time = 0
    best_total_time = 0
    pgvector_time = 0
    ave_a_pred = 0
    ave_a_pgvector = 0
    ave_ratio = 0
    t_model_pred_list = []
    t_pgvector_list = []
    for i, sql_dict in enumerate(all_query):
        is_scalar_first, ef_search, iterative_scan = predict(sql_dict['sql'], threshold, info_dict)
        if is_scalar_first == 1:
            a_model_pred = sql_dict['acc_scalar_first']
            t_model_pred = sql_dict['time_scalar_first']
        else:
            if iterative_scan in sql_dict['time_vector_first'][ef_search]:
                a_model_pred = sql_dict['acc_vector_first'][ef_search][iterative_scan]
                t_model_pred = sql_dict['time_vector_first'][ef_search][iterative_scan]
            else:
                a_model_pred = sql_dict['acc_vector_first'][ef_search]['relaxed_order']
                t_model_pred = sql_dict['time_vector_first'][ef_search]['relaxed_order']

        a_pgvector = sql_dict['acc_vector_first'][200]['relaxed_order']
        t_pgvector = sql_dict['time_vector_first'][200]['relaxed_order']

        pred_total_time += t_model_pred
        pgvector_time += t_pgvector
        t_model_pred_list.append(t_model_pred)
        t_pgvector_list.append(t_pgvector)
        best_total_time += min(sql_dict['time_scalar_first'], t_pgvector)
        real = 1 if sql_dict['time_scalar_first'] < t_pgvector else 0

        ave_a_pred += a_model_pred
        ave_a_pgvector += a_pgvector

        if is_scalar_first == real:
            right_cnt += 1

        ratio = t_model_pred / t_pgvector
        ave_ratio += ratio

        # print(f"{i : 20} {a_model_pred:20} {a_pgvector:20} {ave_a_pred / (i + 1) : 20} {ave_a_pgvector / (i + 1) : 20} {f'pred={is_scalar_first}':10} {f'real={real}':10} {(-1 if ef_search is None else ef_search):10} {('None' if iterative_scan is None else iterative_scan):20} {t_model_pred : 20} {t_pgvector : 20}  {pred_total_time : 20} {best_total_time : 20} {pgvector_time : 20} {pred_total_time / pgvector_time : 20} {best_total_time / pgvector_time : 20}")
        print(f"{i : 20} {extract_table_name(sql_dict['sql'])} {a_model_pred:20} {a_pgvector:20} {ave_a_pred / (i + 1) : 20} {ave_a_pgvector / (i + 1) : 20} {f'pred={is_scalar_first}':10} {f'real={real}':10} {(-1 if ef_search is None else ef_search):10} {('None' if iterative_scan is None else iterative_scan):20} {t_model_pred : 20} {t_pgvector : 20}  {pred_total_time : 20} {pgvector_time : 20} {pred_total_time / pgvector_time : 20} {ratio : 20} {ave_ratio / (i + 1) : 20}")

    # print('right = ', right_cnt)
    print(np.percentile(t_model_pred_list, [1, 50, 99]))
    print(np.percentile(t_pgvector_list, [1, 50, 99]))

if __name__ == '__main__':
    info_dict = get_info_dict()
    predict_full('all_query/all_query.pickle', 0.8, info_dict)

