from global_info import table_name_to_vector_column_name, get_database_cursor, get_number_column_list
from dataset import CustomDataset
from learning import supervised_learning
import torch
import pickle
import numpy as np
from global_info import get_number_column_list, get_database_cursor, get_range_list, column_diff_value_cnt, get_vector_from_sql, get_where_clause_from_sql, execute_sql_with_args_explain_returned, convert_to_list, calc, get_large_k_sql, ef_search_list, iterative_scan_list
from model import vector_scalar, loss_function, NN
import torch.nn as nn
from sql import get_explain_info, extract_table_name
import os
from column_partition import pred_feature
from collections import Counter

# class Binary_Predictor(nn.Module):
#     def __init__(self, hidden_dim, features_dim, output_dim):
#         super(Binary_Predictor, self).__init__()
#         self.f = NN(features_dim, hidden_dim, output_dim)
#
#     def forward(self, features):
#         return self.f(features)

class Predictor(nn.Module):
    def __init__(self,  hidden_dim, features_dim, output_dim):
        super(Predictor, self).__init__()
        self.bn = nn.BatchNorm1d(features_dim)
        self.fc1 = nn.Linear(features_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.bn(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

def model_train(features, label_list, model_save_path, output_dim):
    # device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    n = features.shape[0]
    train_size = int(0.2 * n)
    val_size = int(0.2 * n)
    test_size = n - train_size - val_size

    train_label_list = label_list[:train_size]
    train_features = features[:train_size]

    val_label_list = label_list[train_size:train_size + val_size]
    val_features = features[train_size:train_size + val_size]

    test_label_list = label_list[train_size + val_size:]
    test_features = features[train_size + val_size:]

    train_dataset = CustomDataset([train_features], train_label_list)
    val_dataset = CustomDataset([val_features], val_label_list)
    test_dataset = CustomDataset([test_features], test_label_list)

    model = Predictor(hidden_dim=256, features_dim=features.shape[1], output_dim=output_dim)

    def F(model, bacth):
        # query_scalar_position, query_scalar_magnitude, pre_check_list, index_info = bacth
        # return model(query_scalar_position, query_scalar_magnitude, pre_check_list, index_info)
        features = bacth[0]
        return model(features)

    supervised_learning(F, model, train_dataset, val_dataset, test_dataset, num_epochs =100, batch_size=64)

    torch.save(model, model_save_path)

def get_ef_search_iterative_scan(sql_dict, threshold):
    best_t = 1e10
    best_ef = 9     # 1000
    best_it = 2     # relax
    for iterative_scan_id, iterative_scan in enumerate(iterative_scan_list):
        for ef_search_id, ef_search in enumerate(ef_search_list):
            if sql_dict['acc_vector_first'][ef_search][iterative_scan] > threshold:
                if sql_dict['time_vector_first'][ef_search][iterative_scan] < best_t:
                    best_t = sql_dict['time_vector_first'][ef_search][iterative_scan]
                    best_ef = ef_search_id
                    best_it = iterative_scan_id
    return best_ef, best_it

def get_train_info(kwargs):
    query_sql_list = pickle.load(open(kwargs['sql_list_file'], 'rb'))

    features = []
    pred_ef_search_list = []
    pred_iterative_scan_list = []
    for sql_dict in query_sql_list:
        pre = get_explain_info(kwargs['cursor'], sql_dict['sql'])
        for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99]:
            product_1, product_2 = pred_feature(sql_dict['sql'], kwargs)

            if kwargs['use_selectivity']:
                features.append([sql_dict['selectivity']] + [product_1, product_2] + [pre[0] / kwargs['row_cnt']] + pre[1:] + [threshold])
            else:
                features.append([product_1, product_2] + [pre[0] / kwargs['row_cnt']] + pre[1:] + [threshold])

            best_ef, best_it = get_ef_search_iterative_scan(sql_dict, threshold)
            pred_ef_search_list.append(best_ef)
            pred_iterative_scan_list.append(best_it)

    print(Counter(pred_ef_search_list))
    print(Counter(pred_iterative_scan_list))

    features = torch.tensor(features, dtype=torch.float)
    pred_ef_search_list = torch.tensor(pred_ef_search_list, dtype=torch.int64)
    pred_iterative_scan_list = torch.tensor(pred_iterative_scan_list, dtype=torch.int64)

    return features, pred_ef_search_list, pred_iterative_scan_list

def train(database, table_name, sql_list_file, use_selectivity):
    print('-' * 50)
    print(database, table_name, 'use_selectivity' if use_selectivity else 'no_selectivity')
    kwargs = {}
    kwargs['sql_list_file'] = sql_list_file
    cursor = get_database_cursor(database)
    kwargs['cursor'] = cursor
    cursor.execute(f'select count(*) from {table_name}')
    kwargs['row_cnt'] = cursor.fetchone()[0]
    number_column_list = get_number_column_list(cursor, table_name)
    number_column_range_list = []
    for c in number_column_list:
        number_column_range_list.append((c, column_diff_value_cnt(cursor, table_name, c), get_range_list(cursor, table_name, c)))

    kwargs['number_column_range_list'] = number_column_range_list
    kwargs['table_name'] = table_name
    kwargs['interval_cnt'] = 10
    kwargs['use_selectivity'] = use_selectivity
    kwargs['column_partition'] = pickle.load(open(f'column_statistic/column_partition_[1000].pickle', 'rb'))

    features, pred_ef_search_list, pred_iterative_scan_list = get_train_info(kwargs)

    if use_selectivity:
        model_train(features, pred_ef_search_list, f'model/model_ef_search_selectivity/model_ef_search_selectivity_{table_name}.pth', 10)
        model_train(features, pred_iterative_scan_list, f'model/model_iterative_scan_selectivity/model_iterative_scan_selectivity_{table_name}.pth', 3)
    else:
        model_train(features, pred_ef_search_list, f'model/model_ef_search/model_ef_search_{table_name}.pth', 10)
        model_train(features, pred_iterative_scan_list, f'model/model_iterative_scan/model_iterative_scan_{table_name}.pth', 3)


    return kwargs

def predict(sql_dict, threshold, ef_search_model, iterative_scan_model, kwargs):
    table_name = extract_table_name(sql_dict['sql'])

    pre = get_explain_info(kwargs['cursor'], sql_dict['sql'])
    product_1, product_2 = pred_feature(sql_dict['sql'], kwargs)

    # features = [sql_dict['selectivity']] + [product_1, product_2] + [pre[0] / kwargs['row_cnt']] + pre[1:] + [threshold]
    # features = [product_1, product_2] + [pre[0] / kwargs['row_cnt']] + pre[1:] + [threshold]

    if kwargs['use_selectivity']:
        features = [sql_dict['selectivity']] + [product_1, product_2] + [pre[0] / kwargs['row_cnt']] + pre[1:] + [threshold]
    else:
        features = [product_1, product_2] + [pre[0] / kwargs['row_cnt']] + pre[1:] + [threshold]

    features = torch.tensor(features).unsqueeze(0)

    ef_search_pred = ef_search_model(features).argmax().item()
    iterative_scan_pred = iterative_scan_model(features).argmax().item()

    return ef_search_list[ef_search_pred], iterative_scan_list[iterative_scan_pred]

def run_nn(database, table_name, threshold, kwargs):
    query_sql_list = pickle.load(open(kwargs['sql_list_file'], 'rb'))

    right_cnt = 0
    pred_total_time = 0
    pgvector_time = 0
    ave_a_pred = 0
    ave_a_pgvector = 0
    ave_ratio = 0
    t_model_pred_list = []
    t_pgvector_list = []

    if kwargs['use_selectivity']:
        ef_search_model = torch.load(f'model/model_ef_search_selectivity/model_ef_search_selectivity_{table_name}.pth')
        iterative_scan_model = torch.load(f'model/model_iterative_scan_selectivity/model_iterative_scan_selectivity_{table_name}.pth')
    else:
        ef_search_model = torch.load(f'model/model_ef_search/model_ef_search_{table_name}.pth')
        iterative_scan_model = torch.load(f'model/model_iterative_scan/model_iterative_scan_{table_name}.pth')

    for i, sql_dict in enumerate(query_sql_list):
        ef_search, iterative_scan = predict(sql_dict, threshold, ef_search_model, iterative_scan_model, kwargs)

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

        ave_a_pred += a_model_pred
        ave_a_pgvector += a_pgvector

        ratio = t_model_pred / t_pgvector
        ave_ratio += ratio

        # print(f"{i : 20} {a_model_pred:20} {a_pgvector:20} {ave_a_pred / (i + 1) : 20} {ave_a_pgvector / (i + 1) : 20} {f'pred={is_scalar_first}':10} {f'real={real}':10} {(-1 if ef_search is None else ef_search):10} {('None' if iterative_scan is None else iterative_scan):20} {t_model_pred : 20} {t_pgvector : 20}  {pred_total_time : 20} {best_total_time : 20} {pgvector_time : 20} {pred_total_time / pgvector_time : 20} {best_total_time / pgvector_time : 20}")
        print(f"{i : 20} {extract_table_name(sql_dict['sql'])} {a_model_pred:20} {a_pgvector:20} {ave_a_pred / (i + 1) : 20} {ave_a_pgvector / (i + 1) : 20} {(-1 if ef_search is None else ef_search):10} {('None' if iterative_scan is None else iterative_scan):20} {t_model_pred : 20} {t_pgvector : 20}  {pred_total_time : 20} {pgvector_time : 20} {pred_total_time / pgvector_time : 20} {ratio : 20} {ave_ratio / (i + 1) : 20}")



def solve(file_name, thershold, use_selectivity):
    table_name = file_name.split('all_query_')[1].split('.')[0][:-4]
    if 'comment' in table_name_to_vector_column_name[table_name]:
        database_name = 'tpch'
    else:
        database_name = 'imdb'
    kwargs = train(database_name, table_name, os.path.join('/data/qiuermu/test_102', file_name), use_selectivity)
    run_nn(database_name, table_name, thershold, kwargs)

if __name__ == '__main__':
    file_name_list = [
        'all_query/all_query_aka_title_945.pickle',
        'all_query/all_query_title_421.pickle',
        'all_query/all_query_aka_name_501.pickle',
        'all_query/all_query_part_875.pickle',
        'all_query/all_query_partsupp_494.pickle',
        'all_query/all_query_orders_498.pickle',
        'all_query/all_query_lineitem_455.pickle'
        # 'all_query/all_query_3.pickle'
    ]

    for file_name in file_name_list:
        solve(file_name, 0.8, True)
        solve(file_name, 0.8, False)