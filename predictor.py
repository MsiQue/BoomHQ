from global_info import table_name_to_vector_column_name, get_database_cursor, get_number_column_list
from dataset import CustomDataset
from learning import supervised_learning_regression
import torch
import pickle
import numpy as np
from global_info import get_number_column_list, get_database_cursor, get_range_list, column_diff_value_cnt, get_vector_from_sql, get_where_clause_from_sql, execute_sql_with_args_explain_returned, convert_to_list, calc, get_large_k_sql, ef_search_list, iterative_scan_list
from model import vector_scalar, loss_function, NN
import torch.nn as nn
from sql import get_sql_info, extract_table_name

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

def model_train(features, label_list, model_save_path):
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

    model = Predictor(hidden_dim=256, features_dim=features.shape[1], output_dim=1)

    def F(model, bacth):
        # query_scalar_position, query_scalar_magnitude, pre_check_list, index_info = bacth
        # return model(query_scalar_position, query_scalar_magnitude, pre_check_list, index_info)
        features = bacth[0]
        return model(features)

    supervised_learning_regression(F, model, train_dataset, val_dataset, test_dataset)

    torch.save(model, model_save_path)

def get_train_info(kwargs):
    query_sql_list = pickle.load(open(kwargs['sql_list_file'], 'rb'))

    features = []
    acc_list = []
    time_list = []
    for sql_dict in query_sql_list:
        _, __, s_magnitude, pre = get_sql_info(sql_dict['sql'], kwargs)

        for iterative_scan_id, iterative_scan in enumerate(iterative_scan_list):
            for ef_search_id, ef_search in enumerate(ef_search_list):
                which = np.eye(3)[iterative_scan_id].tolist() + [ef_search_id]
                features.append(which + [sql_dict['selectivity']] + [s_magnitude[0]] + [pre[0] / kwargs['row_cnt']] + pre[1:])
                # features.append(which + [s_magnitude[0]] + [pre[0] / kwargs['row_cnt']] + pre[1:])
                acc_list.append(sql_dict['acc_vector_first'][ef_search][iterative_scan])
                time_list.append(sql_dict['time_vector_first'][ef_search][iterative_scan])

    features = torch.tensor(features, dtype=torch.float)
    acc_list = torch.tensor(acc_list, dtype=torch.float)
    time_list = torch.tensor(time_list, dtype=torch.float)

    return features, acc_list, time_list

def train(database, table_name, sql_list_file):
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
    features, acc_list, time_list = get_train_info(kwargs)

    model_train(features, acc_list, f'model/model_acc_{table_name}.pth')
    model_train(features, time_list, f'model/model_time_{table_name}.pth')

    return kwargs

def predict(sql_dict, threshold, kwargs):
    table_name = extract_table_name(sql_dict['sql'])

    _, __, s_magnitude, pre = get_sql_info(sql_dict['sql'], kwargs)

    # features = [s_magnitude[0]] + [pre[0] / kwargs['row_cnt']] + pre[1:]
    features = [sql_dict['selectivity']] + [s_magnitude[0]] + [pre[0] / kwargs['row_cnt']] + pre[1:]

    features = torch.tensor([features for _ in range(30)])
    prefix1 = torch.cat([torch.eye(3) for _ in range(10)], dim=0)
    prefix2 = torch.tensor([i for i in range(10) for _ in range(3)]).reshape(-1, 1)
    combined = torch.cat((prefix1, prefix2, features), dim=1)

    a_model = torch.load(f'model/model_acc_{table_name}.pth')
    t_model = torch.load(f'model/model_time_{table_name}.pth')
    a_pred = a_model(combined)
    t_pred = t_model(combined)

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

    return ef_search, iterative_scan

def run_nn(database, table_name, threshold, kwargs):
    cursor = get_database_cursor(database)
    query_sql_list = pickle.load(open(kwargs['sql_list_file'], 'rb'))

    right_cnt = 0
    pred_total_time = 0
    pgvector_time = 0
    ave_a_pred = 0
    ave_a_pgvector = 0
    ave_ratio = 0
    t_model_pred_list = []
    t_pgvector_list = []
    for i, sql_dict in enumerate(query_sql_list):
        ef_search, iterative_scan = predict(sql_dict, threshold, kwargs)

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



def solve(database, table_name, N):
    kwargs = train(database, table_name, f'all_query/all_query_{table_name}_{N}.pickle')
    run_nn(database, table_name, 0.8, kwargs)

if __name__ == '__main__':
    res = []
    for table_name, vector_column_name in table_name_to_vector_column_name.items():
        database = 'tpch' if 'comment' in vector_column_name else 'imdb'
        #
        # if table_name == 'title' or table_name == 'aka_title' or table_name == 'aka_name' or table_name == 'supplier':
        #     continue

        if table_name == 'supplier':
            continue

        # if table_name != 'lineitem' and table_name !='orders':
        #     continue

        # if table_name != 'part':
        #     continue

        print('-' * 50)
        print(database, table_name, vector_column_name)

        res.append(solve(database, table_name, 1000))

    print(res)