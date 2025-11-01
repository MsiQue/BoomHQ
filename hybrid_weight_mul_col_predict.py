import os
import torch
import pickle
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split
from collections import defaultdict
from weight_mul_col_query import extract_nth_distance_query
from global_info import get_database_cursor, execute_sql_with_args_explain_returned, convert_to_list, calc, get_where_clause_from_sql, from_mul_col_query_get_weight

def cte_mul_col(ori_sql, pre_sql, large_k, is_where_in_with):
    where_clause = get_where_clause_from_sql(pre_sql)
    pre_sql = pre_sql.replace(';', '')
    last_pos = pre_sql.rfind('100')
    pre_sql = pre_sql[:last_pos] + str(large_k) + pre_sql[last_pos + 3:]
    if not is_where_in_with:
        pre_sql = pre_sql.replace('WHERE', '').replace('where', '').replace(where_clause, '')
        pre_sql = pre_sql.replace('id', '*', 1)
    else:
        pre_sql = pre_sql.replace('id', 'id, p_name_vector, p_comment_vector', 1)
    ori_sql = ori_sql.replace(' part ', ' filtered ')
    if is_where_in_with:
        ori_sql = ori_sql.replace('WHERE', '').replace('where', '').replace(where_clause, '')
    new_sql = f"with filtered as MATERIALIZED ({pre_sql}) {ori_sql}"
    return new_sql

def do_sql(cursor, ori_sql, pre_sql, is_where_in_with, large_k, gt):
    final_sql = cte_mul_col(ori_sql, pre_sql, large_k, is_where_in_with)
    res, t, explain_returned = execute_sql_with_args_explain_returned(cursor, final_sql, [], 'explain ')
    res = convert_to_list(res)
    acc = calc(res, gt)
    # print('-'*50)
    # for _ in explain_returned:
    #     print(_)
    return acc, t, explain_returned

class MultiTaskNet(nn.Module):
    def __init__(self, input_size, num_classes_labels, num_classes_large_k):
        super(MultiTaskNet, self).__init__()
        self.shared_layer1 = nn.Linear(input_size, 64)
        self.shared_layer2 = nn.Linear(64, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

        self.labels_output = nn.Linear(128, num_classes_labels)
        self.large_k_output = nn.Linear(128, num_classes_large_k)

    def forward(self, x):
        x = self.relu(self.shared_layer1(x))
        x = self.dropout(x)
        x = self.relu(self.shared_layer2(x))
        x = self.dropout(x)

        out_labels = self.labels_output(x)
        out_large_k = self.large_k_output(x)

        return out_labels, out_large_k

def run_sql(cursor, query_path, gt_path, save_path, start_pos, end_pos, large_k_list):
    query_sql_list = pickle.load(open(query_path, 'rb'))
    gt_list = pickle.load(open(gt_path, 'rb'))
    result_dict = defaultdict(dict)
    for large_k in large_k_list:
        for i, ((sql, _), (gt, __, ___)) in enumerate(zip(query_sql_list, gt_list)):
            if i < start_pos or i >= end_pos:
                continue
            sql_1 = extract_nth_distance_query(sql, 1, False)
            sql_2, components = extract_nth_distance_query(sql, 2, True)
            sub_vec = [eval(eval(c.split('<->')[1][:-1])) for c in components]
            cat_vec = sub_vec[0] + sub_vec[1]
            sql_12 = sql_1.replace(str(sub_vec[0]), str(cat_vec)).replace('p_comment_vector', 'vec_2_combine')
            sql_dict = {}
            print(large_k, i)
            sql_dict['sql_1_where_in_with'] = do_sql(cursor, sql, sql_1, True, large_k, gt)
            sql_dict['sql_2_where_in_with'] = do_sql(cursor, sql, sql_2, True, large_k, gt)
            sql_dict['sql_12_where_in_with'] = do_sql(cursor, sql, sql_12, True, large_k, gt)
            sql_dict['sql_1_where_not_in_with'] = do_sql(cursor, sql, sql_1, False, large_k, gt)
            sql_dict['sql_2_where_not_in_with'] = do_sql(cursor, sql, sql_2, False, large_k, gt)
            sql_dict['sql_12_where_not_in_with'] = do_sql(cursor, sql, sql_12, False, large_k, gt)
            sql_dict['table_scan'] = (1.0, __)
            result_dict[large_k][i] = sql_dict
            print('='*50)

    pickle.dump(result_dict, open(save_path, 'wb'))

def pred(w1_list, w2_list, selectivity_list, th_list, labels, large_k_list, split_ratio=(0.7, 0.15, 0.15), epochs=50, batch_size=32,
         learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    inputs = torch.tensor(np.stack([w1_list, w2_list, selectivity_list, th_list], axis=1), dtype=torch.float32)

    unique_labels, counts_labels = np.unique(labels, return_counts=True)
    label_map = {label: i for i, label in enumerate(unique_labels)}
    labels_encoded = torch.tensor([label_map[l] for l in labels], dtype=torch.long)
    num_classes_labels = len(unique_labels)

    unique_large_k, counts_large_k = np.unique(large_k_list, return_counts=True)
    large_k_map = {k: i for i, k in enumerate(unique_large_k)}
    large_k_encoded = torch.tensor([large_k_map[k] for k in large_k_list], dtype=torch.long)
    num_classes_large_k = len(unique_large_k)

    dataset = TensorDataset(inputs, labels_encoded, large_k_encoded)

    dataset_size = len(dataset)
    train_size = int(split_ratio[0] * dataset_size)
    val_size = int(split_ratio[1] * dataset_size)
    test_size = dataset_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = MultiTaskNet(input_size=4, num_classes_labels=num_classes_labels,
                         num_classes_large_k=num_classes_large_k).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_inputs, batch_labels, batch_large_k in train_loader:
            batch_inputs, batch_labels, batch_large_k = batch_inputs.to(device), batch_labels.to(
                device), batch_large_k.to(device)

            optimizer.zero_grad()

            outputs_labels, outputs_large_k = model(batch_inputs)

            loss_labels = criterion(outputs_labels, batch_labels)
            loss_large_k = criterion(outputs_large_k, batch_large_k)

            total_loss = loss_labels + loss_large_k

            total_loss.backward()
            optimizer.step()

            train_loss += total_loss.item()

        model.eval()
        val_loss = 0.0
        correct_labels = 0
        correct_large_k = 0
        total_val = 0
        with torch.no_grad():
            for batch_inputs, batch_labels, batch_large_k in val_loader:
                batch_inputs, batch_labels, batch_large_k = batch_inputs.to(device), batch_labels.to(
                    device), batch_large_k.to(device)

                outputs_labels, outputs_large_k = model(batch_inputs)

                loss_labels = criterion(outputs_labels, batch_labels)
                loss_large_k = criterion(outputs_large_k, batch_large_k)
                total_loss = loss_labels + loss_large_k
                val_loss += total_loss.item()

                _, predicted_labels = torch.max(outputs_labels.data, 1)
                _, predicted_large_k = torch.max(outputs_large_k.data, 1)

                total_val += batch_labels.size(0)
                correct_labels += (predicted_labels == batch_labels).sum().item()
                correct_large_k += (predicted_large_k == batch_large_k).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        accuracy_labels = 100 * correct_labels / total_val
        accuracy_large_k = 100 * correct_large_k / total_val

        print(
            f'Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc (labels): {accuracy_labels:.2f}%, Val Acc (large_k): {accuracy_large_k:.2f}%')

    model.eval()
    test_loss = 0.0
    correct_labels = 0
    correct_large_k = 0
    total_test = 0
    with torch.no_grad():
        for batch_inputs, batch_labels, batch_large_k in test_loader:
            batch_inputs, batch_labels, batch_large_k = batch_inputs.to(device), batch_labels.to(
                device), batch_large_k.to(device)

            outputs_labels, outputs_large_k = model(batch_inputs)

            loss_labels = criterion(outputs_labels, batch_labels)
            loss_large_k = criterion(outputs_large_k, batch_large_k)
            total_loss = loss_labels + loss_large_k
            test_loss += total_loss.item()

            _, predicted_labels = torch.max(outputs_labels.data, 1)
            _, predicted_large_k = torch.max(outputs_large_k.data, 1)

            total_test += batch_labels.size(0)
            correct_labels += (predicted_labels == batch_labels).sum().item()
            correct_large_k += (predicted_large_k == batch_large_k).sum().item()

    avg_test_loss = test_loss / len(test_loader)
    accuracy_labels = 100 * correct_labels / total_test
    accuracy_large_k = 100 * correct_large_k / total_test

    print("\n--- Test Results ---")
    print(f'Test Loss: {avg_test_loss:.4f}')
    print(f'Test Accuracy (labels): {accuracy_labels:.2f}%')
    print(f'Test Accuracy (large_k): {accuracy_large_k:.2f}%')

    return model, {"labels_map": label_map, "large_k_map": large_k_map}

def solve(query_sql, file_list, threshold_list):
    new_dict = defaultdict(dict)
    for file_path in file_list:
        sql_dict = pickle.load(open(file_path, 'rb'))
        for large_k, s1 in sql_dict.items():
            for i, s2 in s1.items():
                new_dict[i][large_k] = s2

    min_id = min(new_dict.keys())
    max_id = max(new_dict.keys())
    print(min_id, max_id)

    label_dict = {'table_scan' : 0,
                  'sql_1_where_in_with' : 1,
                  'sql_2_where_in_with' : 2,
                  'sql_12_where_in_with' : 3,
                  'sql_1_where_not_in_with' : 4,
                  'sql_2_where_not_in_with' : 5,
                  'sql_12_where_not_in_with' : 6,
                  }
    label_list = ['table_scan', 'sql_1_where_in_with', 'sql_2_where_in_with', 'sql_12_where_in_with', 'sql_1_where_not_in_with', 'sql_2_where_not_in_with', 'sql_12_where_not_in_with']
    w1_list = []
    w2_list = []
    selectivity_list = []
    th_list = []
    labels = []
    large_k_list = []
    ave_ratio = 0
    min_ratio = 1e99
    total_best_t = 0
    total_table_scan_t = 0
    for i in range(min_id, max_id+1):
        w1, w2 = from_mul_col_query_get_weight(query_sql[i][0])
        for threshold in threshold_list:
            best_t = None
            best_method = None
            best_k = None
            for large_k in range(1000, 10001, 1000):
                g = new_dict[i][large_k]
                for k, v in g.items():
                    if v[0] >= threshold:
                        if best_t is None or v[1] < best_t:
                            best_t = v[1]
                            best_method = label_dict[k]
                            best_k = large_k
            w1_list.append(w1)
            w2_list.append(w2)
            selectivity_list.append(query_sql[i][1])
            print(query_sql[i][1])
            th_list.append(threshold)
            labels.append(best_method)
            large_k_list.append(best_k)

            table_scan_t = new_dict[i][1000]['table_scan'][1]
            ratio = best_t / table_scan_t
            ave_ratio += ratio
            min_ratio = min(ratio, min_ratio)
            total_table_scan_t += table_scan_t
            total_best_t += best_t

    ave_ratio /= (max_id - min_id)

    for i, (w1, w2, selectivity, th, label, large_k) in enumerate(zip(w1_list, w2_list, selectivity_list, th_list, labels, large_k_list)):
        print(f'{i:20} {w1:20} {w2:20} {selectivity:20} {th:20} {label:20} {large_k:20}')
    print(total_best_t, total_table_scan_t, total_best_t / total_table_scan_t, ave_ratio, min_ratio)

    large_k_list = [x / 100 - 1 for x in large_k_list]

    trained_model, class_maps = pred(
        w1_list,
        w2_list,
        selectivity_list,
        th_list,
        labels,
        large_k_list,
        split_ratio=(0.25, 0.25, 0.5),
        epochs=20,
        batch_size=64
    )

    print("\nTraining complete.")
    print("Trained Model:", trained_model)
    print("Class Mappings:", class_maps)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    no_satisfied_cnt = 0
    ave_acc = 0
    ave_ratio = 0
    min_ratio = 1e99
    total_best_t = 0
    total_table_scan_t = 0
    labels_map = {v : k for k, v in class_maps['labels_map'].items()}
    large_k_map = {v : k for k, v in class_maps['large_k_map'].items()}
    for i in range(min_id, max_id+1):
        w1, w2 = from_mul_col_query_get_weight(query_sql[i][0])
        selectivity = query_sql[i][1]
        for threshold in threshold_list:
            _input = torch.tensor([[w1, w2, selectivity, threshold]]).float().to(device)
            labels, large_k = trained_model(_input)
            labels = labels.argmax(dim=1).item()
            large_k = large_k.argmax(dim=1).item()
            labels = label_list[int(labels_map[labels])]
            large_k = int(large_k_map[large_k] + 1) * 100

            table_scan_t = new_dict[i][1000]['table_scan'][1]
            acc = new_dict[i][large_k][labels][0]
            ave_acc += acc
            if new_dict[i][large_k][labels][0] >= threshold:
                min_ratio = min(ratio, min_ratio)
                t = new_dict[i][large_k][labels][1]
                ratio = t / table_scan_t
            else:
                no_satisfied_cnt += 1
                ratio = 1
                t = table_scan_t

            ave_ratio += ratio
            total_table_scan_t += table_scan_t
            total_best_t += t

            print(large_k, labels)

    ave_ratio /= (max_id - min_id)
    ave_acc /= (max_id - min_id)

    print('final = ', total_best_t, total_table_scan_t, total_best_t / total_table_scan_t, ave_ratio, min_ratio, ave_acc, no_satisfied_cnt)

if __name__ == '__main__':
    # cursor = get_database_cursor('tpch')
    # table_name = 'part'
    # # N = 10000
    # N = 200
    # query_path = f'query/{N}_hybrid_weight_mul_col_query_sql_[{table_name}]_limit100.pickle'
    # gt_path = f'ground_truth/{N}_hybrid_weight_mul_col_query_ground_truth_[{table_name}]_limit100.pickle'
    # large_k_list = list(range(1000, 10001, 1000))
    # for i in range(N // 100):
    #     start_pos = i * 100
    #     end_pos = start_pos + 100
    #     save_path = f'result/hybrid_weight_mul_col/{N}_hybrid_weight_mul_col_query_result_[{table_name}]___[{start_pos}-{end_pos}]___[more_large_k]___limit100.pickle'
    #     run_sql(cursor, query_path, gt_path, save_path, start_pos, end_pos, large_k_list)


    query_sql_path = 'query/10000_hybrid_weight_mul_col_query_sql_[part]_limit100.pickle'
    query_sql = pickle.load(open(query_sql_path, 'rb'))
    threshold_list = [0.8]
    file_root_path = 'result/hybrid_weight_mul_col'
    file_list = []
    for file_name in os.listdir(file_root_path):
        if file_name.startswith('10000_hybrid_weight_mul_col_query_result_[part]___['):
            file_list.append(os.path.join(file_root_path, file_name))
    solve(query_sql, file_list, threshold_list)