import random
import time
import pickle
import numpy as np
from global_info import get_database_cursor, query_topk_sql, calc_selectivity, get_range_list, get_number_column_list, table_name_to_vector_column_name

def gen_where_clause(column_info, connective_list):
    selected_columns = random.sample(column_info, random.randint(1, len(column_info)))
    predicates = []
    for col, (lower, upper) in selected_columns:
        if lower is None or upper is None:
            continue
        lower = float(lower)
        upper = float(upper)
        if random.random() < 0.5:
            single_value = random.uniform(lower, upper)
            op = random.choice(['>', '<', '>=', '<='])
            predicate = f"{col} {op} {single_value}"
        else:
            lower_val = random.uniform(lower, upper)
            upper_val = random.uniform(lower_val, upper)
            predicate = f"{col} BETWEEN {lower_val} AND {upper_val}"
        predicates.append(predicate)
    if not predicates:
        return ""
    connective = random.choice(connective_list)
    return "WHERE " + f" {connective} ".join(predicates)

def gen_query_sql(cursor, table_name, vec_column_name, N, limitk, connective_list = ['AND', 'OR'], threshold = 0.05, file_name_suffix = ''):
    number_column = get_number_column_list(cursor, table_name)
    print(number_column)
    L = []
    for c in number_column:
        L.append((c, get_range_list(cursor, table_name, c)))
    print(L)

    query_sql_list = []
    for i in range(N):
        query = f"SELECT {vec_column_name} FROM {table_name} ORDER BY random() LIMIT 1;"
        cursor.execute(query)
        result = cursor.fetchone()
        v_1 = np.array(eval(result[0]))
        v_2 = (0.1 * np.random.uniform(low=-1, high=1, size=768))
        v = (v_1 + v_2).tolist()
        # print(v_2.tolist())
        # print(v)
        target_sql = ''
        target_sql_selectivity = 0
        while True:
            where_clause = gen_where_clause(L, connective_list)
            if len(where_clause) == 0:
                continue
            sql = query_topk_sql('id', table_name, where_clause, vec_column_name, v, limitk)
            selectivity = calc_selectivity(cursor, sql)
            if selectivity > threshold:
                target_sql = sql
                target_sql_selectivity = selectivity
                break
        print(i)
        print(target_sql)
        print(target_sql_selectivity)
        query_sql_list.append((target_sql, target_sql_selectivity))

    filepath = f'query/{N}_query_sql_[{table_name}]_limit{limitk}{file_name_suffix}.pickle'
    pickle.dump(query_sql_list, open(filepath, 'wb'))

if __name__ == '__main__':
    for table_name, vector_column_name in table_name_to_vector_column_name.items():
        if 'comment' in vector_column_name:
            cursor = get_database_cursor('tpch')
        else:
            cursor = get_database_cursor('imdb')
        print(table_name, vector_column_name)
        gen_query_sql(cursor, table_name, vector_column_name, 1000, 100)