import random
import re
import time
import pickle
import numpy as np
from global_info import get_database_cursor, query_topk_sql, calc_selectivity, get_range_list, get_number_column_list, table_name_to_vector_column_name, execute_sql_with_args_explain_returned

def extract_nth_distance_query(sql: str, n: int, return_component = False) -> str:
    order_by_pattern = r'ORDER\s+BY\s+(.*?)(?:\s+ASC|\s+DESC)?\s*(?:LIMIT|;|$)'
    order_by_match = re.search(order_by_pattern, sql, re.IGNORECASE | re.DOTALL)

    if not order_by_match:
        raise ValueError("'ORDER BY' not found")

    order_by_target = order_by_match.group(1).strip()

    full_expression_to_replace = ""
    is_alias = '+' not in order_by_target and '<' not in order_by_target

    if is_alias:
        alias = order_by_target
        select_list_match = re.search(r'SELECT\s+(.*?)\s+FROM', sql, re.IGNORECASE | re.DOTALL)
        if not select_list_match:
            raise ValueError("无效的SQL: 无法解析SELECT列表。")

        select_list_str = select_list_match.group(1)
        alias_pattern = r'\s+AS\s+' + re.escape(alias)
        alias_match_in_select = re.search(alias_pattern, select_list_str, re.IGNORECASE)

        if not alias_match_in_select:
            raise ValueError(f"无法在SELECT列表中找到别名 '{alias}' 的定义。")

        pre_alias_str = select_list_str[:alias_match_in_select.start()]
        comma_index = pre_alias_str.rfind(',')
        expression_start_index = comma_index + 1
        full_expression_to_replace = select_list_str[expression_start_index:alias_match_in_select.start()].strip()
    else:
        full_expression_to_replace = order_by_target

    if '+' not in full_expression_to_replace:
        raise ValueError(f"定位到的表达式 '{full_expression_to_replace}' 似乎不是一个加和表达式。")

    components = [comp.strip().split('*')[1] if '*' in comp.strip() else comp.strip() for comp in full_expression_to_replace.split('+')]

    if not (1 <= n <= len(components)):
        raise ValueError(f"参数 'n' 必须在 1 到 {len(components)} 之间，但输入为 {n}。")

    nth_component = components[n - 1]

    new_sql = sql.replace(full_expression_to_replace, nth_component, 1)

    if return_component:
        return new_sql, components
    else:
        return new_sql

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

def get_query_vector(cursor, table_name, vec_column_name):
    query = f"SELECT {vec_column_name} FROM {table_name} ORDER BY random() LIMIT 1;"
    cursor.execute(query)
    result = cursor.fetchone()
    v_1 = np.array(eval(result[0]))
    v_2 = (0.1 * np.random.uniform(low=-1, high=1, size=768))
    v = (v_1 + v_2).tolist()
    return v

def mul_col_query_topk_sql(column_name, table_name, vector_column_1, vector_1, vector_column_2, vector_2, limit, explain = False, opt = '<->'):
    # single vector column
    prefix = 'EXPLAIN ' if explain else ''
    # return prefix + f"SELECT {column_name} FROM {table_name} ORDER BY ({vector_column_1} {opt} '%s') + ({vector_column_2} {opt} '%s') LIMIT {limit} ;" % (vector_1, vector_2)
    return prefix + f"SELECT {column_name} FROM {table_name} ORDER BY 0.7 * ({vector_column_1} {opt} '%s') + 0.3 * ({vector_column_2} {opt} '%s') LIMIT {limit} ;" % (vector_1, vector_2)
    # return prefix + f"SELECT {column_name} FROM {table_name} ORDER BY 0.0001 * ({vector_column_1} {opt} '%s') + ({vector_column_2} {opt} '%s') LIMIT {limit} ;" % (vector_1, vector_2)
    # return prefix + f"SELECT {column_name} FROM {table_name} ORDER BY 0 * ({vector_column_1} {opt} '%s') + ({vector_column_2} {opt} '%s') LIMIT {limit} ;" % (vector_1, vector_2)
    # return prefix + f"SELECT {column_name} FROM {table_name} ORDER BY ({vector_column_1} {opt} '%s') LIMIT {limit} ;" % (vector_1)
    # return prefix + f"SELECT {column_name} FROM {table_name} ORDER BY ({vector_column_2} {opt} '%s') LIMIT {limit} ;" % (vector_2)
    # return prefix + f"SELECT {column_name} FROM {table_name} ORDER BY 0.1 * ({vector_column_2} {opt} '%s') LIMIT {limit} ;" % (vector_2)
    # return prefix + f"SELECT {column_name} FROM {table_name} ORDER BY 0 * ({vector_column_2} {opt} '%s') LIMIT {limit} ;" % (vector_2)

def print_explain(cursor, sql):
    res, t, explain_returned = execute_sql_with_args_explain_returned(cursor, sql, [], 'explain (ANALYZE, BUFFERS)')
    for _ in explain_returned:
        print(_)

def loop_run(cursor, table_name, vec_column_name_1, vec_column_name_2, limitk):
    while True:
        v1 = get_query_vector(cursor, table_name, vec_column_name_1)
        v2 = get_query_vector(cursor, table_name, vec_column_name_2)
        target_sql = mul_col_query_topk_sql('id', table_name, vec_column_name_1, v1, vec_column_name_2, v2, limitk)
        # res, t, explain_returned = execute_sql_with_args_explain_returned(cursor, target_sql, ['enable_seqscan = off'], 'explain ')
        print(target_sql)
        print(extract_nth_distance_query(target_sql, 1))
        print(extract_nth_distance_query(target_sql, 2))
        print_explain(cursor, target_sql)
        print_explain(cursor, extract_nth_distance_query(target_sql, 1))
        print('-' * 100)

def gen_mul_col_query_sql(cursor, table_name, vec_column_name_1, vec_column_name_2, N, limitk, connective_list = ['AND', 'OR'], threshold = 0.05, file_name_suffix = ''):
    number_column = get_number_column_list(cursor, table_name)
    print(number_column)
    L = []
    for c in number_column:
        L.append((c, get_range_list(cursor, table_name, c)))
    print(L)

    query_sql_list = []
    for i in range(N):
        v1 = get_query_vector(cursor, table_name, vec_column_name_1)
        v2 = get_query_vector(cursor, table_name, vec_column_name_2)
        target_sql = mul_col_query_topk_sql('id', table_name, vec_column_name_1, v1, vec_column_name_2, v2, limitk)
        target_sql_selectivity = calc_selectivity(cursor, target_sql)

        print(i)
        print(target_sql)
        print(target_sql_selectivity)
        query_sql_list.append((target_sql, target_sql_selectivity))

    filepath = f'query/{N}_mul_col_query_sql_[{table_name}]_limit{limitk}{file_name_suffix}.pickle'
    pickle.dump(query_sql_list, open(filepath, 'wb'))

def solve(cursor):
    sql_list = pickle.load(open('query/100_mul_col_query_sql_[part]_limit100.pickle', 'rb'))
    for sql in sql_list:
        res, t, explain_returned = execute_sql_with_args_explain_returned(cursor, sql[0], [], 'explain ')
        for _ in explain_returned:
            print(_)
        print('-' * 100)

import psycopg
import numpy as np

def combine_vector_columns(cur, table_name, col1, col2, new_col):
    sql_function = """
    CREATE OR REPLACE FUNCTION concatenate_vectors(v1 vector, v2 vector)
    RETURNS vector
    LANGUAGE plpgsql
    IMMUTABLE
    AS $$
    DECLARE
        arr1 float4[];
        arr2 float4[];
        new_arr float4[];
    BEGIN
        arr1 := ARRAY(SELECT unnest(v1::float4[]));
        arr2 := ARRAY(SELECT unnest(v2::float4[]));
        new_arr := arr1 || arr2;
        RETURN ('[' || array_to_string(new_arr, ',') || ']')::vector;
    END;
    $$;
    """

    cur.execute(sql_function)

    row = cur.execute(f"SELECT vector_dims({col1}), vector_dims({col2}) FROM {table_name} LIMIT 1").fetchone()
    if not row:
        print(f"Table '{table_name}' is empty or does not exist.")
        return

    dim1, dim2 = row
    new_dim = dim1 + dim2

    alter_query = f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS {new_col} vector({new_dim})"
    cur.execute(alter_query)

    update_query = f"UPDATE {table_name} SET {new_col} = concatenate_vectors({col1}, {col2})"
    cur.execute(update_query)


if __name__ == '__main__':
    cursor = get_database_cursor('tpch')

    # combine_vector_columns(cursor, 'test_items', 'feature1', 'feature2', 'f12')
    combine_vector_columns(cursor, 'part', 'p_comment_vector', 'p_name_vector', 'vec_2_combine')

    # loop_run(cursor, 'part', 'p_comment_vector', 'p_name_vector', 100)

    # gen_mul_col_query_sql(cursor, 'part', 'p_comment_vector', 'p_name_vector', 100, 100)
    # solve(cursor)

    # # for table_name, vector_column_name in table_name_to_vector_column_name.items():
    # #     if 'comment' in vector_column_name:
    # #         print(table_name, vector_column_name)
    # #         gen_query_sql(cursor, table_name, vector_column_name, 10000, 100)
    #
    # for table_name, vector_column_name in table_name_to_vector_column_name.items():
    #
    #     if table_name == 'aka_title':
    #         continue
    #
    #     if 'comment' in vector_column_name:
    #         cursor = get_database_cursor('tpch')
    #     else:
    #         cursor = get_database_cursor('imdb')
    #     print(table_name, vector_column_name)
    #     gen_query_sql(cursor, table_name, vector_column_name, 1000, 100)