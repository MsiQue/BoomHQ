import psycopg
import re
import time

large_k_list = [100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
ef_search_list = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
iterative_scan_list = ['off', 'strict_order', 'relaxed_order']
table_name_to_vector_column_name = {
    'aka_title' : 'title_vector',
    'title' : 'title_vector',
    'aka_name' : 'name_vector',
    'supplier' : 's_comment_vector',
    'part' : 'p_comment_vector',
    'partsupp' : 'ps_comment_vector',
    'orders' : 'o_comment_vector',
    'lineitem' : 'l_comment_vector'
}

def getCursor():
    conn = psycopg.connect(conninfo='postgresql://ogb:@localhost:5454/imdb', autocommit=True)
    cursor = conn.cursor()
    return cursor

def get_database_cursor(database):
    conn = psycopg.connect(conninfo=f'postgresql://ogb:@localhost:5454/{database}', autocommit=True)
    cursor = conn.cursor()
    return cursor

def query_topk_sql(column_name, table_name, where_clause, vector_column, vector, limit, explain = False, opt = '<->'):
    # single vector column
    prefix = 'EXPLAIN ' if explain else ''
    return prefix + f"SELECT {column_name} FROM {table_name} {where_clause} ORDER BY {vector_column} {opt} '%s' LIMIT {limit} ;" % vector

def calc_selectivity(cursor, s):
    if 'where' not in s.lower():
        return 1
    match = re.search(r"FROM\s+(\w+)\s+WHERE\s+(.*?)(\s+ORDER BY|\s+LIMIT|;|$)", s, re.IGNORECASE)
    if not match:
        raise ValueError("invalid SQL")

    table_name = match.group(1)
    where_condition = match.group(2)

    sql_1 = f"SELECT count(*) FROM {table_name} WHERE {where_condition};"
    cursor.execute(sql_1)
    n1 = cursor.fetchone()[0]

    sql_2 = f"SELECT count(*) FROM {table_name};"
    cursor.execute(sql_2)
    n2 = cursor.fetchone()[0]

    return n1 / n2

def execute_sql_with_args_explain_returned(cursor, sql, arg_list, explain_prefix = ''):
    t1 = time.time()
    cursor.execute('BEGIN;')
    for arg in arg_list:
        cursor.execute(f'SET LOCAL {arg} ;')
    explain_returned = ''
    if len(explain_prefix) > 1:
        cursor.execute(explain_prefix + ' ' + sql)
        explain_returned = cursor.fetchall()
    cursor.execute(sql)
    res = cursor.fetchall()
    cursor.execute('COMMIT;')
    # print(res, time.time() - t1)
    return res, time.time() - t1, explain_returned

def convert_to_list(L):
    return [_[0] for _ in L]

def calc(result, ground_truth):
    return len(set(ground_truth) & set(result)) / len(ground_truth)

def get_range_list(cursor, table_name, column_name):
    sql = f'select max({column_name}) from {table_name}'
    cursor.execute(sql)
    _max = cursor.fetchall()[0][0]
    sql = f'select min({column_name}) from {table_name}'
    cursor.execute(sql)
    _min = cursor.fetchall()[0][0]
    return _min, _max

def get_number_column_list(cursor, table_name):
    query = """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = %s
          AND (data_type IN ('integer', 'smallint', 'bigint', 'real', 'double precision', 'numeric'));
    """
    cursor.execute(query, (table_name,))
    results = cursor.fetchall()
    column_list = [row[0] for row in results]
    return column_list

def column_diff_value_cnt(cursor, table_name, column_name):
    query = f"SELECT {column_name}, COUNT(*) FROM {table_name} GROUP BY {column_name}"
    cursor.execute(query)
    results = cursor.fetchall()
    return results

def get_vector_from_sql(sql):
    vector_column_name = ""
    vector = ""
    order_by_match = re.search(r"ORDER BY (\w[\w ]*) <-> '(\[.*?\])'", sql)
    if order_by_match:
        vector_column_name = order_by_match.group(1)
        vector = eval(order_by_match.group(2))
    else:
        print("No vector")

    return vector_column_name, vector

def get_where_clause_from_sql(sql):
    where_clause = ""
    where_match = re.search(r"WHERE (.*) ORDER BY", sql)
    if where_match:
        where_clause = where_match.group(1)
    else:
        print("No where clause")
    return where_clause

def get_large_k_sql(sql, large_k):
    k = int(sql.lower().split('limit')[1][:-1])

    table_name = ""
    table_name_match = re.search(r"FROM\s+(\w+)", sql, re.IGNORECASE)
    if table_name_match:
        table_name = table_name_match.group(1)
    else:
        print("No table name match")

    where_clause = ""
    where_match = re.search(r"WHERE (.*) ORDER BY", sql)
    if where_match:
        where_clause = "WHERE " + where_match.group(1)
    else:
        print("No where clause")

    vector_column_name = ""
    vector = ""
    order_by_match = re.search(r"ORDER BY (\w[\w ]*) <-> '(\[.*?\])'", sql)
    if order_by_match:
        vector_column_name = order_by_match.group(1)
        vector = eval(order_by_match.group(2))
    else:
        print("No vector")

    # def get_sql(column_name, table_name, where_clause, limit):
    #     return f"SELECT {column_name} FROM {table_name} {where_clause} ORDER BY dist LIMIT {limit} ;"

    # sub_sql = get_sql(f"id, {vector_column_name} <-> '%s' as dist" % vector, table_name, where_clause, large_k)
    sub_sql = f"SELECT id, {vector_column_name} <-> '%s' as dist FROM {table_name} {where_clause} ORDER BY dist LIMIT {large_k} ;" % vector

    sql = f'WITH filtered AS MATERIALIZED ({sub_sql[:-1]}) SELECT id FROM filtered ORDER BY dist LIMIT {k};'
    return sql