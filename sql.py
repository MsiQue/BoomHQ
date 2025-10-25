from global_info import get_vector_from_sql, get_where_clause_from_sql
from typing import Optional
import math
import re

def extract_table_name(sql_query):
    match = re.search(r'from\s+([a-zA-Z0-9_]+)(?:\s+where|\s|$)', sql_query, re.IGNORECASE)
    if match:
        return match.group(1)
    return None

def _split_conditions(where_clause):
    where_clause = where_clause.upper()
    sub_conditions = where_clause.split(' AND ')
    cnt = len(sub_conditions)
    p = 0
    conditions = []
    while p < cnt:
        if 'BETWEEN' in sub_conditions[p]:
            conditions.append(sub_conditions[p] + ' AND ' + sub_conditions[p + 1])
            p += 2
        else:
            conditions.append(sub_conditions[p])
            p += 1
    return conditions

def _update_range(column_ranges: dict, column: str, lower: Optional[float], upper: Optional[float]):
    current_lower, current_upper = column_ranges.get(column, (None, None))

    if lower is not None:
        new_lower = max(current_lower, lower) if current_lower is not None else lower
    else:
        new_lower = current_lower

    if upper is not None:
        new_upper = min(current_upper, upper) if current_upper is not None else upper
    else:
        new_upper = current_upper

    if new_lower is not None and new_upper is not None and new_lower > new_upper:
        new_lower, new_upper = None, None

    column_ranges[column] = (new_lower, new_upper)

def get_explain_info(cursor, sql):
    explain_info = cursor.execute('explain ' + sql).fetchall()
    # '  ->  Index Scan using aka_title_index__m16__ef_construction64 on aka_title  (cost=1287.62..661806.55 rows=314109 width=12)',)
    info1 = explain_info[1][0].split('cost=')[1].split('..')
    cost1 = float(info1[0])
    info2 = info1[1].split(' rows=')
    cost2 = float(info2[0])
    rows = float(info2[1].split()[0])

    # print(explain_info[1][0])
    # print(cost1, cost2, rows)

    return [cost1, cost2, rows]

def parse_where_clause(where_clause):
    where_clause = where_clause.strip()
    if where_clause.upper().startswith("WHERE "):
        where_clause = where_clause[6:].strip()

    if not where_clause:
        return []

    conditions = _split_conditions(where_clause)
    column_ranges = {}

    for condition in conditions:
        between_match = re.search(r'(\w+)\s+BETWEEN\s+([\d.]+)\s+AND\s+([\d.]+)', condition, re.IGNORECASE)
        if between_match:
            column = between_match.group(1)
            lower = float(between_match.group(2))
            upper = float(between_match.group(3))
            _update_range(column_ranges, column, lower, upper)
            continue

        op_match = re.search(r'(\w+)\s*(>=|>|<=|<|=)\s*([\d.]+)', condition)
        if op_match:
            column = op_match.group(1)
            operator = op_match.group(2)
            value = float(op_match.group(3))

            if operator in ('>', '>='):
                current_lower, current_upper = column_ranges.get(column, (None, None))
                new_lower = max(value, current_lower) if current_lower is not None else value
                _update_range(column_ranges, column, new_lower, current_upper)
            elif operator in ('<', '<='):
                current_lower, current_upper = column_ranges.get(column, (None, None))
                new_upper = min(value, current_upper) if current_upper is not None else value
                _update_range(column_ranges, column, current_lower, new_upper)
            elif operator == '=':
                _update_range(column_ranges, column, value, value)

    return {col : (bounds[0], bounds[1]) for col, bounds in column_ranges.items()}


def encode_where_clause(where_clause, kwargs):
    where_dict = parse_where_clause(where_clause)
    number_column_range_list = kwargs['number_column_range_list']
    interval_cnt = kwargs['interval_cnt']
    unique_vals = [sorted([y[0] if y[0] is not None else -1 for y in x[1]]) for x in number_column_range_list]
    # print(where_dict)
    encoded_row = []
    encoded_row_magnitude = []
    encoded_row_magnitude_product = 1.0
    for i, (column_name, value_cnt, minmax) in enumerate(number_column_range_list):
        min_val, max_val = minmax
        if len(value_cnt) == 1 and value_cnt[0][0] is None:
            continue
            # this column is all none

        min_val = float(min_val)
        max_val = float(max_val)

        if column_name.upper() in where_dict:
            L, R = where_dict[column_name.upper()]
            if L is None:
                L = min_val
            if R is None:
                R = max_val
        else:
            L, R = min_val, max_val

        if min_val == max_val:
            ratio = 1.0
        else:
            ratio = (R - L) / (max_val - min_val)
        encoded_row_magnitude.append(ratio)
        encoded_row_magnitude_product *= ratio

        unique_count = len(unique_vals[i])
        if unique_count < interval_cnt:
            encoded = [0] * unique_count
            # index_L = unique_vals[i].index(L)
            # index_R = unique_vals[i].index(R)
            # print(unique_vals[i], L, R)
            if len(unique_vals[i]) == 1:
                index_L = 0
                index_R = 0
            else:
                index_L = min([i for i, x in enumerate(unique_vals[i]) if L < x])
                index_R = max([i for i, x in enumerate(unique_vals[i]) if x < R])

        else:
            encoded = [0] * interval_cnt
            min_val, max_val = minmax
            min_val = float(min_val)
            max_val = float(max_val)

            def get_pos(number_value):
                if number_value == max_val or number_value == -1:
                    value_pos = interval_cnt - 1
                else:
                    value_pos = math.floor((number_value - min_val) / (max_val - min_val) * interval_cnt)
                return value_pos

            index_L = get_pos(L)
            index_R = get_pos(R)

        if index_L <= index_R:
            for j in range(index_L, index_R + 1):
                encoded[j] = 1
        encoded_row.extend(encoded)

    # print(encoded_row)
    return encoded_row, [encoded_row_magnitude_product] + encoded_row_magnitude

def get_sql_info(sql, kwargs):
    _, v = get_vector_from_sql(sql)
    where_clause = get_where_clause_from_sql(sql)
    s1, s2 = encode_where_clause(where_clause, kwargs)
    pre = get_explain_info(kwargs['cursor'], sql)
    return v, s1, s2, pre

