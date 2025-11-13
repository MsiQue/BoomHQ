import re
from typing import List, Tuple, Optional

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


# def _split_conditions(where_clause):
#     conditions = []
#     current = []
#     paren_count = 0
#
#     for char in where_clause:
#         if char == '(':
#             paren_count += 1
#         elif char == ')':
#             paren_count -= 1
#
#         if paren_count == 0 and char.upper() in ('A', 'O'):
#             remaining = where_clause[len(''.join(current)) + len(conditions):]
#             if remaining.upper().startswith('AND '):
#                 conditions.append(''.join(current).strip())
#                 current = []
#                 where_clause = remaining[4:]
#                 continue
#             elif remaining.upper().startswith('OR '):
#                 conditions.append(''.join(current).strip())
#                 current = []
#                 where_clause = remaining[3:]
#                 continue
#
#         current.append(char)
#
#     if current:
#         conditions.append(''.join(current).strip())
#
#     return conditions

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