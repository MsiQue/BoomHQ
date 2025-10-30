import pandas as pd
import numpy as np
import tqdm
from sklearn.cluster import KMeans
from global_info import get_database_cursor

def cluster_and_update(cursor, table_name, vector_column_name, n_cluster):
    cursor.execute(f"SELECT id, {vector_column_name} FROM {table_name}")
    data = cursor.fetchall()

    df = pd.DataFrame(data, columns=['id', vector_column_name])

    vectors = np.array([eval(_) for _ in  df[vector_column_name].to_list()])

    kmeans = KMeans(n_clusters=n_cluster, random_state=0)
    labels = kmeans.fit_predict(vectors)

    df[f'cluster_label_{n_cluster}'] = labels

    cursor.execute(f"alter table {table_name} add column cluster_label_{n_cluster} int")

    for _, row in tqdm.tqdm(df.iterrows()):
        update_query = f"UPDATE {table_name} SET cluster_label_{n_cluster} = %s WHERE id = %s"
        cursor.execute(update_query, (int(row[f'cluster_label_{n_cluster}']), int(row['id'])))

    cursor.connection.commit()

def encode_with_hyperplanes(cursor, table_name, vector_column_name, n_hyperplanes):
    cursor.execute(f"SELECT id, {vector_column_name} FROM {table_name}")
    data = cursor.fetchall()

    df = pd.DataFrame(data, columns=['id', vector_column_name])

    vectors = np.array([eval(_) for _ in  df[vector_column_name].to_list()])

    if vectors.size == 0:
        return

    vector_dim = vectors.shape[1]

    hyperplanes = np.random.randn(n_hyperplanes, vector_dim)

    binary_codes = (np.dot(vectors, hyperplanes.T) > 0).astype(int)

    encoded_values = [int("".join(map(str, code)), 2) for code in binary_codes]
    print(encoded_values)

    df[f'hyperplane_code_{n_hyperplanes}'] = encoded_values

    cursor.execute(f"alter table {table_name} add column hyperplane_code_{n_hyperplanes} int")

    for _, row in tqdm.tqdm(df.iterrows()):
        update_query = f"UPDATE {table_name} SET hyperplane_code_{n_hyperplanes} = %s WHERE id = %s"
        cursor.execute(update_query, (int(row[f'hyperplane_code_{n_hyperplanes}']), int(row['id'])))

    cursor.connection.commit()

def calculate_distance_sum(cursor, table_name, vector_column_name, n_points):
    cursor.execute(f"SELECT id, {vector_column_name} FROM {table_name}")
    data = cursor.fetchall()

    if not data:
        return

    df = pd.DataFrame(data, columns=['id', vector_column_name])

    vectors = np.array([eval(_) for _ in  df[vector_column_name].to_list()])

    vector_dim = vectors.shape[1]

    random_points = np.random.randn(n_points, vector_dim)

    distance_sums = np.sum(np.linalg.norm(vectors[:, np.newaxis, :] - random_points, axis=2), axis=1)

    df[f'distance_sum_{n_points}'] = distance_sums

    cursor.execute(f"alter table {table_name} add column distance_sum_{n_points} real")

    for _, row in tqdm.tqdm(df.iterrows()):
        update_query = f"UPDATE {table_name} SET distance_sum_{n_points} = %s WHERE id = %s"
        cursor.execute(update_query, (float(row[f'distance_sum_{n_points}']), int(row['id'])))

    cursor.connection.commit()

if __name__ == '__main__':
    cursor = get_database_cursor('ann_benchmark')
    # cluster_and_update(cursor, 'sift', 'vector1', 5)
    # encode_with_hyperplanes(cursor, 'sift', 'vector1', 5)
    # calculate_distance_sum(cursor, 'sift', 'vector1', 5)

    cluster_and_update(cursor, 'glove', 'vector1', 5)
    encode_with_hyperplanes(cursor, 'glove', 'vector1', 5)
    calculate_distance_sum(cursor, 'glove', 'vector1', 5)