import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pickle
import os
import numpy as np
import random
from sklearn.model_selection import train_test_split
import re

class query_encoder(nn.Module):
    def __init__(self, n_vector_cols, vector_dims_list, n_scalar_cols, vec_embed_dim=32, scalar_embed_dim=16,
                 latent_dim=128):
        super(query_encoder, self).__init__()
        self.n_vector_cols = n_vector_cols
        self.n_scalar_cols = n_scalar_cols
        self.vector_dims_list = vector_dims_list

        self.vector_encoders = nn.ModuleList()
        total_vec_encoded_dim = 0

        for dim in self.vector_dims_list:
            encoder = nn.Sequential(
                nn.Linear(dim, (dim + vec_embed_dim) // 2),
                nn.ReLU(True),
                nn.Linear((dim + vec_embed_dim) // 2, vec_embed_dim)
            )
            self.vector_encoders.append(encoder)
            total_vec_encoded_dim += vec_embed_dim

        self.scalar_encoder = nn.Sequential(
            nn.Linear(n_scalar_cols, (n_scalar_cols + scalar_embed_dim) * 2),
            nn.ReLU(True),
            nn.Linear((n_scalar_cols + scalar_embed_dim) * 2, scalar_embed_dim)
        )

        total_encoded_dim = total_vec_encoded_dim + scalar_embed_dim

        self.prediction_head = nn.Sequential(
            nn.Linear(total_encoded_dim, latent_dim),
            nn.ReLU(True),
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(True),
            nn.Linear(latent_dim // 2, 1)
        )

    def forward(self, vectors_list, scalars_tensor):
        encoded_vectors = []
        for i in range(self.n_vector_cols):
            encoded_vectors.append(self.vector_encoders[i](vectors_list[i]))

        encoded_scalars = self.scalar_encoder(scalars_tensor)

        combined_rep = torch.cat(encoded_vectors + [encoded_scalars], dim=1)

        prediction = self.prediction_head(combined_rep)
        return prediction


def load_and_process_data(pickle_file_path, vector_keys, scalar_keys):
    try:
        with open(pickle_file_path, 'rb') as f:
            queries = pickle.load(f)
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        return None

    all_vectors = [[] for _ in vector_keys]
    all_scalars = []
    all_targets = []

    for q in queries:
        try:
            all_targets.append(q['target'])
            all_scalars.append([q['scalars'][key] for key in scalar_keys])
            for i, key in enumerate(vector_keys):
                all_vectors[i].append(q['vectors'][key])
        except KeyError as e:
            print(f"Data missing key {e} in query {q.get('query_id')}, skipping.")
            continue

    if not all_targets:
        print("No valid data loaded.")
        return None

    try:
        vectors_tensors = [torch.tensor(v, dtype=torch.float32) for v in all_vectors]
        scalars_tensor = torch.tensor(all_scalars, dtype=torch.float32)
        targets_tensor = torch.tensor(all_targets, dtype=torch.float32).view(-1, 1)

        vector_dims_list = [t.shape[1] for t in vectors_tensors]
    except Exception as e:
        print(f"Error converting data to tensors: {e}")
        return None

    return vectors_tensors, scalars_tensor, targets_tensor, vector_dims_list


def train_query_encoder(
        pickle_file_path,
        model_save_path,
        vector_column_keys,
        scalar_column_keys,
        model_params=None,
        num_epochs=20,
        batch_size=32,
        learning_rate=1e-3,
        test_split_ratio=0.15,
        eval_split_ratio=0.15
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    load_result = load_and_process_data(pickle_file_path, vector_column_keys, scalar_column_keys)

    if load_result is None:
        return

    vectors_tensors, scalars_tensor, targets_tensor, vector_dims_list = load_result

    n_vector_cols = len(vector_column_keys)
    n_scalar_cols = len(scalar_column_keys)

    indices = list(range(len(targets_tensor)))

    train_val_indices, test_indices = train_test_split(
        indices, test_size=test_split_ratio, random_state=42
    )

    relative_eval_ratio = eval_split_ratio / (1 - test_split_ratio)

    train_indices, eval_indices = train_test_split(
        train_val_indices, test_size=relative_eval_ratio, random_state=42
    )

    all_tensors = vectors_tensors + [scalars_tensor, targets_tensor]

    train_tensors = [t[train_indices] for t in all_tensors]
    eval_tensors = [t[eval_indices] for t in all_tensors]
    test_tensors = [t[test_indices] for t in all_tensors]

    train_dataset = TensorDataset(*train_tensors)
    eval_dataset = TensorDataset(*eval_tensors)
    test_dataset = TensorDataset(*test_tensors)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Data loaded: {len(train_indices)} train, {len(eval_indices)} eval, {len(test_indices)} test.")

    if model_params is None:
        model_params = {
            "vec_embed_dim": 32,
            "scalar_embed_dim": 16,
            "latent_dim": 64
        }

    model = query_encoder(
        n_vector_cols=n_vector_cols,
        vector_dims_list=vector_dims_list,
        n_scalar_cols=n_scalar_cols,
        **model_params
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    best_eval_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = [b.to(device) for b in batch]
            *batch_vectors, batch_scalars, batch_targets = batch

            predictions = model(batch_vectors, batch_scalars)
            loss = loss_fn(predictions, batch_targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        eval_loss = 0.0
        with torch.no_grad():
            for batch in eval_loader:
                batch = [b.to(device) for b in batch]
                *batch_vectors, batch_scalars, batch_targets = batch

                predictions = model(batch_vectors, batch_scalars)
                loss = loss_fn(predictions, batch_targets)
                eval_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_eval_loss = eval_loss / len(eval_loader)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.6f}, Eval Loss: {avg_eval_loss:.6f}")

        if avg_eval_loss < best_eval_loss:
            best_eval_loss = avg_eval_loss
            try:
                torch.save(model.state_dict(), model_save_path)
                print(f"New best model saved to {model_save_path}")
            except Exception as e:
                print(f"Error saving model: {e}")

    print("Training finished.")

    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch in test_loader:
            batch = [b.to(device) for b in batch]
            *batch_vectors, batch_scalars, batch_targets = batch

            predictions = model(batch_vectors, batch_scalars)
            loss = loss_fn(predictions, batch_targets)
            test_loss += loss.item()

    avg_test_loss = test_loss / len(test_loader)
    print(f"Final Test Loss on best model: {avg_test_loss:.6f}")


def rewrite_sql_query(original_sql, scalar_column_list, vector_column_name, vector):
    pattern = re.compile(
        r"^(SELECT .*? FROM .*?) WHERE (.*?) (ORDER BY .*?) (LIMIT .*?)$",
        re.IGNORECASE | re.DOTALL
    )

    match = pattern.match(original_sql.strip())

    if not match:
        return "Error: SQL a format not recognized."

    select_from_part = match.group(1)
    table_name = select_from_part.split("FROM")[1]

    where_condition = match.group(2)
    orderby_part = match.group(3)
    limit_part = match.group(4)

    # inner_query = f"SELECT {','.join(scalar_column_list)} FROM {table_name} ORDER BY {vector_column_name} <-> '{vector}' {limit_part[:-1]}"
    inner_query = f"SELECT * FROM {table_name} ORDER BY {vector_column_name} <-> '{vector}' {limit_part[:-1]}"
    print(inner_query)

    rewritten_sql = f"SELECT count(*) FROM ({inner_query}) AS subquery WHERE {where_condition}"

    return rewritten_sql

def neighborhood_probe(cursor, original_sql, scalar_column_list, vector_column_name, vector):
    rewritten_sql = rewrite_sql_query(original_sql, scalar_column_list, vector_column_name, vector)
    # print(rewritten_sql)
    cursor.execute(rewritten_sql)
    res = cursor.fetchone()[0]
    limit_k = int(original_sql.split("LIMIT ")[1][:-1])
    # print('limit  ', res, limit_k)
    return res / limit_k

if __name__ == '__main__':
    DUMMY_PICKLE_FILE = "dummy_queries.pkl"
    MODEL_SAVE_PATH = "./model/query_encoder.pth"

    VEC_KEYS = ["vec_1", "vec_2"]
    SCA_KEYS = ["p_size", "p_retailprice"]
    VEC_DIMS = [64, 32]

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    MODEL_PARAMS = {
        "vec_embed_dim": 16,
        "scalar_embed_dim": 8,
        "latent_dim": 32
    }

    train_query_encoder(
        pickle_file_path=DUMMY_PICKLE_FILE,
        model_save_path=MODEL_SAVE_PATH,
        vector_column_keys=VEC_KEYS,
        scalar_column_keys=SCA_KEYS,
        model_params=MODEL_PARAMS,
        num_epochs=10,
        batch_size=64
    )