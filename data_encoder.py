import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import psycopg2
import os
from global_info import get_database_cursor


class data_encoder(nn.Module):
    def __init__(self, n_vector_cols, vector_dims_list, n_scalar_cols, trainable_out_dim, frozen_out_dim, latent_dim):
        super(data_encoder, self).__init__()
        self.n_vector_cols = n_vector_cols
        self.n_scalar_cols = n_scalar_cols
        self.vector_dims_list = vector_dims_list

        self.trainable_nets = nn.ModuleList()
        self.frozen_nets = nn.ModuleList()
        self.frozen_regressors = nn.ModuleList()

        for i in range(n_vector_cols):
            dim = self.vector_dims_list[i]
            self.trainable_nets.append(self._create_mlp(dim, trainable_out_dim))

        for i in range(n_vector_cols):
            dim = self.vector_dims_list[i]
            for j in range(n_scalar_cols):
                self.frozen_nets.append(self._create_mlp(dim, frozen_out_dim))
                self.frozen_regressors.append(nn.Linear(frozen_out_dim, 1))

        self.total_input_dim = n_vector_cols * (trainable_out_dim + n_scalar_cols * frozen_out_dim) + n_scalar_cols

        self.encoder_ae = nn.Sequential(
            nn.Linear(self.total_input_dim, self.total_input_dim // 2),
            nn.ReLU(True),
            nn.Linear(self.total_input_dim // 2, latent_dim)
        )

        self.decoder_ae = nn.Sequential(
            nn.Linear(latent_dim, self.total_input_dim // 2),
            nn.ReLU(True),
            nn.Linear(self.total_input_dim // 2, self.total_input_dim)
        )

    def _create_mlp(self, input_dim, output_dim):
        return nn.Sequential(
            nn.Linear(input_dim, (input_dim + output_dim) // 2),
            nn.ReLU(True),
            nn.Linear((input_dim + output_dim) // 2, output_dim)
        )

    def encode_input(self, vectors_list, scalars_list):
        all_reps = []
        for i in range(self.n_vector_cols):
            v = vectors_list[i]
            t_rep = self.trainable_nets[i](v)

            f_reps = []
            for j in range(self.n_scalar_cols):
                idx = i * self.n_scalar_cols + j
                f_rep = self.frozen_nets[idx](v)
                f_reps.append(f_rep)

            combined = torch.cat([t_rep] + f_reps, dim=1)
            all_reps.append(combined)

        full_input = torch.cat(all_reps + scalars_list, dim=1)
        return full_input

    def forward(self, vectors_list, scalars_list):
        full_input = self.encode_input(vectors_list, scalars_list)
        latent = self.encoder_ae(full_input)
        reconstructed = self.decoder_ae(latent)
        return reconstructed, full_input

    def get_reconstruction_error(self, vectors_list, scalars_list):
        self.eval()
        with torch.no_grad():
            reconstructed, original_input = self.forward(vectors_list, scalars_list)
            error = torch.nn.functional.mse_loss(reconstructed, original_input, reduction='none')
            return torch.mean(error, dim=1)


def train_data_encoder(
        id_L,
        id_R,
        model_load_path,
        model_save_path,
        vector_column_list,
        scalar_column_list,
        ratio,
        cur,
        table_name,
        num_epochs=10,
        batch_size=32,
        learning_rate=1e-3,
        model_params=None,
        num_pretrain_epochs=5
):
    try:
        all_cols_list = vector_column_list + scalar_column_list
        all_cols_str = ", ".join(all_cols_list)

        query = f"SELECT {all_cols_str} FROM {table_name} WHERE id >= %s AND id < %s AND random() < %s"

        cur.execute(query, (id_L, id_R, ratio))
        rows = cur.fetchall()

    except Exception as e:
        print(f"Database error: {e}")
        return None

    if not rows:
        print("No data sampled.")
        return None

    n_vector_cols = len(vector_column_list)
    n_scalar_cols = len(scalar_column_list)

    vector_data = [[] for _ in range(n_vector_cols)]
    scalar_data = [[] for _ in range(n_scalar_cols)]

    for row in rows:
        for i in range(n_vector_cols):
            vector_data[i].append(row[i])
        for j in range(n_scalar_cols):
            scalar_data[j].append(row[n_vector_cols + j])

    try:
        vector_dims_list = [len(eval(rows[0][i])) for i in range(n_vector_cols)]
    except Exception as e:
        print(f"Error determining vector dimensions from first row: {e}")
        return None

    vectors_tensors = [torch.tensor(np.array([eval(row) for row in d]), dtype=torch.float32) for d in vector_data]
    scalars_tensors = [torch.tensor(d, dtype=torch.float32).view(-1, 1) for d in scalar_data]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_params is None:
        model_params = {
            "trainable_out_dim": 64,
            "frozen_out_dim": 16,
            "latent_dim": 128
        }

    model = data_encoder(
        n_vector_cols=n_vector_cols,
        vector_dims_list=vector_dims_list,
        n_scalar_cols=n_scalar_cols,
        **model_params
    )

    if model_load_path and os.path.exists(model_load_path):
        try:
            model.load_state_dict(torch.load(model_load_path, map_location=device))
            print(f"Loaded model from {model_load_path}")
            for param in model.frozen_nets.parameters():
                param.requires_grad = False
            for param in model.frozen_regressors.parameters():
                param.requires_grad = False

        except Exception as e:
            print(f"Could not load model: {e}. Training from scratch.")
            model_load_path = None
    else:
        print("No model found or path not provided. Training from scratch.")
        model_load_path = None

    model.to(device)

    if model_load_path is None:
        print("Starting pre-training of frozen networks...")
        pretrain_loss_fn = nn.MSELoss()

        for i in range(n_vector_cols):
            for j in range(n_scalar_cols):
                k = i * n_scalar_cols + j

                pretrain_net = model.frozen_nets[k]
                pretrain_head = model.frozen_regressors[k]

                pretrain_params = list(pretrain_net.parameters()) + list(pretrain_head.parameters())
                pretrain_optimizer = optim.Adam(pretrain_params, lr=learning_rate)

                pretrain_dataset = TensorDataset(vectors_tensors[i].to(device), scalars_tensors[j].to(device))
                pretrain_loader = DataLoader(pretrain_dataset, batch_size=batch_size, shuffle=True)

                for epoch in range(num_pretrain_epochs):
                    for x_batch, y_batch in pretrain_loader:
                        pred = pretrain_head(pretrain_net(x_batch))
                        loss = pretrain_loss_fn(pred, y_batch)

                        pretrain_optimizer.zero_grad()
                        loss.backward()
                        pretrain_optimizer.step()

        print("Pre-training finished. Freezing networks.")
        for param in model.frozen_nets.parameters():
            param.requires_grad = False
        for param in model.frozen_regressors.parameters():
            param.requires_grad = False

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    loss_fn = nn.MSELoss()

    dataset = TensorDataset(*vectors_tensors, *scalars_tensors)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for i, batch_data in enumerate(dataloader):
            batch_vectors_raw = batch_data[:n_vector_cols]
            batch_scalars_raw = batch_data[n_vector_cols:]

            batch_vectors = [b.to(device) for b in batch_vectors_raw]
            batch_scalars = [s.to(device) for s in batch_scalars_raw]

            reconstructed, original_input = model(batch_vectors, batch_scalars)

            loss = loss_fn(reconstructed, original_input)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(dataloader):.6f}")

    try:
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")
    except Exception as e:
        print(f"Error saving model: {e}")

    return {"vector_dims_list": vector_dims_list}


if __name__ == '__main__':

    cursor = get_database_cursor('tpch')

    TABLE_NAME = "part"

    VEC_COLS = ["p_comment_vector", "p_name_vector"]
    SCA_COLS = ["p_size", "p_retailprice"]

    MODEL_PARAMS = {
        "trainable_out_dim": 32,
        "frozen_out_dim": 8,
        "latent_dim": 64
    }

    MODEL_SAVE_PATH = "./model/data_encoder.pth"

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    print("Starting training run...")

    training_info = train_data_encoder(
        id_L=0,
        id_R=10000,
        model_load_path=None,
        model_save_path=MODEL_SAVE_PATH,
        vector_column_list=VEC_COLS,
        scalar_column_list=SCA_COLS,
        ratio=0.5,
        cur=cursor,
        table_name=TABLE_NAME,
        num_epochs=20,
        model_params=MODEL_PARAMS,
        num_pretrain_epochs=5
    )

    if cursor:
        cursor.close()

    if training_info:
        print("Loading model for error checking...")

        try:
            n_vec = len(VEC_COLS)
            n_sca = len(SCA_COLS)

            inferred_vector_dims = training_info["vector_dims_list"]

            test_model = data_encoder(
                n_vector_cols=n_vec,
                vector_dims_list=inferred_vector_dims,
                n_scalar_cols=n_sca,
                **MODEL_PARAMS
            )

            test_model.load_state_dict(torch.load(MODEL_SAVE_PATH))

            test_vectors = [torch.rand(10, dim) for dim in inferred_vector_dims]
            test_scalars = [torch.rand(10, 1) for _ in range(n_sca)]

            errors = test_model.get_reconstruction_error(test_vectors, test_scalars)

            print(f"Test reconstruction errors (one per item): {errors}")
            print(f"Average reconstruction error: {errors.mean().item()}")

        except Exception as e:
            print(f"Could not run test inference: {e}")
    else:
        print("Training failed or was skipped, skipping inference test.")