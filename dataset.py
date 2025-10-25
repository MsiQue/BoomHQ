from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, input_data_list, target_data):
        self.input_data_list = input_data_list
        self.target_data = target_data

    def __len__(self):
        return len(self.target_data)

    def __getitem__(self, idx):
        input_samples = [data[idx] for data in self.input_data_list]
        target_sample = self.target_data[idx]
        return tuple(input_samples), target_sample