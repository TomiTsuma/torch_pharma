from torch.utils.data import Dataset

class BaseDataset(Dataset):
    """Base dataset class for all molecular datasets."""
    def __init__(self):
        super().__init__()
    
    def download(self, dataset_name):
        raise NotImplementedError
        
    def __len__(self):
        raise NotImplementedError
        
    def __getitem__(self, idx):
        raise NotImplementedError
