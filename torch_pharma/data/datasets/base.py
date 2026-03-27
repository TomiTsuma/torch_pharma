from torch.utils.data import Dataset

class BaseDataset(Dataset):
    """Base dataset class for all molecular datasets."""
    def __init__(self):
        super().__init__()
        
    def __len__(self):
        return 0
        
    def __getitem__(self, idx):
        raise NotImplementedError
