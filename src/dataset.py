from datasets import load_dataset
from torch.utils.data import Dataset


class PavlickFormality(Dataset):
    def __init__(self,):
        super().__init__()

        ds = load_dataset("osyvokon/pavlick-formality-scores")

        self.extracted_samples = []
        self.labels = []

        for split in ['train', 'test']:
            for i, x in enumerate(ds[split]):
                if -0.6 < x['avg_score'] < 0.6:
                    continue

                self.labels.append(int(x['avg_score'] >= 0.6))

                self.extracted_samples.append(x['sentence'])
                
    def __getitem__(self, idx):
        x, y = self.extracted_samples[idx], self.labels[idx]

        return x, y 
    
    def __len__(self):
        return len(self.labels)
