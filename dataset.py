import torch
from torch.utils.data import Dataset

class Shakespeare(Dataset):
    """ Shakespeare dataset for character-level language modeling.
    """
    def __init__(self, input_file, sequence_length=30):
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        chars = sorted(list(set(text)))
        self.char_to_idx = {ch: idx for idx, ch in enumerate(chars)}
        self.idx_to_char = {idx: ch for ch, idx in self.char_to_idx.items()}
        
        # Convert characters to indices
        self.data = [self.char_to_idx[ch] for ch in text]
        
        # Prepare sequences and targets
        self.sequence_length = sequence_length
        self.sequences = []
        self.targets = []
        for i in range(0, len(self.data) - sequence_length, sequence_length):
            self.sequences.append(self.data[i:i+sequence_length])
            self.targets.append(self.data[i+1:i+1+sequence_length])

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx]), torch.tensor(self.targets[idx])

if __name__ == '__main__':
    dataset = Shakespeare('./shakespeare_train.txt')
    print(dataset[0])
