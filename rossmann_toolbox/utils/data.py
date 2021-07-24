import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class Dataset(Dataset):

    def __init__(self, data_dict, embs_dict=None):
        # Dictionary with ids as keys and (if with_labels==True) labels as values
        self.data_dict = data_dict
        # Maping between data ids and dataset indexes
        self.data_keys = {i: key for i, key in enumerate(self.data_dict)}
        #  Handle to files with embeddings. Indexes in h5 file must match the indexes in data_dict
        self.embs_dict = embs_dict
        # Flag for handling labels

    def __getitem__(self, index):
        id_ = self.data_keys[index]  # Item id
        emb = self.embs_dict[id_]
        seq_length = len(emb)  # Embedding length
        return emb, seq_length

    def __len__(self):
        return len(self.data_dict)


def collate():
    def collate_fn(data):
        data.sort(key=lambda x: x[1], reverse=True)
        seq_lengths = torch.LongTensor([item[1] for item in data])
        mask = length_to_mask(seq_lengths)
        embs = pad_sequence([torch.Tensor(item[0]) for item in data],
                            batch_first=True, padding_value=0)
        return embs, mask, seq_lengths

    return collate_fn


def length_to_mask(length, max_len=None, dtype=None):
    """length: B.
    return B x max_len.
    If max_len is None, then max of length will be used.
    """
    assert len(length.shape) == 1, 'Length shape should be 1 dimensional.'
    max_len = max_len or length.max().item()
    mask = torch.arange(max_len, device=length.device,
                        dtype=length.dtype).expand(len(length), max_len) < length.unsqueeze(1)
    if dtype is not None:
        mask = torch.as_tensor(mask, dtype=torch.bool)
    return mask
