import pandas as pd
#import h5py


class Embedder:

    def __init__(self, cuda_device=-1, tokens_per_batch=16000):
        """
        Wrapper for efficient embedding of protein sequences with various embedding methods
        :param cuda_device: Index of the CUDA device to use when embedding (-1 if CPU)
        :param tokens_per_batch: Number of tokens (amino acids per encoded sequence batch) - depends on available GPU VRAM
        """
        self.cuda_device = cuda_device
        self.tokens_per_batch = tokens_per_batch

    @staticmethod
    def _validate_input(data):
        """
        Validates input pd.DataFrame with sequences that are to embedded
        :param data: input pd.DataFrame
        :return:
        """
        # Validate input DataFrame
        if not isinstance(data, pd.DataFrame):
            raise TypeError('Data must be a pandas DataFrame!')
        if 'sequence' not in data.columns:
            raise KeyError('DataFrame must contain sequence column!')

    def _batch_df(self, data):
        """
        Mark the input DataFrame so that each batch contains not more than 'self.tokens_per_batch' amino acids.
        :param data: input DataFrame
        :return: copy of the input DataFrame with additional 'batch' column
        """
        b_df = data.copy()
        b_df['seq_len'] = b_df['sequence'].apply(len)  # Calculate length of each sequence in DataFrame
        b_df = b_df.sort_values(by='seq_len')  # Sort sequences by length
        b_df['cum_seq_len'] = b_df['seq_len'].cumsum()  # Calculate cumulative sequence lengths to split into batches
        b_df['batch'] = b_df['cum_seq_len'] // self.tokens_per_batch
        return b_df

    def _encode_batch_api(self, sequences):
        raise NotImplementedError('Fetching embedding via API is not available for the selected model!')

    def encode(self, data, out_fn=None, api=False):
        if out_fn is not None:
            f = h5py.File(out_fn, 'w')
        self._validate_input(data)
        df = self._batch_df(data)
        results = {}
        for batch in df['batch'].unique():
            b_df = df[df['batch'] == batch]
            sequences = b_df['sequence'].tolist()
            embs = self._encode_batch_api(sequences) if api else self._encode_batch(sequences)
            for emb, idx in zip(embs, b_df.index.values):
                if out_fn is not None:
                    f.create_dataset(idx, data=emb)
                else:
                    results[idx] = emb
        if out_fn is not None:
            f.close()
        else:
            return results