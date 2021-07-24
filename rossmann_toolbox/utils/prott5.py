import re
import torch
import requests
#import h5py
import io
import warnings

from .embedder import Embedder
from transformers import T5Tokenizer, T5EncoderModel


class ProtT5(Embedder):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        warnings.filterwarnings("ignore", category=UserWarning)
        self.tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_uniref50', do_lower_case=False)
        model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")
        self.device = 'cpu' if self.cuda_device == -1 else 'cuda'
        if self.device == 'gpu':
            model = model.half()
        self.model = model.to(self.device).eval()

    def _encode_batch(self, sequences):
        seq_lens = [len(seq) for seq in sequences]
        batch = [re.sub(r"[UZOB]", "X", sequence) for sequence in sequences]
        batch = [" ".join(list(seq)) for seq in batch]
        ids = self.tokenizer.batch_encode_plus(batch, add_special_tokens=True, padding='longest')
        tokenized_sequences = torch.tensor(ids["input_ids"]).to(self.device)
        attention_mask = torch.tensor(ids["attention_mask"]).to(self.device)
        with torch.no_grad():
            embeddings = self.model(input_ids=tokenized_sequences, attention_mask=attention_mask)
        embeddings = embeddings[0].cpu().numpy()
        return [emb[:seq_len] for emb, seq_len in zip(embeddings, seq_lens)]

    def _encode_batch_api(self, sequences):
        embeddings = []
        for sequence in sequences:
            req = {"model": "prottrans_t5_xl_u50", "sequence": sequence}
            r = requests.post('https://api.bioembeddings.com/api/embeddings', json=req)
            if r.status_code == 200:
                f = h5py.File(io.BytesIO(r.content), mode='r')
                embeddings.append(f['sequence'][:])
            else:
                raise ValueError('Communication with bioembeddings API failed!')
        return embeddings
