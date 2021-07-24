import torch.nn.functional as F
from torch import nn
from .crf import CRF


class SeqCoreDetector(nn.Module):

    def __init__(self):
        super(SeqCoreDetector, self).__init__()
        self.linear1 = nn.Linear(1024, 64)
        self.conv1 = nn.Conv1d(64, 16, kernel_size=7, padding=3)
        self.linear2 = nn.Linear(16, 2)
        self.crf = CRF(2, batch_first=True)

    def _get_emission(self, x):
        x = F.relu(F.dropout(self.linear1(x), 0.25))
        x = x.permute(0, 2, 1).contiguous()
        x = F.relu(F.dropout(self.conv1(x), 0.25))  # [batch, length, 64]==>[batch, length, 16]
        x = x.permute(0, 2, 1).contiguous()
        x = self.linear2(x)
        return x

    def predict(self, x, mask):
        x = self._get_emission(x)
        out = self.crf.decode(x, mask)
        prob = self.crf.compute_marginal_probabilities(x, mask=mask)
        return out, prob

    def forward(self, x, mask, labels):
        x = self._get_emission(x)
        loss = self.crf(x, labels, mask)
        return -loss


