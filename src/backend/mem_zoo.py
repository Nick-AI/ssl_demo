import torch
import torch.nn as nn
import torch.nn.functional as F
from .util import AliasMethod


class BaseMem(nn.Module):
    """Base Memory Class"""

    def __init__(self, K=65536, T=0.07, m=0.5):
        super(BaseMem, self).__init__()
        self.K = K
        self.T = T
        self.m = m

    def _update_memory(self, memory, x, y):
        """
        Args:
          memory: memory buffer
          x: features
          y: index of updating position
        """
        with torch.no_grad():
            x = x.detach()
            w_pos = torch.index_select(memory, 0, y.view(-1))
            w_pos.mul_(self.m)
            w_pos.add_(torch.mul(x, 1 - self.m))
            updated_weight = F.normalize(w_pos)
            memory.index_copy_(0, y, updated_weight)

    def _compute_logit(self, x, w):
        """
        Args:
          x: feat, shape [bsz, n_dim]
          w: softmax weight, shape [bsz, self.K + 1, n_dim]
        """
        x = x.unsqueeze(2)
        out = torch.bmm(w, x)
        out = torch.div(out, self.T)
        out = out.squeeze().contiguous()
        return out


class RGBMem(BaseMem):
    """Memory bank for single modality"""

    def __init__(self, n_dim, n_data, K=65536, T=0.07, m=0.5):
        super(RGBMem, self).__init__(K, T, m)
        # create sampler
        self.multinomial = AliasMethod(torch.ones(n_data))
        self.multinomial.cuda()

        # create memory bank
        self.register_buffer('memory', torch.randn(n_data, n_dim))
        self.memory = F.normalize(self.memory)

    def forward(self, x, y, x_jig=None, all_x=None, all_y=None):
        """
        Args:
          x: feat on current node
          y: index on current node
          x_jig: jigsaw feat on current node
          all_x: gather of feats across nodes; otherwise use x
          all_y: gather of index across nodes; otherwise use y
        """
        bsz = x.size(0)
        n_dim = x.size(1)

        # sample negative features
        idx = self.multinomial.draw(bsz * (self.K + 1)).view(bsz, -1)
        idx.select(1, 0).copy_(y.data)
        w = torch.index_select(self.memory, 0, idx.view(-1))
        w = w.view(bsz, self.K + 1, n_dim)

        # compute logits
        logits = self._compute_logit(x, w)
        if x_jig is not None:
            logits_jig = self._compute_logit(x_jig, w)

        # set label
        labels = torch.zeros(bsz, dtype=torch.long).cuda()

        # update memory
        if (all_x is not None) and (all_y is not None):
            self._update_memory(self.memory, all_x, all_y)
        else:
            self._update_memory(self.memory, x, y)

        if x_jig is not None:
            return logits, logits_jig, labels
        else:
            return logits, labels


class BaseMoCo(nn.Module):
    """base class for MoCo-style memory cache"""

    def __init__(self, K=65536, T=0.07):
        super(BaseMoCo, self).__init__()
        self.K = K
        self.T = T
        self.index = 0

    def _update_pointer(self, bsz):
        self.index = (self.index + bsz) % self.K

    def _update_memory(self, k, queue):
        """
        Args:
          k: key feature
          queue: memory buffer
        """
        with torch.no_grad():
            num_neg = k.shape[0]
            out_ids = torch.arange(num_neg).cuda()
            out_ids = torch.fmod(out_ids + self.index, self.K).long()
            queue.index_copy_(0, out_ids, k)

    def _compute_logit(self, q, k, queue):
        """
        Args:
          q: query/anchor feature
          k: key feature
          queue: memory buffer
        """
        # pos logit
        bsz = q.shape[0]
        pos = torch.bmm(q.view(bsz, 1, -1), k.view(bsz, -1, 1))
        pos = pos.view(bsz, 1)

        # neg logit
        neg = torch.mm(queue, q.transpose(1, 0))
        neg = neg.transpose(0, 1)

        out = torch.cat((pos, neg), dim=1)
        out = torch.div(out, self.T)
        out = out.squeeze().contiguous()

        return out


class RGBMoCo(BaseMoCo):
    """Single Modal (e.g., RGB) MoCo-style cache"""

    def __init__(self, n_dim, K=65536, T=0.07):
        super(RGBMoCo, self).__init__(K, T)
        # create memory queue
        self.register_buffer('memory', torch.randn(K, n_dim))
        self.memory = F.normalize(self.memory)

    def forward(self, q, k, q_jig=None, all_k=None):
        """
        Args:
          q: query on current node
          k: key on current node
          q_jig: jigsaw query
          all_k: gather of feats across nodes; otherwise use q
        """
        bsz = q.size(0)
        k = k.detach()

        # compute logit
        queue = self.memory.clone().detach()
        logits = self._compute_logit(q, k, queue)
        if q_jig is not None:
            logits_jig = self._compute_logit(q_jig, k, queue)

        # set label
        labels = torch.zeros(bsz, dtype=torch.long).cuda()

        # update memory
        all_k = all_k if all_k is not None else k
        self._update_memory(all_k, self.memory)
        self._update_pointer(all_k.size(0))

        if q_jig is not None:
            return logits, logits_jig, labels
        else:
            return logits, labels
