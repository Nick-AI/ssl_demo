import math
import torch
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from torch import Tensor, nn
import sklearn.metrics as skm
from torch.nn import functional as F
from typing import Optional, List, Union
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

import pytorch_lightning as pl
from pl_bolts.optimizers.lars import LARS
from pytorch_lightning import LightningModule
from pl_bolts.optimizers.lr_scheduler import linear_warmup_decay

from demo_src.model_utils import resnet


# For multi-gpu training
class SyncFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        ctx.batch_size = tensor.shape[0]

        gathered_tensor = [torch.zeros_like(tensor) for _ in range(
            torch.distributed.get_world_size())]

        torch.distributed.all_gather(gathered_tensor, tensor)
        gathered_tensor = torch.cat(gathered_tensor, 0)

        return gathered_tensor

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        torch.distributed.all_reduce(
            grad_input, op=torch.distributed.ReduceOp.SUM, async_op=False)

        idx_from = torch.distributed.get_rank() * ctx.batch_size
        idx_to = (torch.distributed.get_rank() + 1) * ctx.batch_size
        return grad_input[idx_from:idx_to]


# ReLU projection head
class NonlinProjection(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=2048, output_dim=128):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim, bias=False),
        )

    def forward(self, x):
        x = self.model(x)
        return F.normalize(x, dim=1)


# main framework
class SimCLR(LightningModule):
    def __init__(self,
                 gpus: int,
                 num_samples: int,
                 batch_size: int,
                 num_nodes: int = 1,
                 arch: str = "resnet50",
                 hidden_mlp: int = 2048,
                 feat_dim: int = 128,
                 warmup_epochs: int = 10,
                 max_epochs: int = 100,
                 temperature: float = 0.1,
                 first_conv: bool = True,
                 maxpool1: bool = True,
                 optimizer: str = "adam",
                 exclude_bn_bias: bool = False,
                 start_lr: float = 0.0,
                 learning_rate: float = 1e-3,
                 final_lr: float = 0.0,
                 weight_decay: float = 1e-6,
                 pretrained_backend: bool = True
                 ** kwargs):
        super().__init__()
        self.save_hyperparameters(ignore=['transform'])

        self.gpus = gpus
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.num_nodes = num_nodes
        self.arch = arch
        self.hidden_mlp = hidden_mlp
        self.feat_dim = feat_dim
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.temperature = temperature
        self.first_conv = first_conv
        self.maxpool1 = maxpool1
        self.optimizer = optimizer
        self.exclude_bn_bias = exclude_bn_bias
        self.start_lr = start_lr
        self.learning_rate = learning_rate
        self.final_lr = final_lr
        self.weight_decay = weight_decay
        self.pretrained_backend = pretrained_backend

        global_batch_size = self.num_nodes * self.gpus * \
            self.batch_size if self.gpus > 0 else self.batch_size
        self.train_iters_per_epoch = self.num_samples // global_batch_size

        try:
            assert self.temperature > 0.
        except AssertionError:
            exit('Temperature must be > 0.0.')

        self.encoder = self.init_model()

        self.head = NonlinProjection(
            input_dim=self.hidden_mlp, output_dim=self.feat_dim)

    def forward(self, x):
        return self.encoder(x)

    def init_model(self):
        pretrained = self.pretrained_backend
        if self.arch == "resnet18":
            backbone = resnet.resnet18
        elif self.arch == "resnet50":
            backbone = resnet.resnet50
        else:
            raise NotImplementedError(
                f'Backbone {self.arch} not (yet) supported.')

        # projection head is pulled out in this implementation
        # so backbone is just resnet terminating in pooling layer
        return backbone(
            normalize=False,
            hidden_mlp=0,
            output_dim=0,
            nb_prototypes=0,
            first_conv=self.first_conv,
            maxpool1=self.maxpool1,
            pretrained=pretrained
        )

    def shared_step(self, batch, mode='train'):
        # 3rd image for online eval, 4th unprocessed image
        # idx is for memory bank or momemntum encoder, irrelevant here
        (img1, img2, _, raw), idx, lbl = batch

        if mode != 'train':
            # embed the more lightly augmented image
            h_val = self(_)
            z_val = self.head(h_val)

            self.logger.experiment.add_embedding(
                z_val, metadata=lbl.tolist(), label_img=raw, tag='Val_Projection')

        # get h embeddings
        h1 = self(img1)
        h2 = self(img2)

        # get z projections
        z1 = self.head(h1)
        z2 = self.head(h2)

        loss = self.nt_xent_loss(z1, z2, self.temperature, mode=mode)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, mode='train')

        self.log('train_loss', loss, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, mode='valid')

        self.log('valid_loss', loss, on_step=True, on_epoch=False)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.shared_step(batch, mode='valid')

        self.log('valid_loss', loss, on_step=True, on_epoch=False)
        return loss

    def predict_step(self, batch):
        return self(batch)

    def exclude_from_wt_decay(self, named_params, weight_decay, skip_list=("bias", "bn")):
        params = []
        excluded_params = []

        for name, param in named_params:
            if not param.requires_grad:
                continue
            elif any(layer_name in name for layer_name in skip_list):
                excluded_params.append(param)
            else:
                params.append(param)

        return [
            {"params": params, "weight_decay": weight_decay},
            {
                "params": excluded_params,
                "weight_decay": 0.0,
            },
        ]

    def configure_optimizers(self):
        if self.exclude_bn_bias:
            params = self.exclude_from_wt_decay(
                self.named_parameters(), weight_decay=self.weight_decay)
        else:
            params = self.parameters()

        # paper default
        if self.optimizer == 'lars':
            optimizer = LARS(
                params,
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=self.weight_decay,
                trust_coefficient=0.001,
            )
        # vanilla adam
        elif self.optimizer == 'adam':
            optimizer = torch.optim.Adam(
                params, lr=self.learning_rate, weight_decay=self.weight_decay)
        # adam with momentum
        elif self.optimizer == 'adamw':
            optimizer = optim.AdamW(
                params, lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            raise NotImplementedError(
                f'Optimizer {self.optimizer} not supported.')

        warmup_steps = self.train_iters_per_epoch * self.warmup_epochs
        total_steps = self.train_iters_per_epoch * self.max_epochs

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                linear_warmup_decay(warmup_steps, total_steps, cosine=True),
            ),
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [scheduler]

    def nt_xent_loss(self, out_1, out_2, temperature, eps=1e-6, mode='train'):
        """
        assume out_1 and out_2 are l2 normalized
        out_1: [batch_size, dim]
        out_2: [batch_size, dim]
        """
        # gather representations in case of distributed training
        # out_1_dist: [batch_size * world_size, dim]
        # out_2_dist: [batch_size * world_size, dim]
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            out_1_dist = SyncFunction.apply(out_1)
            out_2_dist = SyncFunction.apply(out_2)
        else:
            out_1_dist = out_1
            out_2_dist = out_2

        # out: [2 * batch_size, dim]
        # out_dist: [2 * batch_size * world_size, dim]
        out = torch.cat([out_1, out_2], dim=0)
        out_dist = torch.cat([out_1_dist, out_2_dist], dim=0)

        # cov and sim: [2 * batch_size, 2 * batch_size]
        # neg: [2 * batch_size]
        cov = torch.mm(out, out_dist.t())
        sim = torch.exp(cov / temperature)

        # loss denominator
        neg = sim.sum(dim=-1)
        # from each row, subtract e^(1/temp) to remove similarity measure for x1.x1 -> easier than masking
        row_sub = Tensor(neg.shape).fill_(
            math.e ** (1 / temperature)).to(neg.device)
        # clamp for numerical stability
        neg = torch.clamp(neg - row_sub, min=eps)

        # loss numerator
        # Positive similarity, pos becomes [2 * batch_size]
        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        pos = torch.cat([pos, pos], dim=0)

        loss = -torch.log(pos / (neg + eps)).mean()

        # calculate accuracy metrics
        # copy similarity matrix, no gradients needed for this part so detach
        sim_eval = sim.detach()
        # first find values to mask where similarity was calculated with itself& zero out
        self_mask = torch.eye(
            sim_eval.shape[0], dtype=torch.bool, device=sim_eval.device)
        sim_eval = sim_eval.masked_fill(self_mask, -9e15)
        # Find positive example -> batch_size//2 away from the original example
        pos_mask = self_mask.roll(shifts=sim_eval.shape[0] // 2, dims=0)
        # Get ranking position of positive example by logit score
        comb_sim = torch.cat(
            [sim_eval[pos_mask][:, None], sim_eval.masked_fill(
                pos_mask, -9e15)],  # First position positive example
            dim=-1,
        )
        sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)
        # Logging ranking metrics
        # positive sample ranked first
        self.log(mode + "_acc_top1", (sim_argsort == 0).float().mean(),
                 on_step=True, on_epoch=False)
        # positive sample ranked in top 5
        self.log(mode + "_acc_top5", (sim_argsort < 5).float().mean(),
                 on_step=True, on_epoch=False)
        # mean position of positive sample among entire batch
        self.log(mode + "_acc_mean_pos", 1 +
                 sim_argsort.float().mean(), on_step=True, on_epoch=False)

        return loss


class ContrastiveEvalFramework(LightningModule):
    def __init__(self,
                 contrastive_model: Union[LightningModule, nn.Module],
                 gpus: int,
                 batch_size: int,
                 num_nodes: int = 1,
                 transform: Optional[nn.Module] = None,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters(ignore=['transform'])

        self.model = contrastive_model
        self.gpus = gpus
        self.batch_size = batch_size
        self.num_nodes = num_nodes
        self.transform = transform

        self.model.cuda()
        self.model.eval()
        self.cuda()
        self.eval()

        self.embeddings = None
        self.labels = None

    def _get_embeddings(self, train_loader, test_loader):
        if self.embeddings:
            pass
        else:
            embs = []
            labels = []
            if train_loader:
                print('Generating embeddings for training data.')
                for x_batch, y_batch in tqdm(train_loader):
                    with torch.no_grad:
                        embs.append(self.model.predict(
                            x_batch).detach().to('cpu'))
                    labels.append(y_batch)
                print('Done.')

                self.embeddings['train'] = np.concatenate(embs, axis=1)
                self.labels['train'] = np.concatenate(labels, axis=1)
            if test_loader:
                print('Generating embeddings for testing data.')
                for x_batch, y_batch in tqdm(test_loader):
                    with torch.no_grad:
                        embs.append(self.model.predict(
                            x_batch).detach().to('cpu'))
                    labels.append(y_batch)
                print('Done.')

                self.embeddings['test'] = np.concatenate(embs, axis=1)
                self.labels['test'] = np.concatenate(labels, axis=1)

    def knn_eval(self, train_loader, test_loader, reload_embeddings=False, **kwargs):
        if not self.embeddings or reload_embeddings:
            self._get_embeddings(train_loader, test_loader)
        knn_cls = KNeighborsClassifier(**kwargs).fit(self.embeddings['train'],
                                                     self.labels['train'])
        test_preds = knn_cls.predict(self.embeddings['test'])
        test_probas = knn_cls.predict_proba(self.embeddings['test'])

        acc = self._acc_helper(test_preds, self.labels['test'])
        auroc = self._auroc_helper(test_probas, self.labels['test'])
        f1_score = self._f1_helper(test_probas, self.labels['test'])

        return {'top-1 accuracy': acc, 'auroc': auroc, 'f1': f1_score}

    def linear_eval(self, loader, reload_embeddings=False, **kwargs):
        if not self.embeddings or reload_embeddings:
            self._get_embeddings(loader)

        logreg_cls = LogisticRegression(**kwargs).fit(self.embeddings['train'],
                                                      self.labels['train'])
        test_preds = logreg_cls.predict(self.embeddings['test'])
        test_probas = logreg_cls.predict_proba(self.embeddings['test'])

        acc = self._acc_helper(test_preds, self.labels['test'])
        auroc = self._auroc_helper(test_probas, self.labels['test'])
        f1_score = self._f1_helper(test_probas, self.labels['test'])

        return {'top-1 accuracy': acc, 'auroc': auroc, 'f1': f1_score}

    def get_alignment(self, loader, transform, alpha=2):
        embsi, embsj = [], []
        for x_batch, _ in tqdm(loader):
            xi, xj = transform(x_batch)
            with torch.no_grad():
                hi = self.model.predict(xi).detach().to('cpu')
                hj = self.model.predict(xj).detach().to('cpu')
            embsi.append(hi)
            embsj.append(hj)

        embsi = np.concatenate(embsi, axis=1)
        embsj = np.concatenate(embsj, axis=1)

        return (embsi - embsj).norm(p=2, dim=1).pow(alpha).mean()

    def get_uniformity(self, loader, transform, t=2):
        embs = []
        for x_batch, _ in tqdm(loader):
            xi, _ = transform(x_batch)
            with torch.no_grad():
                h = self.model.predict(xi).detach().to('cpu')
            embs.append(h)

        embs = np.concatenate(embs, axis=1)
        return torch.pdist(embs, p=2).pow(2).mul(-t).exp().mean().log()

    @staticmethod
    def _acc_helper(preds, labels):
        return skm.accuracy_score(labels, preds)

    @staticmethod
    def _auroc_helper(preds, labels):
        return skm.roc_auc_score(labels, preds)

    @staticmethod
    def _f1_helper(preds, labels):
        return skm.f1_score(labels, preds)

    @staticmethod
    def _silhouette_helper(embeddings, preds):
        return skm.silhouette_score(embeddings, preds)

    @staticmethod
    def _mse_helper(preds, labels):
        return skm.mean_squared_error(labels, preds)

    @staticmethod
    def _mae_helper(preds, labels):
        return skm.mean_absolute_error(labels, preds)
