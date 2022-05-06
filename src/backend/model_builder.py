""" Functions to establish full models for CL frameworks and linear readout. 

So far only supports 3-channel images.  
Code adapted from https://github.com/HobbitLong/PyContrast.
"""

import sys
import torch.nn as nn
from .model_zoo import model_dict
from .util import Normalize, JigsawHead


class RGBSingleHead(nn.Module):
    """RGB model with a single linear/mlp projection head"""

    def __init__(self, name='resnet50', head='linear', feat_dim=128):
        super(RGBSingleHead, self).__init__()

        name, width = self._parse_width(name)
        dim_in = int(2048 * width)
        self.width = width

        self.encoder = model_dict[name](width=width)

        if head == 'linear':
            self.head = nn.Sequential(
                nn.Linear(dim_in, feat_dim),
                Normalize(2)
            )
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim),
                Normalize(2)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    @staticmethod
    def _parse_width(name):
        if name.endswith('x4'):
            return name[:-2], 4
        elif name.endswith('x2'):
            return name[:-2], 2
        else:
            return name, 1

    def forward(self, x, mode=0):
        # mode --
        # 0: normal encoder,
        # 1: momentum encoder,
        # 2: testing mode
        feat = self.encoder(x)
        if mode == 0 or mode == 1:
            feat = self.head(feat)
        return feat


class RGBMultiHeads(RGBSingleHead):
    """RGB model with Multiple linear/mlp projection heads"""

    def __init__(self, name='resnet50', head='linear', feat_dim=128):
        super(RGBMultiHeads, self).__init__(name, head, feat_dim)

        self.head_jig = JigsawHead(dim_in=int(2048*self.width),
                                   dim_out=feat_dim,
                                   head=head)

    def forward(self, x, x_jig=None, mode=0):
        # mode --
        # 0: normal encoder,
        # 1: momentum encoder,
        # 2: testing mode
        if mode == 0:
            feat = self.head(self.encoder(x))
            feat_jig = self.head_jig(self.encoder(x_jig))
            return feat, feat_jig
        elif mode == 1:
            feat = self.head(self.encoder(x))
            return feat
        else:
            feat = self.encoder(x)
            return feat


NAME_TO_FUNC = {
    'RGBSin': RGBSingleHead,
    'RGBMul': RGBMultiHeads
}


def build_model(opt):
    # specify modal key
    branch = 'Mul' if opt.jigsaw else 'Sin'
    try:
        assert opt.modal == 'RGB'
    except AssertionError:
        sys.exit('Modality not supported.')

    model_key = opt.modal + branch

    model = NAME_TO_FUNC[model_key](opt.arch, opt.head, opt.feat_dim)
    if opt.mem == 'moco':
        model_ema = NAME_TO_FUNC[model_key](opt.arch, opt.head, opt.feat_dim)
    else:
        model_ema = None

    return model, model_ema


def build_linear(opt):
    n_class = opt.n_class
    arch = opt.arch
    if arch.endswith('x4'):
        n_feat = 2048 * 4
    elif arch.endswith('x2'):
        n_feat = 2048 * 2
    else:
        n_feat = 2048

    classifier = nn.Linear(n_feat, n_class)
    return classifier
