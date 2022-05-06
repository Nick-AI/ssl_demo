""" Functions to create different memory bank versions. 

So far only supports 3-channel images.  
Code adapted from https://github.com/HobbitLong/PyContrast.
"""

from .mem_zoo import RGBMem, RGBMoCo


def build_mem(opt, n_data):
    if opt.modal != 'RGB':
        raise NotImplementedError(f'Modality not supported: {opt.modal}')
    if opt.mem == 'bank':
        mem_func = RGBMem
        memory = mem_func(opt.feat_dim, n_data,
                          opt.nce_k, opt.nce_t, opt.nce_m)
    elif opt.mem == 'moco':
        mem_func = RGBMoCo
        memory = mem_func(opt.feat_dim, opt.nce_k, opt.nce_t)
    else:
        raise NotImplementedError(
            'mem not suported: {}'.format(opt.mem))

    return memory
