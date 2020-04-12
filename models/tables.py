temp_arch = """ 
c1(och 10, ker 3, str 1, pad 1, bias, bn, pt max, pk 2, ps 1, actv relu); 
fc(out 100, bias, actv relu);
fc(out 1, bias, bn, actv relu)
"""

CONV1D_TABLE = {
    'och':  {'key': 'out_channels', 'default': 'must specify'},
    'ker':  {'key': 'kernel_size', 'default': 5},
    'str':  {'key': 'stride', 'default': 1},
    'pad':  {'key': 'padding', 'default': 0},
    'bias': {'key': 'bias', 'default': False},
    'bn':   {'key': 'use_bn', 'default': False},
    'pt':   {'key': 'pool_type', 'default': None},
    'pk':   {'key': 'pool_kernel_size', 'default': None},
    'ps':   {'key': 'pool_stride', 'default': 1},
    'pp':   {'key': 'pool_padding', 'default': 0},
    'actv': {'key': 'actv_type', 'default': None},

}

FC_TABLE = {
    'out':  {'key': 'out_size', 'default': 'must specify'},
    'bias': {'key': 'bias', 'default': False},
    'bn':   {'key': 'use_bn', 'default': False},
    'actv': {'key': 'actv_type', 'default': None},
}
