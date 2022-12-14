#utils
import os
import pdb
import conf
import numpy as np
import glob

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def get_part_model(exp_name, part, fdir = './exp'):
    exps = []
    exp_paths = glob.glob(f'{fdir}/{exp_name}/*{part}*.pt')
    for exp in sorted(exp_paths):
        if part in exp:
            exps.append(exp)
    return exps[-1]

def feat_ext(data, model):
    feats = model.forward(data, feat_ext = True)
    return feats
