# Copyright (c) Microsoft. All rights reserved.

from .vocab import Vocabulary
from .metrics import compute_acc, compute_f1, compute_mcc, compute_pearson, compute_spearman
# scitail
ScitailLabelMapper = Vocabulary(True)
ScitailLabelMapper.add('neutral')
ScitailLabelMapper.add('entails')

# label map
SNLI_LabelMapper = Vocabulary(True)
SNLI_LabelMapper.add('contradiction')
SNLI_LabelMapper.add('neutral')
SNLI_LabelMapper.add('entailment')

# qnli
QNLILabelMapper = Vocabulary(True)
QNLILabelMapper.add('not_entailment')
QNLILabelMapper.add('entailment')

GLOBAL_MAP = {
 'scitail': ScitailLabelMapper,
 'mnli': SNLI_LabelMapper,
 'snli': SNLI_LabelMapper,
 'qnli': QNLILabelMapper,
 'qnnli': QNLILabelMapper,
 'rte': QNLILabelMapper,
 'diag': SNLI_LabelMapper,
}

# I'm not sure what each of does parameters do. I tried to guess the right values for toxic

# number of class
# This seems to be the number of classes in a classification task. Therefore, for regression,
# we put the value as 1.
# I'm not sure how this works for question anwsering tasks
DATA_META = {
 'mnli': 3,
 'snli': 3,
 'scitail': 2,
 'qqp': 2,
 'qnli': 2,
 'qnnli': 1,
 'wnli': 2,
 'rte': 2,
 'mrpc': 2,
 'diag': 3,
 'sst': 2,
 'stsb': 1,
 'cola': 2,
 'toxic': 1
}

# It seems like this has to be 1 when type_id is an array of zeros
DATA_TYPE = {
 'mnli': 0,
 'snli': 0,
 'scitail': 0,
 'qqp': 0,
 'qnli': 0,
 'qnnli': 0,
 'wnli': 0,
 'rte': 0,
 'mrpc': 0,
 'diag': 0,
 'sst': 1,
 'stsb': 0,
 'cola': 1,
 'toxic':1
}

# I'm assuming this is only needed for qqp
DATA_SWAP = {
 'mnli': 0,
 'snli': 0,
 'scitail': 0,
 'qqp': 1,
 'qnli': 0,
 'qnnli': 0,
 'wnli': 0,
 'rte': 0,
 'mrpc': 0,
 'diag': 0,
 'sst': 0,
 'stsb': 0,
 'cola': 0,
}

# classification/regression
# This one seems pretty straight forward. 1 for regression.
TASK_TYPE = {
 'mnli': 0,
 'snli': 0,
 'scitail': 0,
 'qqp': 0,
 'qnli': 0,
 'qnnli': 0,
 'wnli': 0,
 'rte': 0,
 'mrpc': 0,
 'diag': 0,
 'sst': 0,
 'stsb':1,
 'cola': 0,
 'toxic':1
}

# Since I want to do a regression, let's see if I can use the same as stsb
METRIC_META = {
 'mnli': [0],
 'snli': [0],
 'scitail': [0],
 'qqp': [0, 1],
 'qnli':[0],
 'qnnli': [0],
 'wnli': [0],
 'rte': [0],
 'mrpc': [0, 1],
 'diag': [0],
 'sst': [0],
 'stsb': [3, 4],
 'cola': [0, 2],
 'toxic': [3, 4]
}

METRIC_NAME = {
 0: 'ACC',
 1: 'F1',
 2: 'MCC',
 3: 'Pearson',
 4: 'Spearman',
}

METRIC_FUNC = {
 0: compute_acc,
 1: compute_f1,
 2: compute_mcc,
 3: compute_pearson,
 4: compute_spearman,
}

# note found on a deleted issue:
# SAN_META is to define whether the task can use the SAN net: 0 indicating we cannot use SAN in this task, e.g., regression. There is a small cleaning bug, and we'll fix it soon and add some comments in the code.
SAN_META = {
    'mnli': 1,
    'snli': 1,
    'scitail': 1,
    'qqp': 1,
    'qnli': 1,
    'qnnli': 1,
    'wnli': 1,
    'rte': 1,
    'mrpc': 1,
    'diag': 0,
    'sst': 0,
    'stsb': 0,
    'cola': 0,
    'toxic':0
}

def generate_decoder_opt(task, max_opt):
    assert task in SAN_META
    opt_v = 0
    if SAN_META[task] and max_opt < 3:
        opt_v = max_opt
    return opt_v