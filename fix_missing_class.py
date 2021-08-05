# -*- coding: utf-8 -*-
"""修正 pickle 时缺少类的问题
"""

import sys

import torch

import network

sys.modules['core_seq2seq'] = sys.modules['__main__']
sys.modules['core_seq2seq.network'] = network

str_preprocessor = torch.load('str_preprocessor.class')

del sys.modules['core_seq2seq.network']
del sys.modules['core_seq2seq']

torch.save(str_preprocessor, 'str_preprocessor.class.fixed')