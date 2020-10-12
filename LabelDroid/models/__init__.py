
'''
Author: Jieshan Chen
'''

import os

import numpy as np
import torch

from .decoder_LSTM import DecoderRNN
from .convcap import convcap
from .Transformer import Transformer

def setup(args):
	#lstm
	if args.caption_model == 'lstm':
		model = DecoderRNN(args.embed_size, args.hidden_size, args.vocab_len, args.num_layers, args.max_tokens)
	# convolutional caption
	elif args.caption_model == 'convcap':
		model = convcap(args.numwords, args.embed_size, args.num_layers, is_attention=args.attention)
	# Transformer
	elif args.caption_model == 'transformer':
		model = Transformer(args)
	else:
		raise Exception("Caption model not supported: {}".format(args.caption_model))

	return model
