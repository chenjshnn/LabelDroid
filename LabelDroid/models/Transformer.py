'''
Author: Jieshan Chen 

Modified from 
https://nlp.seas.harvard.edu/2018/04/03/attention.html
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence


# Model Architecture
class EncoderDecoder(nn.Module):
	"""
	A standard Encoder-Decoder architecture. Base for this and many 
	other models.
	"""
	def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
		super(EncoderDecoder, self).__init__()
		self.encoder = encoder
		self.decoder = decoder
		self.src_embed = src_embed
		self.tgt_embed = tgt_embed
		self.generator = generator
		
	def forward(self, src, tgt, src_mask, tgt_mask):
		"Take in and process masked src and target sequences."
		out = self.encode(src, src_mask)
		# print("encoder output:",out.size())
		de = self.decode(out, src_mask,
							tgt, tgt_mask)
		# print("decoder output:", de.size())
		return de 
	
	def encode(self, src, src_mask):
		return self.encoder(self.src_embed(src), src_mask)
	
	def decode(self, memory, src_mask, tgt, tgt_mask):
		return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Generator(nn.Module):
	"Define standard linear + softmax generation step."
	def __init__(self, d_model, vocab):
		super(Generator, self).__init__()
		self.proj = nn.Linear(d_model, vocab)

	def forward(self, x):
		return F.log_softmax(self.proj(x), dim=-1)


# Encoder and Decoder Stacks
## Encoder
def clones(module, N):
	"Produce N identical layers."
	return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class LayerNorm(nn.Module):
	"Construct a layernorm module (See citation for details)."
	def __init__(self, features, eps=1e-6):
		super(LayerNorm, self).__init__()
		self.a_2 = nn.Parameter(torch.ones(features))
		self.b_2 = nn.Parameter(torch.zeros(features))
		self.eps = eps

	def forward(self, x):
		mean = x.mean(-1, keepdim=True)
		std = x.std(-1, keepdim=True)
		return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
	"""
	A residual connection followed by a layer norm.
	Note for code simplicity the norm is first as opposed to last.
	"""
	def __init__(self, size, dropout):
		super(SublayerConnection, self).__init__()
		self.norm = LayerNorm(size)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x, sublayer):
		"Apply residual connection to any sublayer with the same size."
		return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
	"Encoder is made up of self-attn and feed forward (defined below)"
	def __init__(self, size, self_attn, feed_forward, dropout):
		super(EncoderLayer, self).__init__()
		self.self_attn = self_attn
		self.feed_forward = feed_forward
		self.sublayer = clones(SublayerConnection(size, dropout), 2)
		self.size = size

	def forward(self, x, mask):
		"Follow Figure 1 (left) for connections."
		x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
		return self.sublayer[1](x, self.feed_forward)

## Decoder N=6
class Decoder(nn.Module):
	"Generic N layer decoder with masking."
	def __init__(self, layer, N):
		super(Decoder, self).__init__()
		self.layers = clones(layer, N)
		self.norm = LayerNorm(layer.size)
		
	def forward(self, x, memory, src_mask, tgt_mask):
		for layer in self.layers:
			x = layer(x, memory, src_mask, tgt_mask)
		return self.norm(x)

class DecoderLayer(nn.Module):
	"Decoder is made of self-attn, src-attn, and feed forward (defined below)"
	def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
		super(DecoderLayer, self).__init__()
		self.size = size
		self.self_attn = self_attn
		self.src_attn = src_attn
		self.feed_forward = feed_forward
		self.sublayer = clones(SublayerConnection(size, dropout), 3)
 
	def forward(self, x, memory, src_mask, tgt_mask):
		"Follow Figure 1 (right) for connections."
		m = memory
		x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
		x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
		return self.sublayer[2](x, self.feed_forward)

### tgt_mask
def subsequent_mask(size):
	"Mask out subsequent positions."
	attn_shape = (1, size, size)
	# np.triu: return the upper triangle of matrix below the k-th diagonal
	subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
	return torch.from_numpy(subsequent_mask) == 0

# Attention
def attention(query, key, value, mask=None, dropout=None):
	"Compute 'Scaled Dot Product Attention'"
	d_k = query.size(-1)	
	scores = torch.matmul(query, key.transpose(-2, -1)) \
			 / math.sqrt(d_k)
	if mask is not None:
		scores = scores.masked_fill(mask == 0, -1e9)
	p_attn = F.softmax(scores, dim = -1)
	if dropout is not None:
		p_attn = dropout(p_attn)
	return torch.matmul(p_attn, value), p_attn

# h=8
class MultiHeadedAttention(nn.Module):
	def __init__(self, h, d_model, dropout=0.1):
		"Take in model size and number of heads."
		super(MultiHeadedAttention, self).__init__()
		assert d_model % h == 0
		# We assume d_v always equals d_k
		self.d_k = d_model // h
		self.h = h
		self.linears = clones(nn.Linear(d_model, d_model), 4)
		self.attn = None
		self.dropout = nn.Dropout(p=dropout)
		
	def forward(self, query, key, value, mask=None):
		"Implements Figure 2"
		if mask is not None:
			# Same mask applied to all h heads.
			mask = mask.unsqueeze(1)
			
		nbatches = query.size(0)
		
		# 1) Do all the linear projections in batch from d_model => h x d_k 
		query, key, value = \
			[l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
			 for l, x in zip(self.linears, (query, key, value))]
		
		# 2) Apply attention on all the projected vectors in batch. 
		x, self.attn = attention(query, key, value, mask=mask, 
								 dropout=self.dropout)
		
		# 3) "Concat" using a view and apply a final linear. 
		x = x.transpose(1, 2).contiguous() \
			 .view(nbatches, -1, self.h * self.d_k)
		return self.linears[-1](x)

# Position-wise Feed-Forward Networks
class PositionwiseFeedForward(nn.Module):
	"Implements FFN equation."
	def __init__(self, d_model, d_ff, dropout=0.1):
		super(PositionwiseFeedForward, self).__init__()
		self.w_1 = nn.Linear(d_model, d_ff)
		self.w_2 = nn.Linear(d_ff, d_model)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x):
		# print("feed-forward input:", x.size())
		tmp = self.w_2(self.dropout(F.relu(self.w_1(x))))
		# print("feed-forward output:", tmp.size())
		return tmp

# Embeddings and Softmax
class Embeddings(nn.Module):
	def __init__(self, d_model, vocab):
		super(Embeddings, self).__init__()
		self.lut = nn.Embedding(vocab, d_model)
		self.d_model = d_model

	def forward(self, x):
		return self.lut(x) * math.sqrt(self.d_model)


# Positional Encoding
class PositionalEncoding(nn.Module):
	"Implement the PE function."
	def __init__(self, d_model, dropout, max_len=5000):
		super(PositionalEncoding, self).__init__()
		self.dropout = nn.Dropout(p=dropout)
		
		# Compute the positional encodings once in log space.
		pe = torch.zeros(max_len, d_model)
		position = torch.arange(0.0, max_len).unsqueeze(1)
		div_term = torch.exp(torch.arange(0.0, d_model, 2) *
							 -(math.log(10000.0) / d_model))
		pe[:, 0::2] = torch.sin(position * div_term).type(torch.LongTensor)
		pe[:, 1::2] = torch.cos(position * div_term).type(torch.LongTensor)
		pe = pe.unsqueeze(0)
		self.register_buffer('pe', pe)
		
	def forward(self, x):
		x = x + Variable(self.pe[:, :x.size(1)], 
						 requires_grad=False)
		return self.dropout(x)

def sort_pack_padded_sequence(input, lengths):
    sorted_lengths, indices = torch.sort(lengths, descending=True)
    tmp = pack_padded_sequence(input[indices], sorted_lengths, batch_first=True)
    inv_ix = indices.clone()
    inv_ix[indices] = torch.arange(0,len(indices)).type_as(inv_ix)
    return tmp, inv_ix

def pad_unsort_packed_sequence(input, inv_ix):
    tmp, _ = pad_packed_sequence(input, batch_first=True)
    tmp = tmp[inv_ix]
    return tmp

def pack_wrapper(module, att_feats, att_masks):
	if att_masks is not None:
		packed, inv_ix = sort_pack_padded_sequence(att_feats, att_masks.data.long().sum(1))
		return pad_unsort_packed_sequence(PackedSequence(module(packed[0]), packed[1]), inv_ix)
	else:
		return module(att_feats)


def subsequent_mask(batch_size, size):
    "Mask out subsequent positions."
    attn_shape = (batch_size, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

# src_vocab == att_vec_len
class Transformer(nn.Module):

	def identical_map(x):
		return x
	def make_model(self, tgt_vocab, N=6, 
			   d_model=512, d_ff=2048, h=8, dropout=0.1):
		"Helper: Construct a model from hyperparameters."
		c = copy.deepcopy
		attn = MultiHeadedAttention(h, d_model)
		ff = PositionwiseFeedForward(d_model, d_ff, dropout)
		position = PositionalEncoding(d_model, dropout)
		model = EncoderDecoder(
		Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
		Decoder(DecoderLayer(d_model, c(attn), c(attn), 
							 c(ff), dropout), N),
		nn.Identity(), #lambda x:x, #self.identical_map, #nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
		nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
		Generator(d_model, tgt_vocab))
		
		# This was important from their code. 
		# Initialize parameters with Glorot / fan_avg.
		for p in model.parameters():
			if p.dim() > 1:
				nn.init.xavier_uniform_(p)
		return model

	def __init__(self, args):
		super(Transformer, self).__init__()
		self.att_size = args.img_fatures_size
		self.embed_size = args.embed_size
		self.use_bn = args.use_bn
		self.drop_prob_lm = args.drop_prob_lm
		self.tgt_vocab_size = args.vocab_len
		self.num_layers = args.num_layers
		self.ff_size = args.ff_size
		
		self.att_embed = nn.Sequential(*(((nn.BatchNorm1d(self.att_size),) if self.use_bn else ())+(nn.Linear(self.att_size, self.embed_size), nn.ReLU(),nn.Dropout(self.drop_prob_lm))+((nn.BatchNorm1d(self.embed_size),) if self.use_bn==2 else ())))

		self.model = self.make_model(self.tgt_vocab_size, self.num_layers, self.embed_size, 
					     self.ff_size, 8, self.drop_prob_lm)

	def forward(self, att_feats, tgt, tgt_masks, att_masks = None):
		#att_masks = att_feats.new_ones(att_feats.shape[:2], dtype=torch.long)
		att_feats = self.att_embed(att_feats)
		# print("1 img_embed:", att_feats.size())
		out = self.model(att_feats, tgt, att_masks, tgt_masks)
		# print("-2 output:", out.size())
		outputs = self.model.generator(out)
		# print("final output generator:", outputs.size())
		return outputs
	
	def evaluate(self, att_feats, max_len=15, start_symbol=1, att_masks=None):
		att_feats = self.att_embed(att_feats)
		memory = self.model.encode(att_feats, att_masks)
		batch_size = att_feats.shape[0]
		ys = torch.ones(batch_size, 1).fill_(start_symbol).cuda().long()
		for i in range(max_len-1):
			out = self.model.decode(memory, att_masks, 
					Variable(ys), 
					Variable(subsequent_mask(batch_size, ys.size(1))
					.type_as(att_feats.data)))
			prob = self.model.generator(out[:, -1])
			#print(prob.shape)
			#print(prob[0])
			_, next_word = torch.max(prob, dim = 1)
			next_word = next_word
			#print(next_word)
			ys = torch.cat([ys, next_word.unsqueeze(-1)], dim=1)
		#print(ys)
		return ys
		
	def test_beam(self, att_feats, max_len, beam_searcher, start_symbol=1, att_masks=None):
		att_feats = self.att_embed(att_feats)
		memory = self.model.encode(att_feats, att_masks)
		batch_size = att_feats.shape[0]
		current_batch_size = batch_size
	
		ys = torch.ones(batch_size, 1).fill_(start_symbol).cuda().long()
		sentence_ids = np.zeros((batch_size, max_len), dtype='int64')

		for i in range(max_len-1):
			out = self.model.decode(memory, att_masks, 
					Variable(ys), 
					Variable(subsequent_mask(current_batch_size, ys.size(1))
					.type_as(att_feats.data)))

			prob = self.model.generator(out[:, -1])
			beam_indices, wordclass_indices = beam_searcher.expand_beam(prob) 
			current_batch_size = len(beam_indices)
			if len(beam_indices) == 0 or j == (max_tokens-2):
				# return all words index
				generated_captions = beam_searcher.get_results()
				for k in range(batch_size):
					# g store the all beamsize results for img_k
					sentence_ids[k,:] = generated_captions[:, k]
			else:
				ys = ys[beam_indices]
				memory = memory.index_select(0, Variable(torch.cuda.LongTensor(beam_indices)))
				att_feats = att_feats.index_select(0, Variable(torch.cuda.LongTensor(beam_indices)))
				next_word = Variable(torch.from_numpy(wordclass_indices)).cuda()
				ys = torch.cat([ys, next_word.unsqueeze(-1)], dim=1)
			
		return sentence_ids
		
		
		
	
	
