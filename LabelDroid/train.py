'''
Author: Jieshan Chen
'''

import argparse, os, time, pickle, random, sys
import numpy as np
from tqdm import tqdm 

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from data_utils.data_loader import get_loader 
from data_utils.build_vocab import Vocabulary

from tensorboardX import SummaryWriter

from models.image_models import ResNetFeats
from test import test
from opts import get_opt
from optim import NoamOpt, LabelSmoothing
import models

def main(args):
	# Device configuration
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	# Create result model directory
	args.model_path += "_" + args.caption_model
	if args.finetune_cnn:
		args.model_path += "_finetune"
		
	if not os.path.exists(args.model_path):
		os.makedirs(args.model_path)
	writer = SummaryWriter(log_dir=args.model_path)
	# Load vocabulary
	with open(args.vocab_path, 'rb') as f:
		vocab = pickle.load(f)
	
	# Build data loader
	data_loader = get_loader(args, vocab, "train", shuffle=True) 

	max_tokens = args.max_tokens
	args.numwords = data_loader.dataset.numwords
	args.vocab_len = len(vocab)

	print("# args.img_fatures_size:", args.img_fatures_size)

	# Build the models
	encoder = ResNetFeats(args)
	decoder = models.setup(args)

	encoder.to(device)
	encoder.train(True)
	decoder.to(device)
	decoder.train(True)

	# criterion
	if args.caption_model == "transformer":
		criterion = LabelSmoothing(size=args.vocab_len, padding_idx=0, smoothing=0.1)
	else:
		criterion = nn.CrossEntropyLoss()
	criterion.to(device)
	
	# optimizer
	params = list(decoder.parameters()) 
	if args.finetune_cnn:
		params += list(encoder.resnetLayer4.parameters())
	if args.caption_model == "lstm":
		params += list(encoder.linear.parameters()) + list(encoder.bn.parameters())
	elif args.caption_model == "convap":
		params += list(encoder.linear.parameters()) + list(encoder.bn.parameters()) + list(encoder.adaptive_pool7x7.parameters())
	elif args.caption_model == "transformer":
		params +=  list(encoder.adaptive_pool7x7.parameters())

	if args.caption_model != "transformer":
		optimizer = torch.optim.Adam(params, lr=args.learning_rate)
		# optimizer = torch.optim.RMSprop(params, lr=args.learning_rate)
		scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=.1)
	else:
		# as in paper
		# d_model = embed_size
		optimizer = NoamOpt(args.embed_size, 1, 4000,  \
			    torch.optim.Adam(params, lr=0, betas=(0.9, 0.98), eps=1e-9))

	start_epoch = 0
	if getattr(args, "checkpoint", None) is not None:
		# check if all necessary files exist 
		print("[INFO] Loading best model from checkpoint:", os.path.join(args.model_path, 'best_model.ckpt'))
		checkpoint = torch.load(os.path.join(args.model_path, 'best_model.ckpt'))
		#checkpoint = torch.load(os.path.join(args.model_path, 'model-14-44.ckpt'))
		decoder.load_state_dict(checkpoint['decoder_state_dict'])
		encoder.load_state_dict(checkpoint['encoder_state_dict'])

		if args.caption_model == "transformer":
			optimizer.optimizer.load_state_dict((checkpoint['optimizer']))
		else:
			optimizer.load_state_dict(checkpoint['optimizer'])
		start_epoch = checkpoint["epoch"] + 1


	# Train the models
	bestscore = 0# 2.860 #2.690
	loss_train = 0
	total_step = len(data_loader)
	iteration = start_epoch * total_step
	bestiter = iteration
	train_start = time.time()
	for epoch in range(start_epoch, args.num_epochs):
		print("\n==>Epoch:", epoch)
		if args.caption_model != "transformer":
			# update lr
			scheduler.step()    
		random.shuffle(data_loader.dataset.ids)
		for i, current_batch in enumerate(tqdm(data_loader)):
			# break
			images, captions, targets, masks, trans_tgt_masks, img_ids, lengths = current_batch

			images = images.to(device)
			targets = targets.to(device)

			if args.caption_model == "lstm":
				targets_pad = pack_padded_sequence(targets, lengths, batch_first=True)[0]
				# Forward, backward and optimize
				features = encoder(images)
				outputs = decoder(features, targets, lengths)
				loss = criterion(outputs, targets_pad)

			elif args.caption_model == "convcap":
				imgsfeats, imgsfc7 = encoder(images)
				_, _, feat_h, feat_w = imgsfeats.size()

				if(args.attention == True):
					wordact, attn = decoder(imgsfeats, imgsfc7, targets)
					attn = attn.view(args.batch_size, max_tokens, feat_h, feat_w)
				else:
					wordact, _ = decoder(imgsfeats, imgsfc7, targets)

				wordact = wordact[:,:,:-1]
				targets = targets[:,1:]
				masks = masks[:,1:].contiguous()

				wordact_t = wordact.permute(0, 2, 1).contiguous().view(\
				args.batch_size*(max_tokens-1), -1)
				wordclass_t = targets.contiguous().view(\
				args.batch_size*(max_tokens-1), 1)

				maskids = torch.nonzero(masks.view(-1)).numpy().reshape(-1)

				if(args.attention == True):
					loss = criterion(wordact_t[maskids, ...], \
					       wordclass_t[maskids, ...].contiguous().view(maskids.shape[0])) \
					       + (torch.sum(torch.pow(1. - torch.sum(attn, 1), 2)))\
				               /(args.batch_size*feat_h*feat_w)
				else:
					loss = criterion(wordact_t[maskids, ...], \
				 	       wordclass_t[maskids, ...].contiguous().view(maskids.shape[0]))
			elif args.caption_model == "transformer":
				trans_tgt_masks = trans_tgt_masks.to(device)
				features = encoder(images)
				outputs = decoder(features, targets[:, :-1], trans_tgt_masks[:,:-1, :-1])
				# reshape  --> (batch_size*seq_len, vocab_len)
				outputs = outputs.contiguous().view(-1, outputs.size(-1))
				# reshape --> (batch_size*seq_len)
				targets = targets[:, 1:].contiguous().view(-1)
				loss = criterion(outputs, targets)/sum(lengths)

			loss_train = loss_train + loss.item()
			writer.add_scalar("Loss/train", loss.item(), iteration)
                
			decoder.zero_grad()
			encoder.zero_grad()
			loss.backward()
			optimizer.step()
			iteration += 1

		writer.add_scalar("Loss/train_epoch", loss_train, epoch)
		loss_train = 0

		# Save the model checkpoints  & evaluation
		# Run on validation and obtain score
		scores = test(args, 'val', encoder=encoder, decoder=decoder) 
		score = scores[0][args.score_select]
	
		if(score > bestscore):
			print("[INFO] save model")
			save_path = os.path.join(args.model_path, 'model-{}-{}.ckpt'.format(epoch+1, i+1))
			if args.caption_model == "transformer":
				optim_state_dict = optimizer.optimizer.state_dict()
			else:		
				optim_state_dict = optimizer.state_dict()

			torch.save({'epoch':epoch,
				    'decoder_state_dict':decoder.state_dict(),
				    'encoder_state_dict':encoder.state_dict(),
				    'optimizer':optim_state_dict,
				    'iteration':iteration, 
				    args.score_select:score,
				    }, save_path)
			bestiter = iteration
		for metric, value in scores[0].items():
			writer.add_scalar("val/{}".format(metric), value, epoch)
		if(score > bestscore):
			bestscore = score
			print(('[DEBUG] Saving model at epoch %d with %s score of %f'\
				% (epoch, args.score_select, score)))
			bestmodel_path = os.path.join(args.model_path, 'best_model.ckpt')
			os.system('cp %s %s' % (save_path, bestmodel_path))
		
	# Run on validation and obtain score
	scores = test(args, 'val', encoder=encoder, decoder=decoder) 
	score = scores[0][args.score_select]

	print("[INFO] save model")
	save_path = os.path.join(args.model_path, 'model-{}-{}.ckpt'.format(epoch+1, i+1))

	if args.caption_model == "transformer":
		optim_state_dict = optimizer.optimizer.state_dict()
	else:		
		optim_state_dict = optimizer.state_dict()	
	torch.save({'epoch':epoch,
		    'decoder_state_dict':decoder.state_dict(),
		    'encoder_state_dict':encoder.state_dict(),
		    'optimizer':optim_state_dict,
		    'iteration':iteration, 
		    args.score_select:score,
		    }, save_path)

	if(score > bestscore):
		bestscore = score
		print(('[DEBUG] Saving model at epoch %d with %s score of %f'\
			% (epoch, args.score_select, score)))
		bestmodel_path = os.path.join(args.model_path, 'best_model.ckpt')
		os.system('cp %s %s' % (save_path, bestmodel_path))
	# test
	args.vis = True
	test(args, "test", encoder=encoder, decoder=decoder)
				

if __name__ == '__main__':
	args = get_opt()
	print(args)
	main(args)
