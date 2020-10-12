'''
Author: Jieshan Chen
'''

import argparse, os, time, pickle, json
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

from models.image_models import ResNetFeats
from evaluate import language_eval
from opts import get_opt
from optim import NoamOpt, LabelSmoothing
import models


def save_test_json(preds, resFile):
	print(('Writing %d predictions' % (len(preds))))
	json.dump(preds, open(resFile, 'w')) 

def test(args, split, modelfn=None, decoder=None, encoder=None):
	"""Runs test on split=val/test with checkpoint file modelfn or loaded model_*"""
	# Device configuration
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	
	# Find model directory
	if args.caption_model not in args.model_path:
		args.model_path += "_" + args.caption_model
		if args.finetune_cnn:
			args.model_path += "_finetune"

	# Get the best model path
	if encoder == None:
		modelfn = os.path.join(args.model_path, 'best_model.ckpt')

	# Load vocabulary 
	with open(args.vocab_path, 'rb') as f:
		vocab = pickle.load(f)

	# Build data loader
	data_loader = get_loader(args, vocab, split, shuffle=False) 

	max_tokens = args.max_tokens
	args.numwords = data_loader.dataset.numwords
	args.vocab_len = len(vocab)
	idx2word = vocab.idx2word
	num_batches = len(data_loader)
	print(('[DEBUG] Running inference on %s with %d batches' % (split.upper(), num_batches)))
	
	
	# Load model
	if modelfn is not None:
		print(('[INFO] Loading checkpoint %s' % modelfn))
		encoder = ResNetFeats(args) 
		decoder = models.setup(args)
		encoder.cuda()
		decoder.cuda()

		checkpoint = torch.load(modelfn)
		decoder.load_state_dict(checkpoint['decoder_state_dict'])
		encoder.load_state_dict(checkpoint['encoder_state_dict'])

	encoder.eval() 
	decoder.eval()

	pred_captions = []
	for i, current_batch in enumerate(tqdm(data_loader)):

		images, captions, _, _, _, img_ids, _ = current_batch

		images = images.to(device)

		if args.caption_model == "lstm":
			features = encoder(images)
			sentence_ids = decoder.sample(features).cpu().numpy()

			# Convert word_ids to words
			for j in range(args.batch_size):
				sampled_caption = []
				word_raw_id = []
				for word_id in sentence_ids[j]:
					word = idx2word[word_id]
					word_raw_id.append(word_id)
					if word == '<end>':
						break
					sampled_caption.append(word)
				word_raw_id = word_raw_id[1:]
				sentence = ' '.join(sampled_caption[1:])
				word_raw_id = [str(raw) for raw in word_raw_id]
				pred_captions.append({'image_id': img_ids[j], 'caption': sentence, "gt_caption":captions[j]})
	
		elif args.caption_model == "convcap":
			imgsfeats, imgsfc7 = encoder(images)
			_, featdim, feat_h, feat_w = imgsfeats.size()

			wordclass_feed = np.zeros((args.batch_size, max_tokens), dtype='int64')
			wordclass_feed[:,0] = vocab('<start>') 

			outcaps = np.empty((args.batch_size, 0)).tolist()

			for j in range(max_tokens-1):
				wordclass = Variable(torch.from_numpy(wordclass_feed)).cuda()

				wordact, _ = decoder(imgsfeats, imgsfc7, wordclass)

				wordact = wordact[:,:,:-1]
				# batch_size*max_token_len-1, vocab_len
				wordact_t = wordact.permute(0, 2, 1).contiguous().view(args.batch_size * (max_tokens-1), -1)

				wordprobs = F.softmax(wordact_t, dim=1).cpu().data.numpy()
				wordids = np.argmax(wordprobs, axis=1)

				word_raw_id = [[]]*args.batch_size
				for k in range(args.batch_size):
					word = idx2word[wordids[j+k*(max_tokens-1)]]
					outcaps[k].append(word)
					word_raw_id[k].append(wordids[j+k*(max_tokens-1)])
					if(j < max_tokens-1):
						wordclass_feed[k, j+1] = wordids[j+k*(max_tokens-1)]
			
			for j in range(args.batch_size):
				num_words = len(outcaps[j]) 
				if '<end>' in outcaps[j]:
					num_words = outcaps[j].index('<end>')
				outcap = ' '.join(outcaps[j][:num_words])
				
				current_word_raw_id = word_raw_id[k]#[:num_words]
				current_word_raw_id = [str(raw) for raw in current_word_raw_id]
				pred_captions.append({'image_id': img_ids[j], 'caption': outcap, "gt_caption":captions[j]})

		elif args.caption_model == "transformer":
			features = encoder(images)
			sentence_ids = decoder.evaluate(features, args.max_tokens).cpu().numpy()
	
			# Convert word_ids to words
			for j in range(args.batch_size):
				sampled_caption = []
				word_raw_id = []
				for word_id in sentence_ids[j]:
					word = idx2word[word_id]
					word_raw_id.append(word_id)
					if word == '<end>':
						break
					sampled_caption.append(word)
				sentence = ' '.join(sampled_caption[1:])
				word_raw_id = word_raw_id[1:]
				word_raw_id = [str(raw) for raw in word_raw_id]
				pred_captions.append({'image_id': img_ids[j], 'caption': sentence, "gt_caption":captions[j]})
	print(pred_captions[0:2])
	# Calculate scores
	scores = language_eval(args,pred_captions, args.model_path, split)
	
	if args.vis:
		print("[INFO] visualizing...")
		vis_folder = args.model_path.replace("models","vis") + "_" + "_".join(os.path.basename(args.caption_path).split("_")[-2:])
		target = os.path.join(vis_folder, "imgs")
		if not os.path.exists(target):
			os.makedirs(target)
		data = data_loader.dataset.data
		'''
		# save img
		for pred in pred_captions:
			img_id = pred["image_id"]
			path = data.loadImgs(img_id)[0]['filename']
			img_path = os.path.join(args.image_root, split, path)
			os.system("cp {} {}".format(img_path, target))
		'''
		# in order to save space, we use the original location of img to show them
		for k in range(len(pred_captions)):
			pred = pred_captions[k]
			img_id = pred["image_id"]
			path = data.loadImgs(img_id)[0]['filename'].replace(".jpg",".png")
			# need absolute path
			img_path = os.path.join(args.image_root, split, path)
			pred_captions[k]["img_path"] = img_path
		with open(os.path.join(vis_folder, "vis.json"),"w") as f:
			json.dump(pred_captions, f)
	
	encoder.train() 
	decoder.train()
	
	return scores 

def main(args):
	split = args.split
	test(args, split)
 
if __name__ == "__main__":
	args = get_opt()
	main(args)
