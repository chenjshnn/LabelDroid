'''
Author: Jieshan Chen
'''

import argparse, os, time, pickle, json, random, glob
import numpy as np
from tqdm import tqdm 

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.autograd import Variable
from data_utils.build_vocab import Vocabulary


# from models.combined_model import LabelDroid
from models.image_models import ResNetFeats
from opts import get_opt
import models

from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms


class AccessibilityDataset(Dataset):
	"""Loads train/val/test splits of our dataset"""
	def __init__(self, args, vocab, img_size = (224,224)):
		self.image_root = args.image_root
		self.all_image_path = []
		for postfix in [".png", ".PNG", ".JPG", ".JPEG", ".jpg", ".jpeg"]:
			self.all_image_path.extend(glob.glob(self.image_root+"/**/**"+postfix, recursive=True))
		self.all_image_path = list(set(self.all_image_path))

		# <pad>:0, <start>:1, <end>:2, <unk>:3
		self.vocab = vocab

		# Image preprocessing, normalization for the pretrained resnet
		self.img_size = img_size

		self.img_transforms_test = transforms.Compose([
					transforms.Resize(img_size),
					transforms.ToTensor(),
					transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ], 
							     std  = [ 0.229, 0.224, 0.225 ])
					])

	def __getitem__(self, idx):
		curr_path = self.all_image_path[idx]
		# preprocess img
		img = Image.open(curr_path).convert('RGB')
		img = self.img_transforms_test(img)
	
		data = [img, curr_path]
		
		return data

	def __len__(self):
		return len(self.all_image_path)

def collate_fn(data):

	vecs = list(map(lambda x:x[0], data))
	paths = list(map(lambda x:x[1], data))
	return [torch.stack(vecs, 0), list(paths)]

def get_loader(args, vocab):
	"""Returns torch.utils.data.DataLoader for custom dataset."""
	# caption dataset
	dataset = AccessibilityDataset(args, vocab)
	data_loader = torch.utils.data.DataLoader(dataset=dataset, 
						  batch_size=args.batch_size,
						  shuffle=False,
						  num_workers=args.num_workers,
						  collate_fn=collate_fn, drop_last=False)
	return data_loader


def save_test_json(preds, resFile):
	print(('Writing %d predictions' % (len(preds))))
	json.dump(preds, open(resFile, 'w')) 


def sample(args):
	# Device configuration
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	# Load vocabulary 
	with open(args.vocab_path, 'rb') as f:
		vocab = pickle.load(f)

	# Build data loader
	data_loader = get_loader(args, vocab) 

	max_tokens = args.max_tokens
	args.vocab_len = len(vocab)
	idx2word = vocab.idx2word

	num_batches = len(data_loader)
	print(('[DEBUG] Running inference on %d batches' % (num_batches)))
	
	
	# Load model
	if os.path.exists(args.model_path):
		print(('[INFO] Loading checkpoint %s' % args.model_path))
		# encoder = torch.load("encoder.pt")
		# decoder = torch.load("decoder.pt")

		checkpoint = torch.load(args.model_path)


		# labeldroid = LabelDroid(args)
		# labeldroid.decoder.load_state_dict(checkpoint['decoder_state_dict'])
		# labeldroid.encoder.load_state_dict(checkpoint['encoder_state_dict'])
		# torch.save(labeldroid, "labeldroid.pt")

		encoder = ResNetFeats(args) 
		decoder = models.setup(args)
		decoder.load_state_dict(checkpoint['decoder_state_dict'])
		encoder.load_state_dict(checkpoint['encoder_state_dict'])

		encoder.cuda()
		decoder.cuda()

		# torch.save(encoder, "encoder.pt")
		# torch.save(decoder, "decoder.pt")
		# return
	else:
		print("Error: the model path does not exist -", args.model_path)
		sys.exit(0)

	encoder.eval() 
	decoder.eval()

	pred_captions = []
	for i, current_batch in enumerate(tqdm(data_loader)):
		images, img_ids = current_batch
		images = images.to(device)

		if args.caption_model == "lstm":
			features = encoder(images)
			sentence_ids = decoder.sample(features).cpu().numpy()

			# Convert word_ids to words
			for j in range(min(len(sentence_ids), args.batch_size)):
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
				pred_captions.append({'image_id': img_ids[j], 'caption': sentence})
	
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
			
			for j in range(min(len(outcaps), args.batch_size)):
				num_words = len(outcaps[j]) 
				if '<end>' in outcaps[j]:
					num_words = outcaps[j].index('<end>')
				outcap = ' '.join(outcaps[j][:num_words])
				
				current_word_raw_id = word_raw_id[k]#[:num_words]
				current_word_raw_id = [str(raw) for raw in current_word_raw_id]
				pred_captions.append({'image_id': img_ids[j], 'caption': outcap})

		elif args.caption_model == "transformer":
			features = encoder(images)
			sentence_ids = decoder.evaluate(features, max_len=args.max_tokens).cpu().numpy()
	
			# Convert word_ids to words
			for j in range(min(len(sentence_ids), args.batch_size)):
				sampled_caption = []
				word_raw_id = []
				print(sentence_ids[j])
				for word_id in sentence_ids[j]:
					word = idx2word[word_id]
					word_raw_id.append(word_id)
					if word == '<end>':
						break
					sampled_caption.append(word)
				sentence = ' '.join(sampled_caption[1:])
				word_raw_id = word_raw_id[1:]
				word_raw_id = [str(raw) for raw in word_raw_id]
				pred_captions.append({'image_id': img_ids[j], 'caption': sentence})
	print(pred_captions[0:2])
	
	with open(os.path.join(args.image_root, "result.json"),"w") as f:
		json.dump(pred_captions, f)
	

def main(args):
	sample(args)
 
if __name__ == "__main__":
	args = get_opt()
	main(args)
