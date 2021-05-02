'''
Author: Jieshan Chen
'''

import os, nltk, json
import numpy as np
from PIL import Image
from data_utils.build_vocab import Vocabulary
from pycocotools.coco import COCO

import torch
from torch.utils.data import Dataset
from torch.autograd import Variable
from torchvision import transforms
import random
from gensim.models import FastText as ft

from imgaug import augmenters as iaa
import imgaug as ia

class ImgAugTransform:
  def __init__(self):
    self.aug = iaa.Sequential([
        iaa.Scale((224, 224)),
        iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=(0, 3.0))),
        iaa.Affine(rotate=(-20, 20), mode='constant'),
        iaa.Sometimes(0.25,
                      iaa.OneOf([iaa.Dropout(p=(0, 0.1)),
                                 iaa.CoarseDropout(0.1, size_percent=0.5),
				 iaa.Sharpen(alpha=0.5)])),
        iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True),
	iaa.Sometimes(0.15,iaa.WithChannels(0, iaa.Add((10, 100)))),
	iaa.Sometimes(0.05,iaa.ChangeColorspace(from_colorspace="RGB", to_colorspace="HSV"))
    ])
      
  def __call__(self, img):
    img = np.array(img)
    return self.aug.augment_image(img)

class AccessibilityDataset(Dataset):
	"""Loads train/val/test splits of our dataset"""

	def __init__(self, args, vocab, split, img_size = (224,224)):
		self.max_tokens = args.max_tokens
		self.root = args.image_root
		caption_json = os.path.join(args.caption_path, "captions_{}.json".format(split))
		self.caption_json = caption_json
		self.data = COCO(self.caption_json)
		self.ids = list(self.data.anns.keys())
		if split == "train":
			random.shuffle(self.ids)
		else:

			sorted(self.ids)
		
		# build for category info
		caption_file = json.load(open(self.caption_json, "r"))
		# print(caption_file["annotations"][0].keys())

		# <pad>:0, <start>:1, <end>:2, <unk>:3
		self.vocab = vocab
		self.split = split
		self.att_size = args.att_size
		self.img_seq_len = self.att_size ** 2
		self.numwords = len(self.vocab)
		print(('[DEBUG] #words in wordlist: %d' % (self.numwords)))

		# Image preprocessing, normalization for the pretrained resnet
		self.img_size = img_size
		self.img_transforms = transforms.Compose([
			transforms.Resize((256,256)),
			transforms.RandomCrop((224,224)),
			transforms.ColorJitter(hue=.05, saturation=.05),
			#transforms.RandomHorizontalFlip(), 
			transforms.ToTensor(),
			transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ], 
					     std  = [ 0.229, 0.224, 0.225 ])
		])
		self.img_transforms = transforms.Compose([
				        ImgAugTransform(),
				        lambda x: Image.fromarray(x),
					transforms.ToTensor(),
					transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ], 
							     std  = [ 0.229, 0.224, 0.225 ])
					])

		self.img_transforms_test = transforms.Compose([
					transforms.Resize(img_size),
					transforms.ToTensor(),
					transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ], 
							     std  = [ 0.229, 0.224, 0.225 ])
					])

	def subsequent_mask(self, size):
	    "Mask out subsequent positions."
	    attn_shape = (size, size)
	    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
	    return torch.from_numpy(subsequent_mask) == 0

	def __getitem__(self, idx):
		data = self.data
		vocab = self.vocab

		ann_id = self.ids[idx]
		caption = data.anns[ann_id]['caption']
		# Convert caption (string) to word ids.
		words = nltk.tokenize.word_tokenize(str(caption).lower())
			
		img_id = data.anns[ann_id]['image_id']
		path = data.loadImgs(img_id)[0]['filename'].replace(".jpg",".png")
		
		# preprocess img
		img = Image.open(os.path.join(self.root, self.split, path)).convert('RGB')

		#
		
		if self.split == "train":
			img = self.img_transforms(img)
		else:
			# do not use data augmentation when testing/validation
			img = self.img_transforms_test(img)
		

		# token2idx
		target = torch.LongTensor(self.max_tokens).zero_()
		# tgt_mask
		tgt_mask = torch.ByteTensor(self.max_tokens).zero_()
		att_mask = torch.ByteTensor(self.img_seq_len, self.img_seq_len).zero_()

		# add <start>
		words = ['<start>'] + words
		# cut words longer than (max_token-1)
		length = min(len(words), self.max_tokens-1)
		tgt_mask[:(length+1)] = 1
		words = words[:length]
		words += ['<end>']

		tmp = [vocab(token) for token in words]
		target[:length+1] = torch.LongTensor(tmp)
		
		# tgt_mask_transformer for transformer
		tgt_mask_transformer = (target != 0).unsqueeze(-2)
		tgt_mask_transformer = tgt_mask_transformer & Variable(self.subsequent_mask(target.size(-1)).type_as(tgt_mask_transformer.data))
		# add <end>
		length += 1

		data = [img, caption, target, tgt_mask, tgt_mask_transformer, img_id, length]
		

		return data

	def __len__(self):
		return len(self.ids)

def collate_fn(data):
	# Sort a data list by caption length (descending order).
	data.sort(key=lambda x: x[6], reverse=True)
	#images, captions, targets, sentence_masks, tgt_mask_transformers, img_ids, lengths, cate_features  = zip(*data)

	new_data = []
	for idx, item in enumerate(zip(*data)):
		if idx not in [1, 5, 6]:
			item = torch.stack(item, 0)
		else:
			# 1:captions 5:img_ids 6:lengths
			item = list(item)
		new_data.append(item)
	
	return new_data

def get_loader(args, vocab, split, shuffle=True):
	"""Returns torch.utils.data.DataLoader for custom dataset."""
	# caption dataset
	dataset = AccessibilityDataset(args, vocab, split)
	
	data_loader = torch.utils.data.DataLoader(dataset=dataset, 
						  batch_size=args.batch_size,
						  shuffle=shuffle,
						  num_workers=args.num_workers,
						  collate_fn=collate_fn, drop_last=True)
	return data_loader


