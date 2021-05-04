'''
Author: Jieshan Chen
'''

import argparse, os, time, pickle, json, random, glob
import numpy as np
from tqdm import tqdm 
from PIL import Image

import torch
from torchvision import transforms
# from torch import nn
# from torch.utils.data import Dataset
# from torch.autograd import Variable

# from opts import get_opt
# from models.combined_model import LabelDroid

parser = argparse.ArgumentParser(description='PyTorch Convolutional Image Captioning Model')

# Data settings
parser.add_argument('--image_root', type=str, default= './sample',\
		help='directory containing coco dataset train2014, val2014, & annotations')
parser.add_argument('--vocab_path', type=str, default= "data/vocab_idx2word.josn", help='vocabulary idx2word path')
parser.add_argument('--model_path', type=str, help='load trained pt model')
parser.set_defaults(attention=True)
args = parser.parse_args()


def sample(args):
	# Device configuration
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	# Load vocabulary idx2word dict
	with open(args.vocab_path, 'r') as f:
		idx2word = json.load(f)

	## Load model
	if os.path.exists(args.model_path):
		print(('[INFO] Loading checkpoint %s' % args.model_path))

		labeldroid = torch.load(args.model_path)
		labeldroid.to(device)
		labeldroid.eval()

	else:
		print("Error: the model path does not exist -", args.model_path)
		sys.exit(0)


	## LOAD DATA 
	# image preprocessing 
	img_size = (224,224)
	img_transforms_test = transforms.Compose([
					transforms.Resize(img_size),
					transforms.ToTensor(),
					transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ], 
							     std  = [ 0.229, 0.224, 0.225 ])
					])
	# get all images
	all_image_path = []
	# accepted image format
	for postfix in [".png", ".PNG", ".JPG", ".JPEG", ".jpg", ".jpeg"]:
		all_image_path.extend(glob.glob(args.image_root+"/**/**"+postfix, recursive=True))
	all_image_path = list(set(all_image_path))

	print(('[DEBUG] Running inference on %d images' % (len(all_image_path))))


	## start inference
	pred_captions = []
	for i, curr_path in enumerate(tqdm(all_image_path)):
		img_ids = [curr_path]

		# orginal image size: [3, ??, ??]
		images = Image.open(curr_path).convert('RGB')
		# resize to [3, 244, 244]
		images = img_transforms_test(images)
		# convert to torch tensor 
		# and then add one dimension -> [1, 3, 244, 244]
		images = images.unsqueeze(0).to(device)

		# get generated token ids
		sentence_ids = labeldroid(images).cpu().numpy()

		# torch.onnx.export(labeldroid, images, "labeldroid.onnx", \
		# 				  verbose=True, input_names=["images"], output_names=["sentence_ids"])


		# Convert word_ids to words
		for j in range(len(sentence_ids)):
			sampled_caption = []
			word_raw_id = []
			for word_id in sentence_ids[j]:
				word = idx2word[str(word_id)]
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
	

if __name__ == "__main__":
	sample(args)
