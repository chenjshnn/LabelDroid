'''
Author: Jieshan Chen
'''

import argparse

def get_opt():
	parser = argparse.ArgumentParser(description='PyTorch Convolutional Image Captioning Model')

	# Data settings
	parser.add_argument('--image_root', type=str, default= './data/coco/',\
			help='directory containing coco dataset train2014, val2014, & annotations')
	parser.add_argument('--caption_path', type=str, default= "/DATASET/annotation/captions_train.json", help='caption_train/val/test.json path')
	parser.add_argument('--vocab_path', type=str, default= "data/vocab.pkl", help='vocabulary path')
	parser.add_argument('--checkpoint', type=str, help='load trained model')
	

	# General setting
	parser.add_argument('--caption_model', type=str, choices=["transformer", "lstm", "convcap"], help='which model to decode img')
	parser.add_argument('--max_tokens', type=int, default= 15, help='max_tokens')
	parser.add_argument('--split', type=str, choices=["train", "test"] , help='which split (train/test)')
	parser.add_argument('--num_layers', type=int, default=3,\
			    help='depth of convcap network(3) or number of layers in lstm(1)')
	parser.add_argument('--embed_size', type=int , default=512, help='dimension of word embedding vectors')
	parser.add_argument('--finetune_cnn', type=bool , default=False, help='whether to finetune ResNet')
	parser.add_argument('--model_path', type=str, default="run/models", help='output directory to save models & results')
	parser.add_argument('--gpu', type=int, default=0, help='gpu device id')


	# Optimization settings
	parser.add_argument('--num_epochs', type=int, default=100, help='number of training epochs')
	parser.add_argument('--batch_size', type=int, default=64, help='number of images per training batch')
	parser.add_argument('--num_workers', type=int, default=1, help='pytorch data loader threads')
	parser.add_argument('-lr', '--learning_rate', type=float, default=0.001 ,\
			    help='learning rate for convcap(5e-5), for lstm(0.001)')
	parser.add_argument('-st', '--lr_step_size', type=int, default=10,\
			    help='epochs to decay learning rate after')

	# convcap
	parser.add_argument('--attention', dest='attention', action='store_true', \
			    help='Use this for convcap with attention (by default set)')
	parser.add_argument('--no-attention', dest='attention', action='store_false', \
			    help='Use this for convcap without attention')

    # lstm
	parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')

	# transformer
	parser.add_argument('--att_size', type=int , default=7, help='attention size for transfomer')
	parser.add_argument('--use_bn', type=int, help="whether to use batch normalzation when embedding attention vector of img")
	parser.add_argument('--drop_prob_lm', type=float, default=0.1, help="dropout rate for language model")
	parser.add_argument('--ff_size', type=int, default = 2048, help="feed forward size for transformer")
	parser.add_argument('--img_fatures_size', type=int, default=2048,help='embedding size for img')
  	
	# evaluation
	parser.add_argument('--log_step', type=int , default=10, help='step size for prining log info')
	parser.add_argument('--save_step', type=int , default=50, help='step size for saving trained models')

	parser.add_argument('-sc', '--score_select', type=str, default='CIDEr',\
			    help='metric to pick best model')
	parser.add_argument('--beam_size', type=int, default=1, \
			    help='beam size to use for test') 
	parser.add_argument('--vis', type=bool, default=False,\
			    help='whether to visualize results')
	

	parser.set_defaults(attention=True)
	args = parser.parse_args()
	
	return args
