'''
Author: Jieshan Chen 
'''

import torch
import torch.nn as nn
from torchvision import models
from torch.autograd import Variable
import torch.nn.functional as F
import copy


class ResNetFeats(nn.Module):
	def __init__(self, args):
		"""Load the pretrained ResNet-101 and replace top fc layer."""
		super(ResNetFeats, self).__init__()
		self.caption_model = args.caption_model
		self.att_size = getattr(args, "att_size", 7)
		self.embed_size = getattr(args, "embed_size", 4096)
		self.finetune_cnn = getattr(args, "finetune_cnn", False)
		resnet = models.resnet101(pretrained=True)
		if not self.finetune_cnn:
			modules = list(resnet.children())[:-2]      # delete the last fc layer. 
		else:
			modules = list(resnet.children())[:-3] 
			self.resnetLayer4 = resnet.layer4

		self.resnet = nn.Sequential(*modules)

		self.adaptive_pool1x1 = list(resnet.children())[-2]

		self.adaptive_pool7x7 = nn.AdaptiveAvgPool2d((self.att_size, self.att_size))

		self.linear = nn.Linear(resnet.fc.in_features, self.embed_size)
		self.bn = nn.BatchNorm1d(self.embed_size, momentum=0.01)


	def forward(self, images):
		print("image size:", list(images.size()))
		"""Extract feature vectors from input images."""
		with torch.no_grad():
			x = self.resnet(images)
		if self.finetune_cnn:
			x = self.resnetLayer4(x)
		if self.caption_model in ["lstm","convcap"]:
			# [batch_size, 2048, 1, 1]
			features = self.adaptive_pool1x1(x)
			# [batch_size, 2048]
			features = torch.flatten(features, 1)
			# img features
			features = features.reshape(features.size(0), -1)
			features = self.bn(self.linear(features))

		if self.caption_model == "convcap":
			# att: [batchsize, 2048, 7, 7]
			att = self.adaptive_pool7x7(x)
			return att, features
		elif self.caption_model == "lstm":
			return features
		elif self.caption_model == "transformer":
			# fc: [batchsize, 8]
			# fc = x.mean(3).mean(2)
			# att: [batchsize, 7, 7, 2048]

			att = self.adaptive_pool7x7(x).squeeze()
			if images.size(0) == 1:
				att = att.unsqueeze(0)
			att = att.permute(0, 2, 3, 1)
			att = att.view(images.size(0), -1, att.size(-1))
			
				
			return att
				

