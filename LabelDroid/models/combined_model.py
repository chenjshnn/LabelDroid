import torch
import torch.nn as nn
from models.image_models import ResNetFeats
import models

class LabelDroid(nn.Module):
	def __init__(self, args):
		"""Load the pretrained ResNet-101 and replace top fc layer."""
		super(LabelDroid, self).__init__()
		self.encoder = ResNetFeats(args) 
		self.decoder = models.setup(args)
		self.args = args

	def forward(self, images):
		features = self.encoder(images)
		sentence_ids = self.decoder.evaluate(features, self.args.max_tokens)
		return sentence_ids
	