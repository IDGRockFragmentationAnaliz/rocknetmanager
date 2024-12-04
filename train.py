import cv2
import time
from pathlib import Path
import matplotlib.pyplot as plt
import tifffile

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as f
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset
from torch.nn.functional import binary_cross_entropy
from tqdm import tqdm

from .utils import cross_entropy_loss_RCF
from .dataset import Dataset
from .utils import save_checkpoint


class ModelTrain:
	grad_com_size = 24

	def __init__(
		self,
		dataset: Dataset,
		model: nn.Module,
		optimizer: torch.optim.Optimizer
	):
		self.train_loader = DataLoader(dataset, batch_size=1, num_workers=4, shuffle=True, pin_memory=True)
		self.model = model
		self.optimizer = optimizer
		self.epoch = 100
		self.progress = tqdm(total=len(self.train_loader))
		self.counter = 0

	@staticmethod
	def cross_entropy_loss(prediction, labelf, beta):
		label = labelf.long()
		mask = labelf.clone()
		num_positive = torch.sum(label == 1).float()
		num_negative = torch.sum(label == 0).float()

		mask[label == 1] = 1.0 * num_negative / (num_positive + num_negative)
		mask[label == 0] = beta * num_positive / (num_positive + num_negative)
		mask[label == 2] = 0
		cost = binary_cross_entropy(
			prediction, labelf, weight=mask, reduction='sum')
		return cost

	def train_instance(self, image, label):
		print("1")
		outputs = self.model(image)
		print("2")
		loss = 0
		if isinstance(outputs, list):
			for output in outputs:
				loss = loss + self.cross_entropy_loss(output, label, 1.1)
		print("3")
		self.counter = self.counter + 1
		loss = loss / self.grad_com_size
		print("4")
		loss.backward()
		print("5")
		if self.counter == self.grad_com_size:
			self.optimizer.step()
			self.optimizer.zero_grad()
			self.counter = 0
		self.progress.update(1)
		print("6")

	def train(self):
		print("train start")
		self.model.train()
		self.optimizer.zero_grad()
		self.counter = 0
		#stream = torch.cuda.Stream()
		for i, (image, label) in enumerate(self.train_loader):
			image = image.cuda(non_blocking=True)
			label = label.cuda(non_blocking=True)
			#with torch.cuda.stream(stream):
			self.train_instance(image, label)


# if i % 100 == 0:
			# 	save_checkpoint({
			# 		'epoch': self.epoch,
			# 		'state_dict': self.model.model.state_dict(),
			# 		'optimizer': self.optimizer.state_dict(),
			# 	}, self.epoch, "save_models")