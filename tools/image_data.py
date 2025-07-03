import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from rocknetmanager.tools.shape_load import shape_load
import pandas as pd


class ImageData:
	def __init__(self, image, label, area):
		self.image = image
		self.label = label
		self.area = area

	def crop_image(self, x, y, dx, dy):
		image = self.crop(self.image, x, y, dx, dy)
		label = self.crop(self.label, x, y, dx, dy)
		area = self.crop(self.area, x, y, dx, dy)
		return self.__class__(image, label, area)

	def is_in_bound(self, percent=0.4):
		area = self.area[:, :, 0]
		bound_area = np.prod(area.shape)
		count = cv2.countNonZero(area)
		return count/bound_area > percent

	def is_have_edges(self):
		label = self.label[:, :, 0]
		count = cv2.countNonZero(label)
		return count > 9

	def is_accessible(self):
		return self.is_in_bound() and self.is_have_edges()


	def rotate(self):
		self.image = cv2.rotate(self.image, cv2.ROTATE_90_CLOCKWISE)
		self.label = cv2.rotate(self.label, cv2.ROTATE_90_CLOCKWISE)
		self.area = cv2.rotate(self.area, cv2.ROTATE_90_CLOCKWISE)
		return self

	def save(self, root, name=None):
		name = "" if name is None else "_" + name
		#
		image_path = root / "images"
		label_path = root / "labels"
		#
		image_path.mkdir(parents=False, exist_ok=True)
		label_path.mkdir(parents=False, exist_ok=True)
		#
		image_path = image_path / ("image" + name + ".png")
		label_path = label_path / ("label" + name + ".png")
		#
		cv2.imwrite(str(image_path), self.image)
		cv2.imwrite(str(label_path), self.label)
		#
		return image_path, label_path

	@classmethod
	def from_vectors(cls, image, poly_areas, line_edges):
		label = np.zeros(image.shape, np.uint8)
		label = cv2.polylines(label, line_edges, False, (255, 255, 255), 3)
		#
		area = np.zeros(image.shape, np.uint8)
		area = cv2.fillPoly(area, poly_areas, (255, 255, 255))
		return cls(image, label, area)

	@classmethod
	def load(cls, image_folder):
		# Поддерживаемые расширения изображений
		supported_extensions = ['.png', '.jpg', '.jpeg']

		# Ищем файл с нужным именем и поддерживаемым расширением
		image_path = None
		for ext in supported_extensions:
			potential_path = (image_folder / image_folder.name).with_suffix(ext)
			if potential_path.exists():
				image_path = potential_path
				break

		if image_path is None:
			raise FileNotFoundError(
				f"No image file found with supported extensions {supported_extensions} in {image_folder}")

		path_areas = (image_folder / "areas")
		path_traces = (image_folder / "traces")

		# Чтение изображения
		image = cv2.imread(str(image_path))
		if image is None:
			raise ValueError(f"Failed to read image from {image_path}")
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

		lines, bbox = shape_load(path_traces)
		polies, bbox = shape_load(path_areas)

		return cls.from_vectors(image, poly_areas=polies, line_edges=lines), bbox

	@staticmethod
	def crop(image, x, y, dx, dy):
		return image[y:y + dy, x:x + dx]
