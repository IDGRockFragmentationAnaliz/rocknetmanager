import cv2
import numpy as np
import sknw
from pathlib import Path
from shapely.geometry import LineString
import shapefile
from typing import List


def shape_lines_load(path: Path):
	with shapefile.Reader(path) as shp:
		shapes = shp.shapes()
		bbox = np.array(shp.bbox, np.float32)
		lines = []
		for shape in shapes:
			line = np.array(shape.points, np.float32) * [1, -1]
			lines.append(line)
		return lines, bbox
