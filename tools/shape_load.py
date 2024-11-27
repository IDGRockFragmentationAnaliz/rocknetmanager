import cv2
import numpy as np
import sknw
from pathlib import Path
from shapely.geometry import LineString
import shapefile
from typing import List


def shape_lines_load(path: Path):
	path = (path / path.stem).with_suffix(".shp")
	with shapefile.Reader(str(path)) as shp:
		shapes = shp.shapes()
		bbox = np.array(shp.bbox, np.float32)
		lines = []
		for shape in shapes:
			line = np.array(shape.points, np.int32)
			if len(line) > 1:
				line = line*[1, -1]
				lines.append(line)
		return lines, bbox
