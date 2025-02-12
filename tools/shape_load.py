import numpy as np
from pathlib import Path
import shapefile


def shape_load(path: Path, factor=1, shift=np.zeros((1, 2))):
	if path.is_dir() and path.exists():
		for file in path.iterdir():
			if file.suffix == ".shp":
				path = file
				break

	with shapefile.Reader(str(path)) as shp:
		shapes = shp.shapes()
		bbox = np.array(shp.bbox, np.float64)
		_bbox = bbox.copy()
		# _bbox[0] = (bbox[0] + shift[0, 0]) * factor
		# _bbox[1] = (-bbox[3] + shift[0, 1]) * factor
		# _bbox[3] = (-bbox[1] + shift[0, 1]) * factor
		bbox = _bbox.astype(np.int32)
		lines = []
		for shape in shapes:
			line = np.array(shape.points, np.float64)
			if len(line) <= 1:
				continue
			line = (line*[1, -1] + shift) * factor
			line = line.astype(np.int32)
			lines.append(line)
		return lines, bbox


