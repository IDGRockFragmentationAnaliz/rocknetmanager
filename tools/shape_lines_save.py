import cv2
import numpy as np
import sknw
from pathlib import Path
from shapely.geometry import LineString
import shapefile


def shape_line_save(path: Path, polylines):
	path.mkdir(parents=False, exist_ok=True)
	with shapefile.Writer(path / (path.stem + ".shp"), shapefile.POLYLINE) as shp:
		shp.field("NAME", "C")
		for polyline in polylines:
			shp.line([(polyline * [1, -1]).tolist()])
			shp.record("Polyline")
