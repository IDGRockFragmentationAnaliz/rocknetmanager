from pathlib import Path
import matplotlib.pyplot as plt
import cv2
import numpy as np
import rasterio
from rasterio.merge import merge


def geotif2geotif_standart(input_path: Path, output_folder: Path):
	with rasterio.open(input_path) as tif:
		meta = tif.meta.copy()
		image = tif.read([1, 2, 3]).astype(np.uint8)
		meta.update({
			"dtype": rasterio.uint8,
			'count': 3
		})

		with rasterio.open(
			output_folder / f"{input_path.stem}.tif",
			'w',
			**meta
		) as dst:
			dst.write(image)

def put_labels(geo_mosaic, geo_parts, geo_paths):
	np_mosaic = geo_mosaic.read()
	print(geo_mosaic.meta)
	np_mosaic = np.transpose(np_mosaic, (1, 2, 0))
	np_mosaic = np.ascontiguousarray(np_mosaic)

	bounds = geo_mosaic.bounds
	dx = (bounds.right - bounds.left)
	dy = (bounds.top - bounds.bottom)
	print(np_mosaic.shape)
	print(dx / 0.0005)
	print(dy / 0.0005)

	bb = geo_parts[0].bounds
	center_x = (bb.left + bb.right) / 2 - bounds.left
	center_y = (bb.top + bb.bottom) / 2 - bounds.bottom

	col = int(center_x / 0.0005)
	row = -int(center_y / 0.0005)
	print(col, row)
	cv2.putText(
		np_mosaic,
		geo_paths[0].stem,
		(col, row),
		cv2.FONT_HERSHEY_SIMPLEX,
		5,
		(0, 0, 255),
		2,
		cv2.LINE_AA
	)
	np_mosaic = np.transpose(np_mosaic, (2, 0, 1))
	geo_mosaic.write(np_mosaic)



def mosaic_merge(images_folder: Path, save_path='mosaic.tif', labels=False):
	path_images = [path for path in images_folder.iterdir()]
	geo_images = [rasterio.open(f) for f in path_images]
	mosaic, out_trans = merge(geo_images)


	out_meta = geo_images[0].meta.copy()
	out_meta.update({
		"height": mosaic.shape[1],
		"width": mosaic.shape[2],
		"transform": out_trans
	})
	with rasterio.open(save_path, 'r+', **out_meta) as geo_mosaic:
		geo_mosaic.write(mosaic)
		put_labels(geo_mosaic, geo_images, path_images)
