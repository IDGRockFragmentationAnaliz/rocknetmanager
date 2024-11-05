from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import rasterio


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


def mosaic_merge(images_folder: Path, save_path='mosaic.tif'):
	path_images = [path for path in images_folder.iterdir()]
	geo_images = [rasterio.open(f) for f in path_images]
	mosaic, out_trans = merge(geo_images)
	out_meta = geo_images[0].meta.copy()
	out_meta.update({
		"height": mosaic.shape[1],
		"width": mosaic.shape[2],
		"transform": out_trans
	})
	with rasterio.open(save_path, 'w', **out_meta) as dest:
		dest.write(mosaic)
