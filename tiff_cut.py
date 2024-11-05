import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import rasterio

mateo_root = Path("D:/1.ToSaver/profileimages/Matteo_database")
images_path = mateo_root / Path("Site_A/images")
path_images = [path for path in images_path.iterdir()]


def convert_tiff_to_uint8(input_path: Path, output_folder: Path):
	with rasterio.open(input_path) as tif:
		meta = tif.meta.copy()
		image = tif.read([1, 2, 3]).astype(np.uint8)
		meta.update({
			"dtype": rasterio.uint8,
			'count': 3
		})

		with rasterio.open(output_folder / f"{input_path.stem}.tif", 'w', **meta) as dst:
			dst.write(image)


def main():
	path_to_save = Path("D:/1.ToSaver/profileimages/Matteo_database/Site_A/Images_uint8")
	for path in path_images:
		convert_tiff_to_uint8(path, path_to_save)


# image = cv2.imread(path_image, cv2.IMREAD_UNCHANGED)
# print("Загрузка заверешена")
#
# image = np.uint8(image/np.max(image)*255)[:, :, 0:3]
#
# fig = plt.figure(figsize=(7, 9))
# axs = [fig.add_subplot(1, 1, 1)]
# axs[0].imshow(image[:, :, 0:3])
# plt.show()
#
# cv2.imwrite("test.png", image)


if __name__ == '__main__':
	main()
