import matplotlib.pyplot as plt
import cv2
import numpy as np
import sknw
from shapely.geometry import LineString
import shapefile


def vectorize2(image_thin):
	image_thin = cv2.ximgproc.thinning(image_thin)
	contours, _ = cv2.findContours(image_thin, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_KCOS)

	polygons = []
	for contour in contours:
		poly = contour.squeeze(axis=1)
		if len(poly) > 2:
			polygons.append(poly)

	fig = plt.figure(figsize=(7, 9))
	axs = [fig.add_subplot(1, 1, 1)]
	axs[0].imshow(cv2.merge((image_thin, image_thin, image_thin)))

	for poly in polygons:
		axs[0].plot(poly[:, 0], poly[:, 1], '.-', color="blue")
	plt.show()


def vectorize(image_thin):
	image_thin = cv2.ximgproc.thinning(image_thin)
	graph = sknw.build_sknw(image_thin, multi=False)

	polygons = []
	for (s, e) in graph.edges():
		eds = graph[s][e]
		poly = LineString(eds['pts'])
		poly = poly.simplify(2)
		polygons.append(np.array(poly.coords).tolist())


	# fig = plt.figure(figsize=(7, 9))
	# axs = [fig.add_subplot(1, 1, 1)]
	# axs[0].imshow(image_thin, cmap='gray')
	#
	# for poly in polygons:
	# 	axs[0].plot(poly[:, 1], poly[:, 0], '.-', color="blue")
	#
	# plt.show()

	with shapefile.Writer("shape/test.shp", shapefile.POLYGON) as shp:
		shp.field("NAME", "C")
		shp.poly(polygons)
		shp.record("Polygon")



