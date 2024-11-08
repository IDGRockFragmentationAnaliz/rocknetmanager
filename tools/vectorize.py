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


def shpline_save(path, polylines):
	with shapefile.Writer(path, shapefile.POLYLINE) as shp:
		shp.field("NAME", "C")
		for polyline in polylines:
			shp.line([(polyline * [1, -1]).tolist()])
			shp.record("Polyline")
#poly = LineString(eds['pts'])
		#poly = poly.simplify(1)
		#poly = poly.coords
		#poly = np.flip(np.array(poly))


def graph_to_lines(graph):
	polylines = []
	for (s, e, key, data) in graph.edges(keys=True, data=True):
		eds = graph[s][e][key]
		poly = eds['pts']
		polylines.append(poly)
	return polylines

def filer_corners(image_thin):
	filter_image = (image_thin.copy()/255).astype(np.int16)
	kernel = np.array([
		[0, 1, 0],
		[0, 10, 1],
		[-1, 0, 0]
	], dtype=np.int16)
	filtered = cv2.filter2D(filter_image, -1, kernel)
	mask = (filtered >= 12)
	image_thin[mask] = 0
	print(np.sum(mask))
	return image_thin

def vectorize(image_thin):
	image_thin = cv2.ximgproc.thinning(image_thin)
	image_thin = filer_corners(image_thin)

	graph = sknw.build_sknw(image_thin, multi=True)
	polylines = graph_to_lines(graph)

	fig = plt.figure(figsize=(7, 5))
	axs = [fig.add_subplot(1, 2, 1),
	       fig.add_subplot(1, 2, 2)]
	axs[0].imshow(image_thin, cmap='gray')
	axs[1].imshow(image_thin, cmap='gray')
	for polyline in polylines:
		axs[1].plot(polyline[:,1], polyline[:,0])#, color="blue"
	plt.show()

	#shpline_save("shape/test.shp", polylines)





