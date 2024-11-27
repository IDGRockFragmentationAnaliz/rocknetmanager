import cv2
import numpy as np
import sknw
from pathlib import Path
from shapely.geometry import LineString
import shapefile
from skimage import morphology
import networkx as nx
import skan
import scipy
import tqdm
import matplotlib.pyplot as plt
import pickle


def shpline_save(path: Path, polylines):
	path.mkdir(parents=False, exist_ok=True)
	with shapefile.Writer(path / (path.stem + ".shp"), shapefile.POLYLINE) as shp:
		shp.field("NAME", "C")
		for polyline in polylines:
			shp.line([(polyline * [1, -1]).tolist()])
			shp.record("Polyline")


def graph_to_lines(graph):
	polylines = []
	for (s, e, key, data) in graph.edges(keys=True, data=True):
		eds = graph[s][e][key]
		poly = eds['pts']
		poly = LineString(poly)
		poly = poly.simplify(1)
		poly = poly.coords
		poly = np.fliplr(np.array(poly))
		polylines.append(poly)
	return polylines


def filer_corners(image_thin):
	filter_image = (image_thin.copy() / 255).astype(np.int16)
	kernel = np.array([
		[0, 1, 0],
		[0, 10, 1],
		[-1, 0, 0]
	], dtype=np.int16)
	mask = (cv2.filter2D(filter_image, -1, kernel) >= 12)
	image_thin[mask] = 0

	kernel = np.array([
		[-1, 0, 0],
		[0, 10, 1],
		[0, 1, 0]
	], dtype=np.int16)
	mask = (cv2.filter2D(filter_image, -1, kernel) >= 12)
	image_thin[mask] = 0

	kernel = np.array([
		[0, 0, -1],
		[1, 10, 0],
		[0, 1, 0]
	], dtype=np.int16)
	mask = (cv2.filter2D(filter_image, -1, kernel) >= 12)
	image_thin[mask] = 0

	kernel = np.array([
		[0, 1, 0],
		[1, 10, 0],
		[0, 0, -1]
	], dtype=np.int16)
	mask = (cv2.filter2D(filter_image, -1, kernel) >= 12)
	image_thin[mask] = 0

	return image_thin


def vectorize2(image_thin, save_folder: Path):
	image_thin = cv2.ximgproc.thinning(image_thin)
	image_thin = filer_corners(image_thin)
	#skeleton = morphology.skeletonize(image_thin)
	graph, coordinates = skeleton_to_csgraph(image_thin)
	graph = sknw.build_sknw(image_thin[0:2**12, 0:2**12], multi=True)
	polylines = graph_to_lines(graph)
	#shpline_save(save_folder, polylines)
	exit()
	#return polylines


def vectorize(image_thin, save_folder: Path):
	image_thin = cv2.dilate(image_thin, np.ones((3, 3), np.uint8), iterations=1)
	image_thin = cv2.ximgproc.thinning(image_thin)
	#image_thin = filer_corners(image_thin)

	# создание графа
	graph, coordinates = skan.csr.skeleton_to_csgraph(image_thin)
	from .line_extractor import LineExtractor
	line_nodes = LineExtractor(graph).extruct()
	# with open('arrays.pkl', 'wb') as f:
	# 	pickle.dump(line_nodes, f)
	# with open('arrays.pkl', 'rb') as f:
	# 	line_nodes = pickle.load(f)
	lines = LineExtractor.convert_nodes_to_coordinates(line_nodes, coordinates)
	shpline_save(save_folder, lines)
	return lines

