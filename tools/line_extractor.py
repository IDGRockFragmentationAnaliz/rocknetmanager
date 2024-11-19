import numpy as np
from shapely.geometry import LineString
import shapefile
import skan
import scipy
from scipy.sparse import find
from tqdm import tqdm
from shapely.geometry import LineString


class LineExtractor:
	def __init__(self, csr_graph):
		self.graph = csr_graph
		# Число узлов
		self.num_nodes = self.graph.shape[0]
		# список числа связей узла
		self.num_binds = self.graph.getnnz(axis=0)
		# создание массива связей
		rows, cols = self.graph.nonzero()
		self.binds = set(zip(rows, cols))

	def extruct_bind(self):
		bind = self.binds.pop()
		self.binds.remove((bind[1], bind[0]))
		return bind

	def remove_bind(self, bind):
		self.binds.remove((bind[0], bind[1]))
		self.binds.remove((bind[1], bind[0]))

	def complete_to_head(self, line: list):
		while True:
			next_node = self.find_next_node(line[-1], line[-2])
			if next_node is None:
				return line
			line.append(next_node)
			self.remove_bind((line[-1], line[-2]))
			if next_node == line[0]:
				return line

	def find_next_node(self, node, node_before):
		if self.num_binds[node] == 2:
			binded_nodes = find(self.graph[node, :])[1]
			if binded_nodes[0] == node_before:
				return binded_nodes[1]
			else:
				return binded_nodes[0]
		else:
			return None

	def extruct(self):
		with tqdm(total=len(self.binds), desc="Processing binds") as pbar:
			line_nodes = []
			while len(self.binds) > 0:
				bind = self.extruct_bind()
				line = [bind[0], bind[1]]
				self.complete_to_head(line).reverse()
				if line[0] != line[-1]:
					self.complete_to_head(line)
				line = np.array(line)
				line_nodes.append(line)
				pbar.update((len(line)-1)*2)
			return line_nodes

	@staticmethod
	def convert_nodes_to_coordinates(line_nodes, coordinates):
		lines = []
		for line in tqdm(line_nodes):
			x = coordinates[0][line]
			y = coordinates[1][line]
			line = np.array([y, x]).T
			if len(line) <= 2:
				continue
			line = LineString(line)
			line = line.simplify(1)
			line = line.coords
			line = np.array(line, np.int16)
			lines.append(line)
		return lines


