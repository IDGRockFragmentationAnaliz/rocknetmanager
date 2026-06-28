from pathlib import Path
import pandas as pd


class DatasetPathList:
	def __init__(self, root):
		self.root = Path(root)
		self.lst = pd.DataFrame(columns=["images", "labels"])

	def add(self, path_image, path_label):
		self.lst.loc[len(self.lst)] = {
			"images": str(path_image),
			"labels": str(path_label),
		}

	def save(self, save_path=None):
		if save_path is None:
			save_path = self.root / "train.lst"
		else:
			save_path = Path(save_path)

		save_path.parent.mkdir(parents=True, exist_ok=True)

		self.lst.to_csv(
			str(save_path),
			sep="\t",
			index=False,
			header=False,
		)

	def __len__(self):
		return len(self.lst)