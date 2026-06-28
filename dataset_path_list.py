from pathlib import Path
import pandas as pd


class DatasetPathList:
    def __init__(self, root_path: Path | None = None):
        self.root_path = Path(root_path).resolve() if root_path is not None else None
        self.lst = pd.DataFrame(columns=["images", "labels", "masks"])

    def add(
        self,
        path_image: Path,
        path_label: Path,
        path_mask: Path | None = None,
    ):
        row = {
            "images": self._prepare_path(path_image),
            "labels": self._prepare_path(path_label),
        }

        if path_mask is not None:
            row["masks"] = self._prepare_path(path_mask)

        self.lst.loc[len(self.lst)] = row

    def save(self, save_path: Path):
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        columns = ["images", "labels"]

        if self.lst["masks"].notna().any():
            columns.append("masks")

        self.lst.to_csv(
            str(save_path),
            sep="\t",
            index=False,
            header=False,
            columns=columns,
        )

    def _prepare_path(self, path: Path) -> str:
        path = Path(path)

        if self.root_path is None:
            return str(path)

        if not path.is_absolute():
            return str(path)

        try:
            return str(path.resolve().relative_to(self.root_path))
        except ValueError:
            raise ValueError(
                f"Путь {path} не лежит внутри root_path={self.root_path}"
            )

    @staticmethod
    def load(lst_path: Path, root_path: Path | None = None) -> "DatasetPathList":
        lst_path = Path(lst_path)

        if root_path is None:
            root_path = lst_path.parent

        dataset = DatasetPathList(root_path=root_path)

        try:
            df = pd.read_csv(
                str(lst_path),
                sep="\t",
                header=None,
                dtype=str,
            )
        except pd.errors.EmptyDataError:
            dataset.lst = pd.DataFrame(columns=["images", "labels", "masks"])
            return dataset

        if df.shape[1] == 2:
            df.columns = ["images", "labels"]
            df["masks"] = pd.NA
        elif df.shape[1] == 3:
            df.columns = ["images", "labels", "masks"]
        else:
            raise ValueError(
                f"Ожидалось 2 или 3 колонки в {lst_path}, получено {df.shape[1]}"
            )

        dataset.lst = df[["images", "labels", "masks"]]
        return dataset

    def __len__(self):
        return len(self.lst)