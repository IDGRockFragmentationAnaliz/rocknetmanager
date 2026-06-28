from pathlib import Path
import sys
import csv
import pandas as pd


class DatasetPathList:
    def __init__(
        self,
        root_path: Path | None = None,
        save_path: Path | None = None,
    ):
        self.root = Path(root_path).resolve() if root_path is not None else None
        self.save_path = Path(save_path) if save_path is not None else None
        self.lst = pd.DataFrame(columns=["images", "labels", "masks"])

        self._file = None
        self._writer = None
        self._is_streaming = False

    def __enter__(self):
        if self.save_path is not None:
            self.save_path.parent.mkdir(parents=True, exist_ok=True)

            self._file = open(
                self.save_path,
                mode="w",
                encoding="utf-8",
                newline="",
            )

            self._writer = csv.writer(
                self._file,
                delimiter="\t",
                lineterminator="\n",
            )

            self._is_streaming = True

        return self

    def __exit__(self, exc_type, exc, traceback):
        self._close_stream()

        # Ошибку не подавляем
        return False

    def add(
        self,
        path_image: Path,
        path_label: Path,
        path_mask: Path | None = None,
    ):
        row = {
            "images": self._prepare_path(path_image),
            "labels": self._prepare_path(path_label),
            "masks": (
                self._prepare_path(path_mask)
                if path_mask is not None
                else pd.NA
            ),
        }

        self.lst.loc[len(self.lst)] = row

        if self._is_streaming:
            self._write_row(row)

    def save(self, save_path: Path | None = None):
        if save_path is not None:
            self.save_path = Path(save_path)

        if self.save_path is None:
            raise ValueError("save_path не задан")

        # Если файл уже писался напрямую внутри with,
        # то повторно перезаписывать его не нужно.
        if self._is_streaming:
            self._file.flush()
            return

        save_path = self.save_path
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

    def _write_row(self, row: dict):
        if self._writer is None or self._file is None:
            raise RuntimeError("Файл для записи не открыт")

        # В streaming-режиме всегда пишем 3 колонки.
        # Если mask нет, третья колонка будет пустой.
        self._writer.writerow([
            row["images"],
            row["labels"],
            "" if pd.isna(row["masks"]) else row["masks"],
        ])

        # Важно: сразу сбрасываем Python-буфер.
        # Тогда при Stop в PyCharm уже записанные строки обычно остаются в файле.
        self._file.flush()

    def _close_stream(self):
        if self._file is not None:
            try:
                self._file.flush()
            finally:
                self._file.close()

        self._file = None
        self._writer = None

    def _prepare_path(self, path: Path) -> str:
        path = Path(path)

        if self.root is None:
            return str(path)

        if not path.is_absolute():
            return str(path)

        try:
            return str(path.resolve().relative_to(self.root))
        except ValueError:
            raise ValueError(
                f"Путь {path} не лежит внутри root_path={self.root}"
            )

    @staticmethod
    def load(
        lst_path: Path,
        root_path: Path | None = None,
    ) -> "DatasetPathList":
        lst_path = Path(lst_path)

        if root_path is None:
            root_path = lst_path.parent

        dataset = DatasetPathList(
            root_path=root_path,
            save_path=lst_path,
        )

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