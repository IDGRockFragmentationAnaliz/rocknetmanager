from pathlib import Path
import numpy as np
import cv2

from .image_data import ImageData
from ..dataset_path_list import DatasetPathList


class ImageTiler:
    def __init__(
            self,
            tile_resolution=(512, 512),
            cropshift=(512, 512),
            lst: DatasetPathList | None = None,
            rotation: bool = False,
    ):
        crop_w, crop_h = map(int, tile_resolution)
        shift_x, shift_y = map(int, cropshift)

        if crop_w <= 0 or crop_h <= 0:
            raise ValueError(
                f"tile_resolution должен быть положительным, получено: {(crop_w, crop_h)}"
            )

        if shift_x <= 0 or shift_y <= 0:
            raise ValueError(
                f"cropshift должен быть положительным, получено: {cropshift}"
            )

        self.crop_w = crop_w
        self.crop_h = crop_h

        self.shift_x = shift_x
        self.shift_y = shift_y

        self.lst = lst
        self.is_rotate = rotation

    def run(
            self,
            data: ImageData,
            bbox=None,
            name_image: str = "",
            save_path=None,
    ):
        lst = self.lst
        crop_w = self.crop_w
        crop_h = self.crop_h
        shift_x = self.shift_x
        shift_y = self.shift_y

        if lst is None:
            raise ValueError("lst должен быть передан")

        root = Path(lst.root)
        save_path = root if save_path is None else Path(save_path)

        height, width = data.image.shape[:2]

        angles = (0, 90, 180, 270) if self.is_rotate else (0,)

        paths = {
            angle: save_path / f"rotate_{angle}"
            for angle in angles
        }

        for path in paths.values():
            path.mkdir(parents=False, exist_ok=True)
            (path / "image").mkdir(parents=False, exist_ok=True)
            (path / "label").mkdir(parents=False, exist_ok=True)

        coordinates = self._calculate_coordinates(
            data=data,
            bbox=bbox,
            shift_x=shift_x,
            shift_y=shift_y,
        )

        image_stem = Path(name_image).stem

        for x, y in coordinates:
            if x < 0 or y < 0:
                continue

            if x + crop_w > width or y + crop_h > height:
                continue

            cropped_data = data.crop_image(x, y, crop_w, crop_h)

            if not cropped_data.is_accessible(threshold=5):
                continue

            rotated_data = cropped_data

            for angle in angles:
                path = paths[angle]

                crop_name = f"{image_stem}_x{x}_y{y}_r{angle}.png"

                path_image = path / "image" / crop_name
                path_label = path / "label" / crop_name

                crop_save(rotated_data, path_image, path_label, lst)

                if self.is_rotate:
                    rotated_data = rotated_data.rotate()

    @staticmethod
    def _calculate_coordinates(
            data: ImageData,
            bbox,
            shift_x: int,
            shift_y: int,
    ) -> list[tuple[int, int]]:
        height, width = data.image.shape[:2]

        if bbox is None:
            x0, y0 = 0, 0
            x1, y1 = width, height
        else:
            x0, y0 = np.floor(bbox[:2]).astype(int)
            x1, y1 = np.ceil(bbox[2:]).astype(int)

        list_x = np.arange(x0, x1, shift_x, dtype=int)
        list_y = np.arange(y0, y1, shift_y, dtype=int)

        coordinates = [(x, y) for y in list_y for x in list_x]

        return coordinates