import numpy as np
import cv2
from pathlib import Path
from rocknetmanager.tools.shape_load import shape_load


class ImageData:
    def __init__(self, image, label, mask):
        self.image = image
        self.label = label
        self.mask = mask

        if self.image is None:
            raise ValueError("image не должен быть None")

        if self.label is None:
            raise ValueError("label не должен быть None")

    def crop_image(self, x, y, dx, dy):
        image = self.crop(self.image, x, y, dx, dy)
        label = self.crop(self.label, x, y, dx, dy)
        if self.mask is not None:
            mask = self.crop(self.mask, x, y, dx, dy)
        else:
            mask = None

        return self.__class__(image, label, mask)

    def is_in_bound(self, percent=0.4):
        if self.mask is None or self.mask.size == 0:
            return True

        bound_area = np.prod(self.mask.shape)
        count = cv2.countNonZero(self.mask)

        return count / bound_area > percent

    def is_have_label(self):
        count = cv2.countNonZero(self.label)
        return count > 9

    def is_accessible(self):
        return self.is_in_bound() and self.is_have_label()


    def rotate(self):
        self.image = cv2.rotate(self.image, cv2.ROTATE_90_CLOCKWISE)
        self.label = cv2.rotate(self.label, cv2.ROTATE_90_CLOCKWISE)
        if self.mask is not None:
            self.mask = cv2.rotate(self.mask, cv2.ROTATE_90_CLOCKWISE)
        return self

    def save(self, path_image: Path, path_label: Path):
        path_image = Path(path_image)
        path_label = Path(path_label)

        path_image.parent.mkdir(parents=True, exist_ok=True)
        path_label.parent.mkdir(parents=True, exist_ok=True)

        ok_image = cv2.imwrite(
            str(path_image),
            cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)
        )
        if not ok_image:
            raise IOError(f"Не удалось сохранить изображение: {path_image}")

        ok_label = cv2.imwrite(str(path_label), self.label)
        if not ok_label:
            raise IOError(f"Не удалось сохранить label: {path_label}")

    @classmethod
    def from_vectors(cls, image, polies_label, polies_mask=None):
        height, width = image.shape[:2]

        label = np.zeros((height, width), dtype=np.uint8)
        if polies_label is not None and len(polies_label) > 0:
            label = cv2.polylines(
                label,
                polies_label,
                isClosed=False,
                color=255,
                thickness=3,
            )

        mask = None
        if polies_mask is not None and len(polies_mask) > 0:
            mask = np.zeros((height, width), dtype=np.uint8)
            mask = cv2.fillPoly(
                mask,
                polies_mask,
                color=255,
            )

        return cls(image, label, mask)

    @classmethod
    def load(cls, path_image, path_labels, path_mask=None):
        path_image = Path(path_image)
        path_labels = Path(path_labels)

        if path_mask is not None:
            path_mask = Path(path_mask)

        if not path_image.exists():
            raise FileNotFoundError(f"Не найден файл изображения: {path_image}")

        if not path_labels.exists():
            raise FileNotFoundError(f"Не найден файл векторов трасс: {path_labels}")

        if path_mask is not None and not path_mask.exists():
            raise FileNotFoundError(f"Не найден файл векторов маски: {path_mask}")

        image = cv2.imread(str(path_image))
        if image is None:
            raise ValueError(f"OpenCV не смог прочитать изображение: {path_image}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        polies_label, bbox_labels = shape_load(path_labels)

        polies_mask = None
        bbox_mask = None

        if path_mask is not None:
            polies_mask, bbox_mask = shape_load(path_mask)

        obj = cls.from_vectors(
            image,
            polies_label=polies_label,
            polies_mask=polies_mask,
        )

        bbox = bbox_mask if bbox_mask is not None else bbox_labels

        return obj, bbox

    @staticmethod
    def crop(image, x, y, dx, dy):
        return image[y:y + dy, x:x + dx]
