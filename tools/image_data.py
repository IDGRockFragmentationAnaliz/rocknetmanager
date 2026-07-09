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

    def is_have_label(self, threshold=25):
        count = cv2.countNonZero(self.label)
        return count > threshold

    def is_accessible(self, threshold=25):
        return self.is_in_bound() and self.is_have_label(threshold)


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

    @staticmethod
    def crop(image, x, y, dx, dy):
        return image[y:y + dy, x:x + dx]

    @classmethod
    def load(cls, path_image, path_labels, path_mask=None, thickness=None):
        path_image = Path(path_image)
        path_labels = Path(path_labels)

        if path_mask is not None:
            path_mask = Path(path_mask)

        if not path_image.exists():
            raise FileNotFoundError(f"Не найден файл изображения: {path_image}")

        if not path_labels.exists():
            raise FileNotFoundError(f"Не найден файл label: {path_labels}")

        if path_mask is not None and not path_mask.exists():
            raise FileNotFoundError(f"Не найден файл mask: {path_mask}")

        image = cv2.imread(str(path_image))
        if image is None:
            raise ValueError(f"OpenCV не смог прочитать изображение: {path_image}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label, bbox_labels = cls.load_label(
            path_label=path_labels,
            image_shape=image.shape,
            thickness=thickness
        )

        mask, bbox_mask = cls.load_mask(
            path_mask=path_mask,
            image_shape=image.shape,
        )

        bbox = bbox_mask if bbox_mask is not None else bbox_labels

        return cls(image, label, mask), bbox

    @staticmethod
    def load_label_from_vector(path_labels, image_shape, thickness=None):
        thickness = thickness if thickness is not None else 3
        path_labels = Path(path_labels)

        if not path_labels.exists():
            raise FileNotFoundError(f"Не найден файл векторов label: {path_labels}")

        height, width = image_shape[:2]

        polies_label, bbox = shape_load(path_labels)

        label = np.zeros((height, width), dtype=np.uint8)

        if polies_label is not None and len(polies_label) > 0:
            label = cv2.polylines(
                label,
                polies_label,
                isClosed=False,
                color=255,
                thickness=thickness,
            )

        return label, bbox

    @staticmethod
    def load_label(path_label, image_shape, thickness=None):
        path_label = Path(path_label)

        if path_label.is_dir() or path_label.suffix.lower() == ".shp":
            return ImageData.load_label_from_vector(
                path_labels=path_label,
                image_shape=image_shape,
                thickness=thickness
            )

        if ImageData.is_image_path(path_label):
            label = ImageData.load_label_from_image(
                path_label=path_label,
                image_shape=image_shape,
            )
            return label, None

        raise ValueError(
            f"Неподдерживаемый формат label: {path_label}. "
            f"Ожидалась папка, .shp или файл изображения."
        )

    @staticmethod
    def load_mask(path_mask, image_shape):
        if path_mask is None:
            return None, None

        path_mask = Path(path_mask)

        if path_mask.is_dir() or path_mask.suffix.lower() == ".shp":
            return ImageData.load_mask_from_vector(
                path_mask=path_mask,
                image_shape=image_shape,
            )

        if ImageData.is_image_path(path_mask):
            mask = ImageData.load_mask_from_image(
                path_mask=path_mask,
                image_shape=image_shape,
            )
            return mask, None

        raise ValueError(
            f"Неподдерживаемый формат mask: {path_mask}. "
            f"Ожидалась папка, .shp или файл изображения."
        )


    @staticmethod
    def load_label_from_image(path_label, image_shape=None):
        path_label = Path(path_label)

        if not path_label.exists():
            raise FileNotFoundError(f"Не найден файл label: {path_label}")

        label = cv2.imread(str(path_label), cv2.IMREAD_GRAYSCALE)

        if label is None:
            raise ValueError(f"OpenCV не смог прочитать label: {path_label}")

        if image_shape is not None:
            height, width = image_shape[:2]

            if label.shape != (height, width):
                raise ValueError(
                    f"Размер label не совпадает с image: "
                    f"label={label.shape}, image={(height, width)}"
                )

        return label

    @staticmethod
    def load_mask_from_vector(path_mask, image_shape):
        if path_mask is None:
            return None, None

        path_mask = Path(path_mask)

        if not path_mask.exists():
            raise FileNotFoundError(f"Не найден файл векторов mask: {path_mask}")

        height, width = image_shape[:2]

        polies_mask, bbox = shape_load(path_mask)

        mask = None
        if polies_mask is not None and len(polies_mask) > 0:
            mask = np.zeros((height, width), dtype=np.uint8)
            mask = cv2.fillPoly(
                mask,
                polies_mask,
                color=255,
            )

        return mask, bbox

    @staticmethod
    def load_mask_from_image(path_mask, image_shape=None):
        if path_mask is None:
            return None

        path_mask = Path(path_mask)

        if not path_mask.exists():
            raise FileNotFoundError(f"Не найден файл mask: {path_mask}")

        mask = cv2.imread(str(path_mask), cv2.IMREAD_GRAYSCALE)

        if mask is None:
            raise ValueError(f"OpenCV не смог прочитать mask: {path_mask}")

        if image_shape is not None:
            height, width = image_shape[:2]

            if mask.shape != (height, width):
                raise ValueError(
                    f"Размер mask не совпадает с image: "
                    f"mask={mask.shape}, image={(height, width)}"
                )

        return mask

    @staticmethod
    def is_image_path(path):
        return Path(path).suffix.lower() in {
            ".png",
            ".jpg",
            ".jpeg",
            ".bmp",
            ".tif",
            ".tiff",
        }