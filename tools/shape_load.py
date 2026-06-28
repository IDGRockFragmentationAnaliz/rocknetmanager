import numpy as np
from pathlib import Path
import shapefile


def shape_load(path: Path, factor=1, shift=np.zeros((1, 2))):
    if path.is_dir() and path.exists():
        for file in path.iterdir():
            if file.suffix == ".shp":
                path = file
                break

    shift = np.asarray(shift, dtype=np.float64)

    if shift.shape == (2,):
        shift = shift.reshape(1, 2)

    with shapefile.Reader(str(path)) as shp:
        shapes = shp.shapes()

        raw_bbox = np.array(shp.bbox, dtype=np.float64)
        x_min, y_min, x_max, y_max = raw_bbox

        bbox = np.array(
            [
                x_min + shift[0, 0],
                -y_max + shift[0, 1],
                x_max + shift[0, 0],
                -y_min + shift[0, 1],
            ],
            dtype=np.float64,
        ) * factor

        bbox = bbox.astype(np.int32)

        lines = []

        for shape in shapes:
            line = np.array(shape.points, dtype=np.float64)

            if len(line) <= 1:
                continue

            line = (line * [1, -1] + shift) * factor
            line = line.astype(np.int32)

            lines.append(line)

        return lines, bbox


