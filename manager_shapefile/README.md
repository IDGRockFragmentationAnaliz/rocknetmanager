# manager_shapefile

Пакет преобразует геометрию между NumPy-массивами в координатах изображения и файлами ESRI Shapefile. Основные сценарии:

- сохранить бинарную маску как полигональный Shapefile;
- загрузить полигональный Shapefile как бинарную маску;
- загрузить линейный Shapefile как бинарную разметку линий;
- напрямую читать и сохранять полигоны и полилинии.

Публичный API пакета, доступный через `rocknetmanager.manager_shapefile`:

```python
from rocknetmanager.manager_shapefile import label_load, mask_load, mask_save
```

## Поток данных

```text
NumPy mask
    └─ mask_save()
        └─ OpenCV contours
            └─ shape_polygon_save()
                └─ POLYGON Shapefile

POLYGON Shapefile
    └─ shape_load()
        └─ mask_load() / mask_prepare()
            └─ uint8 mask со значениями 0 и 255

POLYLINE Shapefile
    └─ shape_load()
        └─ label_load() / label_prepare()
            └─ uint8 label со значениями 0 и 255
```

## Координаты

В массивах точки задаются как `(x, y)` в координатах изображения: начало находится сверху слева, а `y` увеличивается вниз. При сохранении в Shapefile пакет меняет знак `y`; при загрузке выполняется обратное преобразование. Поэтому Shapefile, созданные этим пакетом, корректно возвращаются в координаты исходного изображения.

Это преобразование не является географической привязкой. Пакет не применяет affine transform, не записывает CRS и не создаёт `.prj`.

## Функции

### `mask_save(path, mask, replace=False, epsilon=0.5)`

Сохраняет двумерную маску как `POLYGON` Shapefile.

- все ненулевые пиксели считаются передним планом;
- контуры извлекаются через `cv2.findContours`;
- `epsilon` управляет упрощением контуров через `cv2.approxPolyDP`;
- контуры короче трёх точек пропускаются;
- при `replace=False` существующий Shapefile не перезаписывается и выдаётся `UserWarning`.

Если `path` заканчивается на `.shp`, родительская директория должна существовать. Если передан путь без `.shp`, создаётся директория и файл `<имя директории>/<имя директории>.shp`.

```python
from pathlib import Path

import numpy as np
from rocknetmanager.manager_shapefile import mask_save

mask = np.zeros((512, 512), dtype=np.uint8)
mask[100:300, 150:400] = 255

output_path = Path("markup/polygons.shp")
output_path.parent.mkdir(exist_ok=True)

mask_save(
    output_path,
    mask,
    replace=True,
    epsilon=0.5,
)
```

### `mask_load(path, shape)`

Загружает `POLYGON`, `POLYGONZ` или `POLYGONM` Shapefile и растеризует полигоны через `cv2.fillPoly`. Возвращает двумерный `np.uint8` массив размера `shape[:2]` со значениями `0` и `255`.

```python
from pathlib import Path

from rocknetmanager.manager_shapefile import mask_load

mask = mask_load(
    Path("markup/polygons.shp"),
    shape=(512, 512),
)
```

Для Shapefile другого типа функция выбрасывает `ValueError`.

### `label_load(path, shape, thickness=1)`

Загружает `POLYLINE`, `POLYLINEZ` или `POLYLINEM` Shapefile и рисует линии через `cv2.polylines`. Возвращает двумерный `np.uint8` массив размера `shape[:2]` со значениями `0` и `255`.

```python
from pathlib import Path

from rocknetmanager.manager_shapefile import label_load

label = label_load(
    Path("markup/edges.shp"),
    shape=(512, 512),
    thickness=2,
)
```

`thickness` должен быть положительным.

## Низкоуровневые функции

Эти функции не экспортируются из `manager_shapefile.__init__`, но доступны из соответствующих модулей:

- `shape_load(path)` из `shape_load.py` — читает первый `.shp` из директории либо указанный файл и возвращает `(lines, shape_type)`;
- `shape_polygon_save(path, polygons, replace=False)` из `shape_save.py` — сохраняет последовательность массивов точек как `POLYGON`;
- `shape_polyline_save(path, polylines)` из `shape_save.py` — сохраняет последовательность массивов точек как `POLYLINE`;
- `mask_prepare(lines, shape)` из `mask_load.py` — растеризует уже загруженные полигоны;
- `label_prepare(lines, shape, thickness=1)` из `mask_load.py` — растеризует уже загруженные полилинии.

Пример прямой работы с геометрией:

```python
from pathlib import Path

import numpy as np
from rocknetmanager.manager_shapefile.shape_load import shape_load
from rocknetmanager.manager_shapefile.shape_save import shape_polygon_save

polygon = np.array([
    [10, 10],
    [100, 10],
    [100, 80],
    [10, 80],
], dtype=np.int32)

output_path = Path("markup/polygon.shp")
output_path.parent.mkdir(exist_ok=True)

shape_polygon_save(
    output_path,
    [polygon],
    replace=True,
)

lines, shape_type = shape_load(output_path)
```

## Формат Shapefile

Shapefile состоит минимум из файлов `.shp`, `.shx` и `.dbf` с одинаковым именем. При принудительной перезаписи `shape_polygon_save` также удаляет найденные сопутствующие `.prj`, `.cpg`, `.qix`, `.sbn` и `.sbx`, чтобы не оставить файлы от предыдущей версии.

Каждая записанная геометрия получает текстовое поле `NAME` со значением `Polygon` или `Polyline`.

## Текущие ограничения

- CRS и географическая привязка не поддерживаются.
- `shape_load` использует `shape.points` без обработки `shape.parts`, поэтому multipart-геометрия может быть объединена в одну последовательность точек.
- При передаче директории `shape_load` выбирает первый найденный `.shp`; директория должна содержать только нужный набор данных.
- Внутренние кольца и отверстия масок могут восстановиться некорректно: контуры сохраняются отдельно, а загрузка заполняет каждый полигон значением `255`.
- Координаты при загрузке округляются приведением к `np.int32`.

## Зависимости

- `numpy`;
- `opencv-python`/`opencv-contrib-python` (`cv2`);
- `pyshp` (импортируется как `shapefile`).
