# rocknetmanager

```python
import rocknetmanager.tools as nt




def main():
    mateo_root = Path("D:/1.ToSaver/profileimages/Matteo_database")
    images_path = mateo_root / Path("Site_A/images")
    path_images = [path for path in images_path.iterdir()]
    path_to_save = Path("D:/1.ToSaver/profileimages/Matteo_database/Site_A/Images_uint8")
    for path in path_images:
        nt.geotif2geotif_standart(path, path_to_save)
```