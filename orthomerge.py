from pathlib import Path
from tools import mosaic_merge

mateo_root = Path("D:/1.ToSaver/profileimages/Matteo_database")

images_path = mateo_root / Path("Site_A/images_uint8")

mosaic_merge(images_path)
