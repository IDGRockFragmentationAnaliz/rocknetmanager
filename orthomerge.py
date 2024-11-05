import rasterio
from rasterio.merge import merge
from rasterio.plot import reshape_as_raster
from pathlib import Path

mateo_root = Path("D:/1.ToSaver/profileimages/Matteo_database")

images_path = mateo_root / Path("Site_A/images_uint8")
path_images = [path for path in images_path.iterdir()]


src_files = [rasterio.open(f) for f in path_images]
mosaic, out_trans = merge(src_files)

out_meta = src_files[0].meta.copy()
out_meta.update({"height": mosaic.shape[1],
                 "width": mosaic.shape[2],
                 "transform": out_trans})

with rasterio.open('mosaic.tif', 'w', **out_meta) as dest:
    dest.write(mosaic)


