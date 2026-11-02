Batch tiling for broccoli head center annotation

Script: batch_tile_160.py
Default input: the script directory, searched recursively for JPG/JPEG/PNG/TIF/TIFF files.
Default output: annotation_tiles_160 inside the script directory. This output directory is excluded from input discovery.
Requirements: Python 3.10+, Pillow, and numpy.

Installation and usage:
python -m pip install pillow numpy
python batch_tile_160.py

Default options: --size 160 --overlap 40 --workers 4 --filter conservative
The stride is 120 pixels. The last row and column align with the image boundary, so their overlap can exceed 40 pixels.
Crops preserve original pixels without enlargement. Images smaller than 160 pixels are padded with black on the right/bottom; valid_size records the unpadded dimensions.

Filtering:
This batch does not reuse the manually drawn planting boundaries from the previous six images.
The conservative filter uses blue-green leaf color cues at reduced resolution and expands the mask by approximately 64 original-image pixels to exclude obvious bare-soil tiles.
This is a conservative heuristic, not a broccoli head detector or precise planting-area segmentation. It may retain weeds, grass, and empty tiles, and may miss targets with unusual colors.
The overviews directory contains source-image previews with red rectangles indicating excluded tiles. Review these previews for filtering errors.
To disable filtering entirely:
python batch_tile_160.py --filter none
Rerunning overwrites matching tiles and removes obsolete PNG files listed in previous manifests. Original images are not modified.
Do not change tiling parameters and overwrite images after annotation: the same filename may then refer to a different source region.

Outputs and annotation:
Each source image has its own output subdirectory containing PNG tiles and manifest.json.
summary.json records source paths, retained/excluded counts, parameters, and errors.
Annotate every confidently identifiable head center in each tile. Grid tiling does not guarantee exactly one plant per tile.
Original-image point = tile-local point + the top-left coordinates in box_xyxy.
When merging points back into a source image, keep only points inside core_local_xyxy (left/top inclusive, right/bottom exclusive) to avoid overlap-based double counting.
Coordinates refer to the EXIF-oriented source image. No registration or plant deduplication is performed across different source images.
Split training and validation data by source image or capture area to prevent leakage from overlapping views.
