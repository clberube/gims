# gims
Python command line interface for **g**eospatial **im**age **s**egmentation.

![img](/assets/img/heatmap-outcrops.png)
<p align="center">
  <i>Heatmap of bedrock exposure in the Baie-James area, northern Québec.</i>
</p>

## Application
See [my article](https://medium.com/@charleslberube/orthophoto-segmentation-for-outcrop-detection-in-the-boreal-forest-679c3071d51f?source=friends_link&sk=003ef605211c68e12ae3879edb5e81e1) covering this project to assist geological field work planning with drone photography.

## How it works
- Implements the UNet deep learning architecture for supervised segmentation of GIS data.
- Can be trained, validated and used for predictions in any GIS-related segmentation task.
- Uses any n-band GeoTIFF image as input data.
- Uses any multi-polygon shapefile as target data.
- Comes with pre-trained weights for waterbody and bare land detection in the boreal forest.

## Dependencies
- [PyTorch](https://pytorch.org/)
- [GDAL](https://gdal.org/index.html)
- [numpy](https://numpy.org/)
- [rasterio](https://rasterio.readthedocs.io/en/latest/)
- [h5py](https://www.h5py.org/)

## Installation

```console
$ git clone https://github.com/clberube/gims
$ cd gims
$ python setup.py install -f
```

## Training a model
This example shows how to train a model using an input GeoTIFF and target shapefiles. Training weights are automatically saved with their timestamp as `saved_weights/YYYY-MM-DDTHH-MM-SS.pt`.

#### Script location
```console
./train_model.py
```

#### Help
```
usage: train_model.py [-h] [-a] [-s] [-t] [-p] [-b] [--username USERNAME]
                      [--tile-size TILE_SIZE] [--tile-step TILE_STEP]
                      [--val-size VAL_SIZE] [--threshold THRESHOLD]
                      [--max-epochs MAX_EPOCHS]
                      [--learning-rate LEARNING_RATE]
                      [--batch-size BATCH_SIZE]
                      [--class-weights CLASS_WEIGHTS [CLASS_WEIGHTS ...]]
                      [--input-channels INPUT_CHANNELS [INPUT_CHANNELS ...]]
                      raster vectors [vectors ...]

positional arguments:
  raster                path to the input raster file
  vectors               paths to the target vector files

optional arguments:
  -h, --help            show this help message and exit
  -a, --augmentation    toggle image augmentation to training process
                        (default: False)
  -s, --scheduler       toggle use of learning rate scheduler (default: False)
  -t, --test-auto       toggle automatic testing on the South block (default:
                        False)
  -p, --push-git        toggle automatic git push -- ONLY USE IF YOU KNOW WHAT
                        YOU ARE DOING (default: False)
  -b, --batch-norm      toggle batch normalization (default: True)
  --username USERNAME   name of current user (default: $USERNAME)
  --tile-size TILE_SIZE
                        pixel size of the tiles (default: 128)
  --tile-step TILE_STEP
                        pixel step between two consecutive tiles (default:
                        None)
  --val-size VAL_SIZE   fraction size of the validation subset (default: 0.2)
  --threshold THRESHOLD
                        probability threshold to compute IoU (default: 0.5)
  --max-epochs MAX_EPOCHS
                        number of epochs to train for (default: 20)
  --learning-rate LEARNING_RATE
                        training learning rate (default: 0.0001)
  --batch-size BATCH_SIZE
                        number of tiles in each training batch (default: 32)
  --class-weights CLASS_WEIGHTS [CLASS_WEIGHTS ...]
                        relative weights of the target classes (default: None)
  --input-channels INPUT_CHANNELS [INPUT_CHANNELS ...]
                        indices of the image channels to train on (default:
                        None)
```

#### Example
```console
$ python train_model.py /path/to/rasters/TRAIN_Mosaic_1m.tif /path/to/vectors/TRAIN_waterbodies.shp /path/to/vectors/TRAIN_bareland.shp  --input-channels 0 1 2 --max-epochs 20
```

## Testing a model
This example shows how to test a previously trained model using an arbitrary input GeoTIFF and target shapefiles.

#### Script location
```console
./test_model.py
```

#### Help
```
usage: test_model.py [-h] [--username USERNAME] [--threshold THRESHOLD]
                     [--batch-size BATCH_SIZE] [--tile-size TILE_SIZE]
                     [--tile-step TILE_STEP]
                     [--input-channels INPUT_CHANNELS [INPUT_CHANNELS ...]]
                     [--robust-iou]
                     model raster vectors [vectors ...]

positional arguments:
  model                 path to the pretrained model
  raster                path to the input test raster
  vectors               paths to the target test vectors

optional arguments:
  -h, --help            show this help message and exit
  --username USERNAME   name of current user (Default: $USERNAME)
  --threshold THRESHOLD
                        probability threshold to compute IoU
  --batch-size BATCH_SIZE
                        number of tiles in each testing batch
  --tile-size TILE_SIZE
                        pixel size of the tiles
  --tile-step TILE_STEP
                        pixel step between two consecutive tiles
  --input-channels INPUT_CHANNELS [INPUT_CHANNELS ...]
                        indices of the image channels to test on
  --robust-iou          toggle to compute IoU only on tiles where class is
                        present (Default: False)
```

#### Example
```console
$ python test_model.py ./saved_weights/YYYY-MM-DDTHH-MM-SS.pt /path/to/rasters/TEST_Mosaic_1m.tif /path/to/vectors/TEST_waterbodies.shp /path/to/vectors/test/TEST_bareland.shp --input-channels 0 1 2
```

## Predicting targets
This example shows how to predict targets with a previously trained model on a new input GeoTIFF.

#### Script location
```console
./predict_targets.py
```

#### Help
```
usage: predict_targets.py [-h] [-p] [--username USERNAME]
                          [--threshold THRESHOLD] [--batch-size BATCH_SIZE]
                          [--tile-size TILE_SIZE] [--tile-step TILE_STEP]
                          [--input-channels INPUT_CHANNELS [INPUT_CHANNELS ...]]
                          [--subfolder SUBFOLDER] [--dst-fname DST_FNAME]
                          [--qgis-vis QGIS_VIS]
                          model raster

positional arguments:
  model                 path to the pretrained model
  raster                path to the input test raster

optional arguments:
  -h, --help            show this help message and exit
  -p, --polygonize      polygonizes the rasters
  --username USERNAME   name of current user (Default: $USERNAME)
  --threshold THRESHOLD
                        probability threshold to compute IoU
  --batch-size BATCH_SIZE
                        number of tiles in each testing batch
  --tile-size TILE_SIZE
                        pixel size of the tiles
  --tile-step TILE_STEP
                        pixel step between two consecutive tiles
  --input-channels INPUT_CHANNELS [INPUT_CHANNELS ...]
                        indices of the image channels to test on
  --subfolder SUBFOLDER
                        subfolder where to save under username_exports
  --dst-fname DST_FNAME
                        filename for the export files
  --qgis-vis QGIS_VIS   re-organize output channels for QGIS
```

#### Example
```console
$ python predict_targets.py ./saved_weights/pretrained.pt /path/to/rasters/NEW_Mosaic_1m.tif --input-channels 0 1 2
```
