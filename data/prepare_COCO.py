import json
import os
import shutil
from tqdm import tqdm

import sys
from pycocotools.coco import COCO

DATA_DIR = '/mnt/coco'
DATA_TYPE = 'train2017'
RESULT_PATH = '/mnt/coco/train_annotations/'

# DATA_TYPE = 'val2017'
# RESULT_PATH = '/mnt/coco/val_annotations/'

coco = COCO('{}/annotations/instances_{}.json'.format(DATA_DIR, DATA_TYPE))

catIds = coco.getCatIds()
imgIds = coco.getImgIds()

print('number of images:', len(imgIds))

def get_annotation(i):
    metadata = coco.loadImgs(i)[0]
    annIds = coco.getAnnIds(imgIds=metadata['id'], catIds=catIds, iscrowd=False)
    height, width = metadata['height'], metadata['width']
    annotation = {
      "filename": metadata['file_name'],
      "size": {"depth": 3, "width": width, "height": height}
    }
    objects = []
    for a in coco.loadAnns(annIds):
        label = coco_id_to_name[a['category_id']]
        xmin, ymin, w, h = a['bbox']
        xmax, ymax = xmin + w, ymin + h
        
        ymin = min(ymin, ymax)
        xmin = min(xmin, xmax)
        ymax = max(ymin, ymax)
        xmax = max(xmin, xmax)
        
        ymin = min(max(0, ymin), height)
        xmin = min(max(0, xmin), width)
        ymax = max(min(height, ymax), 0)
        xmax = max(min(width, xmax), 0)
        
        if (ymax - ymin) < 1 or (xmax - xmin) < 1:
            continue

        objects.append({"bndbox": {"ymin": ymin, "ymax": ymax, "xmax": xmax, "xmin": xmin}, "name": label})

    annotation["object"] = objects
    return annotation

shutil.rmtree(RESULT_PATH, ignore_errors=True)
os.mkdir(RESULT_PATH)

for i in tqdm(imgIds):
    d = get_annotation(i)
    filename = d['filename']
    assert filename.endswith('.jpg')
    name = filename[:-4]
    json.dump(d, open(os.path.join(RESULT_PATH, name + '.json'), 'w'))

