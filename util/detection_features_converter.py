#!/usr/bin/env python

'''
Reads in a tsv file with pre-trained bottom up attention features and stores it in HDF5 format.

Hierarchy of HDF5 file for n total features/bounding boxes:

{
  'image_features': n x 2048 array of features
  'image_bb': n x 4 array of bounding boxes
  'image_id': {
                'image_h': height of image
                'image_w': width of image
                'num_boxes': number of bounding boxes/features for this image
                'index': starting index into 'image_features' and 'image_bb'
              }
}
'''

import base64
import numpy as np
import csv
import sys
import zlib
import time
import mmap
import h5py

csv.field_size_limit(sys.maxsize)

FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features']
infile = 'data/test2014_resnet101_faster_rcnn_genome_36.tsv'
outfile = 'data/test_bottom_up.hdf5'

feature_length = 2048
num_fixed_boxes = 36

if __name__ == '__main__':
    # Verify we can read a tsv
    in_data = {}
    total_boxes = 0

    print("reading tsv...")
    with open(infile, "r+b") as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = FIELDNAMES)
        for item in reader:
            item['image_id'] = int(item['image_id'])
            item['image_h'] = int(item['image_h'])
            item['image_w'] = int(item['image_w'])
            item['num_boxes'] = int(item['num_boxes'])
            for field in ['boxes', 'features']:
                item[field] = np.frombuffer(base64.decodestring(item[field]),
                      dtype=np.float32).reshape((item['num_boxes'],-1))
            in_data[item['image_id']] = item
            total_boxes += item['num_boxes']

    print("saving h5py file...")

    f = h5py.File(outfile, "w")
    img_features = f.create_dataset('image_features', (len(in_data), num_fixed_boxes, feature_length), 'f')
    img_bb = f.create_dataset('image_bb', (len(in_data), num_fixed_boxes, 4), 'f')

    counter = 0
    for image_id in in_data:
      item = in_data[image_id]
      grp = f.create_group(str(image_id))
      num_boxes = item['num_boxes']

      grp.attrs['image_h'] = item['image_h']
      grp.attrs['image_w'] = item['image_w']
      grp.attrs['num_boxes'] = num_boxes
      grp.attrs['index'] = counter

      img_features[counter, :, :] = item['features']
      img_bb[counter, :, :] = item['boxes']

      counter += 1

    f.close()
    print("done!")

