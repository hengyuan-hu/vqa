#!/usr/bin/env python

'''
Reads in a tsv file with pre-trained bottom up attention features and stores it in HDF5 format.
Also store {image_id: feature_idx} dictinoary as a pickle file.

Hierarchy of HDF5 file:

{
  'image_features': num_images x num_boxes x 2048 array of features
  'image_bb': num_images x num_boxes x 4 array of bounding boxes
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
import pickle

csv.field_size_limit(sys.maxsize)

FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features']
infile = 'data/trainval_36/trainval_resnet101_faster_rcnn_genome_36.tsv'
data_outfile = 'data/trainval_bottom_up.hdf5'
indices_outfile = 'data/trainval_bottom_up_indicies.pkl'

feature_length = 2048
num_fixed_boxes = 36

if __name__ == '__main__':
    # Verify we can read a tsv
    with open(infile, "r+b") as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = FIELDNAMES)
        total_count = sum(1 for item in reader)

    h = h5py.File(data_outfile, "w")
    f = open(indices_outfile, "w")
    indices = {}

    counter = 0
    img_features = h.create_dataset('image_features', (total_count, num_fixed_boxes, feature_length), 'f')
    img_bb = h.create_dataset('image_bb', (total_count, num_fixed_boxes, 4), 'f')

    print("reading tsv...")
    with open(infile, "r+b") as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = FIELDNAMES)
        for item in reader:
            item['num_boxes'] = int(item['num_boxes'])

            img_bb[counter, :, :] = np.frombuffer(base64.decodestring(item['boxes']),
                  dtype=np.float32).reshape((item['num_boxes'],-1))
            img_features[counter, :, :] = np.frombuffer(base64.decodestring(item['features']),
                  dtype=np.float32).reshape((item['num_boxes'],-1))
            indices[item['image_id']] = counter
            counter += 1

    pickle.dump(indices, f)
    f.close()
    h.close()
    print("done!")
