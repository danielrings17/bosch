#!/usr/bin/env python3
"""
Sample script to receive traffic light labels and images
of the Bosch Small Traffic Lights Dataset.

Example usage:
    python bosch.py input_yaml
"""

import os
import sys
import yaml

from skimage.io import imread, imshow, show
from skimage.transform import resize
from skimage.draw import polygon
import numpy as np
#from bstld.tf_object_detection import constants

WIDTH = 1280
HEIGHT = 720

T_WIDTH = 200
T_HEIGHT = 200

x_scale = WIDTH/T_WIDTH
y_scale = HEIGHT/T_HEIGHT


def drawBoundingBox(img_arr, coords, norm=False, b=1):
    #c [xmin, ymin, xmax, ymax]

    IMG_H = img_arr.shape[0]
    IMG_W = img_arr.shape[1]
    IMG_C = img_arr.shape[2]

    coords = np.array(coords).astype(int)
    xmin, ymin, xmax, ymax = coords[0], coords[1], coords[2], coords[3]
    width = np.floor(xmax-xmin)
    height = np.floor(ymax-ymin)

    r_up = np.array([ymin-b, ymin-b, ymin, ymin])
    c_up = np.array([xmin, xmax, xmax, xmin])
    r_left = np.array([ymin-b, ymin-b, ymax+b, ymax+b])
    c_left = np.array([xmin-b, xmin, xmin, xmin-b])

    r1, c1 = polygon(r_up, c_up)
    r2, c2 = polygon(r_up + height + b, c_up)
    r3, c3 = polygon(r_left, c_left)
    r4, c4 = polygon(r_left, c_left + width + b)

    rows = np.concatenate([r1,r2,r3,r4])
    cols = np.concatenate([c1,c2,c3,c4])
    rows = np.maximum(np.minimum(rows, IMG_H-1), 0)
    cols = np.maximum(np.minimum(cols, IMG_W-1), 0)

    #arr_copy = np.copy(img_arr)
    if norm:
        img_arr[rows,cols,:] = np.array([0,1,0])
    else:
        img_arr[rows, cols, :] = np.array([0, 255, 0])

    return img_arr


class BoschDataset():

    def __init__(self, label_path, W=1280, H=720):
        self.label_path = label_path
        self.W = W
        self.H = H
        self.labels = self.get_all_labels(self.label_path)


    def get_all_labels(self, input_yaml, riib=False, clip=True):
        """ Gets all labels within label file

        Note that RGB images are 1280x720 and RIIB images are 1280x736.
        Args:
            input_yaml->str: Path to yaml file
            riib->bool: If True, change path to labeled pictures
            clip->bool: If True, clips boxes so they do not go out of image bounds
        Returns: Labels for traffic lights
        """
        assert os.path.isfile(input_yaml), "Input yaml {} does not exist".format(input_yaml)
        with open(input_yaml, 'rb') as iy_handle:
            labels = yaml.load(iy_handle)

        if not labels or not isinstance(labels[0], dict) or 'path' not in labels[0]:
            raise ValueError('Something seems wrong with this label-file: {}'.format(input_yaml))

        #only images with bounding boxes
        labels = [label for label in labels if len(label['boxes']) != 0]


        for i in range(len(labels)):
            labels[i]['path'] = os.path.abspath(os.path.join(os.path.dirname(input_yaml),
                                                             labels[i]['path']))

            # There is (at least) one annotation where xmin > xmax
            for j, box in enumerate(labels[i]['boxes']):
                if box['x_min'] > box['x_max']:
                    labels[i]['boxes'][j]['x_min'], labels[i]['boxes'][j]['x_max'] = (
                        labels[i]['boxes'][j]['x_max'], labels[i]['boxes'][j]['x_min'])
                if box['y_min'] > box['y_max']:
                    labels[i]['boxes'][j]['y_min'], labels[i]['boxes'][j]['y_max'] = (
                        labels[i]['boxes'][j]['y_max'], labels[i]['boxes'][j]['y_min'])

            # There is (at least) one annotation where xmax > 1279
            if clip:
                for j, box in enumerate(labels[i]['boxes']):
                    labels[i]['boxes'][j]['x_min'] = max(min(box['x_min'], self.W - 1), 0)
                    labels[i]['boxes'][j]['x_max'] = max(min(box['x_max'], self.W - 1), 0)
                    labels[i]['boxes'][j]['y_min'] = max(min(box['y_min'], self.H - 1), 0)
                    labels[i]['boxes'][j]['y_max'] = max(min(box['y_max'], self.H - 1), 0)


        return labels

    def size(self):
        return len(self.labels)

    def getImage(self, i):
        return resize(imread(self.labels[i]['path']), (T_HEIGHT, T_WIDTH))

    def getBoxes(self, i):
        boxes_dicts = self.labels[i]['boxes']
        boxes = [[box['x_min']/x_scale, box['y_min']/y_scale, box['x_max']/x_scale, box['y_max']/y_scale] for box in boxes_dicts]
        boxes = np.array(boxes)
        #print(boxes)
        return boxes

    def showImgBoxes(self, i):
        img = self.getImage(i).copy()
        for box in self.getBoxes(i):
            img = drawBoundingBox(img, box)

        return img


BOSCH_DATA = BoschDataset('/home/colt/datasets/bosch_data/train_rgb/train.yaml')



if __name__ == '__main__':
    """
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(-1)
    get_all_labels(sys.argv[1])
    """
    bdata = BoschDataset('/home/colt/datasets/bosch_data/train_rgb/train.yaml')


    imshow(bdata.showImgBoxes(25))
    show()
