import tensorflow as tf
from tensorflow.keras.utils import Sequence
from skimage.transform import resize
from skimage.io import imread
import numpy as np
import math

from bosch import BoschDataset

dataset = BoschDataset()
labels = dataset.labels

def image_process(X, image_paths):
    batch_size = X.shape[0]
    target_dim = X.shape[1:]


    for i,path in enumerate(image_paths):
        x = imread(path)
        x = resize(x, target_dim)
        #x = scale(x, x_factor, y_factor)
        #x = translate(x, delta)
        #x = hsv_transform(x, factor)
        X[i,...] = x

        #TODO transformacije


def label_process(y, box_list, img_dim, n_boxes, classes):
    batch_size = y.shape[0]
    target_dim = y.shape[1:]

    s = y.shape[1]
    d = img_dim // s #velicina celije

    for boxes in box_list:

        z = np.zeros(n_boxes * 5 + classes)

        '''
        Trebalo bi rasporediti na 2 slučaja kad boxeva ima vise nego objekata u ćeliji i obratno
        '''

        for b in boxes:
            xc = (b['x_max'] - b['x_min']) / 2.0
            yc = (b['y_max'] - b['y_min']) / 2.0
            w = b['x_max'] - b['x_min']
            h = b['y_max'] - b['x_min']

            cell_x = xc // s
            cell_y = yc // s

            #TODO

class DataParser(Sequence):

    def __init__(self, list_IDs, batch_size=32, input_dim=(224, 224, 3), output_dim=(7,7,10), boxes_per_cell=2, classes=0, shuffle=True):
        self.list_IDs = list_IDs
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.shuffle = shuffle
        self.n_boxes = boxes_per_cell
        self.classes = classes
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.list_IDs) / self.batch_size)

    def __getitem__(self, idx):
        indexes = self.indexes[idx*self.batch_size:(idx+1)*self.batch_size]
        X,y = self.__data_generation(indexes)

        return X,y


    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        """
        Treba generirati ciljni vektor y ovisno o ulazu X:
        y će biti veličine batch_size x 7 x 7 x (B*5 + C)
        Ako se centar objekta nalazi u području ćelije, onda je ta ćelija zadužena za bounding box detekciju tog objekta
        xc i yc se računaju ovisno o kordinatama te ćelije, tj. xc,yc u [0,1]
        w, h se računaju ovisno o širini, tj. visini slike
        """

        #assert len(y.shape) == 4, "Target array does not have 4 dimensions"
        #assert target_dim[0] == target_dim[1], "Target array dimension mismatch."

        X = np.empty((self.batch_size, *self.input_dim))
        y = np.zeros((self.batch_size, *self.output_dim))

        image_paths = [labels[k]['path'] for k in indexes]
        image_process(X, image_paths)

        box_list = [labels[k]['boxes'] for k in indexes]
        label_process(y, box_list, self.input_dim[0], self.n_boxes, self.classes)

        #TODO


