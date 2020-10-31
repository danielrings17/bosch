from layers import cnl_cascade, conv_norm_lrelu, reorg_layer
import tensorflow as tf
import numpy as np

image_dim = 448
grid_dim = 7
B = 2
C = 20
anchor_boxes = 5
n_classes = 0
feature_tensor_size = grid_dim * grid_dim * (B*5 + C)
feature_tensor_shape = (grid_dim, grid_dim, B*5 + C)

img_resolutions = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]

class Detector():

    def __init__(self):
        pass

    def save_weights(self):
        self.model.save_weights(self.name +'.h5', save_format='h5')

    def load_weights(self, filename):
        self.model.load_weights(filename)

class Extraction(Detector):

    def __init__(self, input_shape=(448, 448, 3)):
        super().__init__()
        self.input_shape = input_shape
        self.name = "ExtractionNet"

        self.model = self.buildModel()

    def buildModel(self, training=True):
        graph_input = tf.keras.Input(shape=self.input_shape, dtype='float32')

        x = conv_norm_lrelu(graph_input, 64, 7, stride=2)
        x = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2, padding='valid')(x)

        x = conv_norm_lrelu(x, 192, 3)
        x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='valid')(x)

        x = cnl_cascade(x, [(128,1,1), (256,3,1), (256,1,1), (512,3,1)])
        x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='valid')(x)

        x = cnl_cascade(x, [(256,1,1), (512,3,1), (256,1,1), (512,3,1), (256,1,1), (512,3,1), (256,1,1), (512,3,1),
                                            (512,1,1), (1024,3,1)])
        x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='valid')(x)

        x = cnl_cascade(x, [(512,1,1), (1024,3,1), (512,1,1), (1024,3,1)])

        ######## detektorski dio

        x = cnl_cascade(x, [(1024,3,1), (1024,3,2), (1024,3,1), (1024,3,1)])

        x = tf.keras.layers.Flatten()(x)

        x = tf.keras.layers.Dense(4096)(x)

        #add dropout

        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

        x = tf.keras.layers.Dense(feature_tensor_size)(x) # 7 * 7 * 30 (B*5 + C)

        x = tf.keras.layers.Reshape(feature_tensor_shape)(x)

        return tf.keras.Model(inputs=graph_input, outputs=x, name='yolo-model')



class ModelDetector(Detector):

    def __init__(self, input_shape=(416, 416, 3)):
        super().__init__()
        self.input_shape = input_shape
        self.name = "ModelDetector"

        self.model = self.buildModel()

    def buildModel(self):
        tensor_input = tf.keras.Input(shape=self.input_shape, dtype='float32')

        #conv1
        x = conv_norm_lrelu(tensor_input, 32, 3, stride=1)
        x = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2, padding='valid')(x)

        #conv2
        x = conv_norm_lrelu(x, 64, 3)
        x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='valid')(x)

        #conv3-5
        x = cnl_cascade(x, [(128,3,1), (64,1,1), (128,3,1)])
        x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='valid')(x)

        #conv6-8
        x = cnl_cascade(x, [(256, 3, 1), (128, 1, 1), (256, 3, 1)])
        x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='valid')(x)

        #conv9-13
        x = cnl_cascade(x, [(512, 3, 1), (256, 1, 1), (512, 3, 1), (256,1,1), (512,3,1)])
        skip = x
        x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='valid')(x)

        #conv14-18
        x = cnl_cascade(x, [(1024, 3, 1), (512, 1, 1), (1024, 3, 1), (512,1,1), (1024,3,1)])
        #x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='valid')(x)

        #detekcija

        x = conv_norm_lrelu(x, 1024, 3)

        x_c = conv_norm_lrelu(x, 1024, 3)

        reorg = reorg_layer(skip, 2)

        concat = tf.keras.layers.concatenate([x_c, reorg], axis=-1)

        x = conv_norm_lrelu(concat, 1024, 3)

        x = tf.keras.layers.Conv2D(anchor_boxes*(5 + n_classes), 1, strides=1, padding='same')(x)



        return tf.keras.Model(inputs=tensor_input, outputs=x, name='yolo-model')


#yolo = Extraction()
yolo = ModelDetector()
model = yolo.buildModel()

model.summary()
