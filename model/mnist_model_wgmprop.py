import tensorflow as tf
from model.enums import *

class MNISTBlock(tf.keras.layers.Layer):

    def __init__(self, nb_filter, name=None):
        super(MNISTBlock, self).__init__(name=name)

        self.fc1 = tf.keras.layers.Dense(nb_filter, activation=None, use_bias=True)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.act1 = tf.keras.layers.Activation("relu")


    def call(self, inputs, training=False):
        x = self.fc1(inputs)
        x = self.bn1(x, training=training)
        x = self.act1(x)

        return x
    
    def linear_network(self, inputs_tilde, inputs_bar):
        x = self.fc1(inputs_tilde)
        x = self.bn1(x)
        mean_x_act1 = tf.reduce_mean(x, axis=0)
        rectified_mean_act1 = tf.where(tf.greater(mean_x_act1, 0), 1.0, 0.0)
        y_tilde = self.act1(x)

        x = self.fc1(inputs_bar)
        x = self.bn1(x)
        y_bar = tf.multiply(x, rectified_mean_act1)

        return y_tilde, y_bar


class MNISTModel(tf.keras.Model):
    def __init__(self, input_shape, nb_class):
        super(MNISTModel, self).__init__()
     
        self.fc_block1 = MNISTBlock(200, "fc_block1")
        self.fc_block2 = MNISTBlock(200, "fc_block2")
        self.fc_block3 = MNISTBlock(200, "fc_block3")
        self.fc_block4 = MNISTBlock(200, "fc_block4")
        self.output_layer = tf.keras.layers.Dense(nb_class, activation=None, use_bias=False)

    def call(self, inputs, training):
        
        x = self.fc_block1(inputs, training)
        x = self.fc_block2(x, training)
        x = self.fc_block3(x, training)
        x = self.fc_block4(x, training)

        return self.output_layer(x)
    
    def linear_network(self, inputs):

        x_tilde, x_bar = self.fc_block1.linear_network(inputs, inputs)
        x_tilde, x_bar = self.fc_block2.linear_network(x_tilde, x_bar)
        x_tilde, x_bar = self.fc_block3.linear_network(x_tilde, x_bar) 
        x_tilde, x_bar = self.fc_block4.linear_network(x_tilde, x_bar) 

        y_tilde = self.output_layer(x_tilde)
        y_bar = self.output_layer(x_bar)

        return y_bar, y_tilde



