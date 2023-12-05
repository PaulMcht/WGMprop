import tensorflow as tf

class CifarConvBlock(tf.keras.layers.Layer):

    def __init__(self, nb_filter, name=None):
        super(CifarConvBlock, self).__init__(name=name)

        self.conv1 = tf.keras.layers.Conv2D(nb_filter, 3, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.act1 = tf.keras.layers.Activation("relu")

        self.conv2 = tf.keras.layers.Conv2D(nb_filter, 3, padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.act2 = tf.keras.layers.Activation("relu")

        self.pool = tf.keras.layers.AveragePooling2D((2, 2))


    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.pool(x)

        return x
    
    def linear_network(self, inputs_tilde, inputs_bar):
        x = self.conv1(inputs_tilde)
        x = self.bn1(x)
        mean_x_act1 = tf.reduce_mean(x, axis=0)
        rectified_mean_act1 = tf.where(tf.greater(mean_x_act1, 0), 1.0, 0.0)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        mean_x_act2 = tf.reduce_mean(x, axis=0)
        rectified_mean_act2 = tf.where(tf.greater(mean_x_act2, 0), 1.0, 0.0)
        x = self.act2(x)
        y_tilde = self.pool(x)

        x = self.conv1(inputs_bar)
        x = self.bn1(x)
        x = tf.multiply(x, rectified_mean_act1)
        x = self.conv2(x)
        x = self.bn2(x)
        x = tf.multiply(x, rectified_mean_act2)
        #Careful only for average pooling
        y_bar = self.pool(x)

        return y_tilde, y_bar

class Cifar10Model(tf.keras.Model):
    def __init__(self, input_shape, nb_class):
        super(Cifar10Model, self).__init__()
     
        self.conv_block1 = CifarConvBlock(32, "conv_block1")
        self.conv_block2 = CifarConvBlock(64, "conv_block2")
        self.conv_block3 = CifarConvBlock(128, "conv_block3")

        self.flat = tf.keras.layers.Flatten()
        self.dropout1 = tf.keras.layers.Dropout(0.2)
        
        # Hidden layer
        self.dense1 = tf.keras.layers.Dense(1024, activation=None)
        self.act4 = tf.keras.layers.ReLU()
        self.dropout2 = tf.keras.layers.Dropout(0.2)
        
        # last hidden layer i.e.. output layer
        self.dense2 = tf.keras.layers.Dense(nb_class, activation=None, use_bias=False)

    def call(self, inputs):
        
        x = self.conv_block1(inputs)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        
        x = self.flat(x)
        # x = self.dropout1(x)
        x = self.dense1(x)
        x = self.act4(x)
        # x = self.dropout2(x)

        return self.dense2(x)
    
    def linear_network(self, inputs):

        x_tilde, x_bar = self.conv_block1.linear_network(inputs, inputs)
        x_tilde, x_bar = self.conv_block2.linear_network(x_tilde, x_bar)
        x_tilde, x_bar = self.conv_block3.linear_network(x_tilde, x_bar) 

        x_tilde = self.flat(x_tilde)
        x_tilde = self.dense1(x_tilde)
        
        mean_x_act4 = tf.reduce_mean(x_tilde, axis=0)
        rectified_mean_act4 = tf.where(tf.greater(mean_x_act4, 0), 1.0, 0.0)
        x_tilde = self.act4(x_tilde)
        y_tilde = self.dense2(x_tilde)

        x_bar = self.flat(x_bar)
        x_bar = self.dense1(x_bar)
        x_bar = tf.multiply(x_bar, rectified_mean_act4)
        y_bar = self.dense2(x_bar)

        return y_bar, y_tilde

class Cifar100Model(tf.keras.Model):
     def __init__(self, input_shape, nb_class):
        super(Cifar100Model, self).__init__()
     
        self.conv_block1 = CifarConvBlock(32, "conv_block11")
        self.conv_block2 = CifarConvBlock(64, "conv_block21")
        self.conv_block3 = CifarConvBlock(128, "conv_block31")
        self.conv_block4 = CifarConvBlock(256, "conv_block41")
        self.conv_block5 = CifarConvBlock(512, "conv_block51")

        self.flat = tf.keras.layers.Flatten()
        self.dropout0 = tf.keras.layers.Dropout(0.2)
        
        # Hidden layer
        self.dense1 = tf.keras.layers.Dense(2048, activation=None)
        self.act1 = tf.keras.layers.ReLU()
        self.dropout1 = tf.keras.layers.Dropout(0.2)

        self.dense2 = tf.keras.layers.Dense(2048, activation=None)
        self.act2 = tf.keras.layers.ReLU()
        self.dropout2 = tf.keras.layers.Dropout(0.2)
        
        # last hidden layer i.e.. output layer
        self.dense3 = tf.keras.layers.Dense(nb_class, activation=None, use_bias=False)
        self.act3 = tf.keras.layers.Softmax()

     def call(self, inputs):
        
        x = self.conv_block1(inputs)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        
        x = self.flat(x)
        x = self.dropout0(x)
        x = self.dense1(x)
        x = self.act1(x)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.act2(x)
        x = self.dropout2(x)
        x = self.dense3(x)

        return self.act3(x)

     def linear_network(self, inputs):
        x_tilde, x_bar = self.conv_block1.linear_network(inputs, inputs)
        x_tilde, x_bar = self.conv_block2.linear_network(x_tilde, x_bar)
        x_tilde, x_bar = self.conv_block3.linear_network(x_tilde, x_bar) 
        x_tilde, x_bar = self.conv_block4.linear_network(x_tilde, x_bar) 
        x_tilde, x_bar = self.conv_block5.linear_network(x_tilde, x_bar) 

        x_tilde = self.flat(x_tilde)
        x_tilde = self.dense1(x_tilde)
        mean_x_act1 = tf.reduce_mean(x_tilde, axis=0)
        rectified_mean_act1 = tf.where(tf.greater(mean_x_act1, 0), 1.0, 0.0)
        x_tilde = self.act1(x_tilde)
        x_tilde = self.dense2(x_tilde)
        mean_x_act2 = tf.reduce_mean(x_tilde, axis=0)
        rectified_mean_act2 = tf.where(tf.greater(mean_x_act2, 0), 1.0, 0.0)
        x_tilde = self.act2(x_tilde)
        x_tilde = self.dense3(x_tilde)
        mean_x_act3 = tf.reduce_mean(x_tilde, axis=0)
        y_tilde = self.act3(x_tilde)

        x_bar = self.flat(x_bar)
        x_bar = self.dense1(x_bar)
        x_bar = tf.multiply(x_bar, rectified_mean_act1)
        x_bar = self.dense2(x_bar)
        x_bar = tf.multiply(x_bar, rectified_mean_act2)
        x_bar = self.dense3(x_bar)
        
        softmax_mean = self.act3(mean_x_act3)[:,tf.newaxis]
        softmax_prime_mean = softmax_mean * (tf.eye(tf.shape(mean_x_act3)[0]) - tf.transpose(softmax_mean))
        
        y_bar = softmax_mean[:,0] + tf.subtract(x_bar, mean_x_act3) @ softmax_prime_mean

        return y_bar, y_tilde
