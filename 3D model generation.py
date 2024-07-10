import glob
import os
import time
import numpy as np
import scipy.io as io
import scipy.ndimage as nd
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Conv3D, Conv3DTranspose, BatchNormalization, LeakyReLU, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Function to build the generator model
def build_generator():
    z_size = 200
    gen_filters = [512, 256, 128, 64, 1]
    gen_kernel_sizes = [4, 4, 4, 4, 4]
    gen_strides = [1, 2, 2, 2, 2]
    gen_activations = ['relu', 'relu', 'relu', 'relu', 'sigmoid']
    gen_convolutional_blocks = 5
    gen_input_shape = (1, 1, 1, z_size)

    gen_input_layer = Input(shape=gen_input_shape)
    x = gen_input_layer

    for i in range(gen_convolutional_blocks):
        x = Conv3DTranspose(filters=gen_filters[i], kernel_size=gen_kernel_sizes[i], 
                            strides=gen_strides[i], padding='same')(x)
        x = BatchNormalization()(x, training=True)
        x = Activation(gen_activations[i])(x)
    
    gen_model = Model(inputs=[gen_input_layer], outputs=[x])
    return gen_model

# Function to build the discriminator model
def build_discriminator():
    dis_input_shape = (64, 64, 64, 1)
    dis_filters = [64, 128, 256, 512, 1]
    dis_kernel_sizes = [4, 4, 4, 4, 4]
    dis_strides = [2, 2, 2, 2, 1]
    dis_paddings = ['same', 'same', 'same', 'same', 'valid']
    dis_alphas = [0.2, 0.2, 0.2, 0.2, 0.2]
    dis_activations = ['leaky_relu', 'leaky_relu', 'leaky_relu', 'leaky_relu', 'sigmoid']
    dis_convolutional_blocks = 5

    dis_input_layer = Input(shape=dis_input_shape)
    x = dis_input_layer

    for i in range(dis_convolutional_blocks):
        x = Conv3D(filters=dis_filters[i], kernel_size=dis_kernel_sizes[i], 
                   strides=dis_strides[i], padding=dis_paddings[i])(x)
        x = BatchNormalization()(x, training=True)
        
        if dis_activations[i] == 'leaky_relu':
            x = LeakyReLU(dis_alphas[i])(x)
        elif dis_activations[i] == 'sigmoid':
            x = Activation(activation='sigmoid')(x)

    dis_model = Model(inputs=[dis_input_layer], outputs=[x])
    return dis_model

# Function to write log data to TensorBoard
def write_log(callback, name, value, batch_no):
    summary = tf.Summary()
    summary_value = summary.value.add()
    summary_value.simple_value = value
    summary_value.tag = name
    callback.writer.add_summary(summary, batch_no)
    callback.writer.flush()

# Function to load 3D voxel images from a directory
def get3DImages(data_dir):
    all_files = np.random.choice(glob.glob(data_dir), size=10)
    all_volumes = np.asarray([getVoxelsFromMat(f) for f in all_files], dtype=np.bool)
    return all_volumes

# Function to load 3D voxel data from .mat file
def getVoxelsFromMat(path, cube_len=64):
    voxels = io.loadmat(path)['instance']
    voxels = np.pad(voxels, (1, 1), 'constant', constant_values=(0, 0))
    if cube_len != 32 and cube_len == 64:
        voxels = nd.zoom(voxels, (2, 2, 2), mode='constant', order=0)
    return voxels

if __name__ == "__main__":
    # Example usage
    data_dir = '/path/to/your/data/directory/*.mat'  # Update with your data directory
    volumes = get3DImages(data_dir)
    print(f"Loaded {len(volumes)} volumes.")
    
    # Build generator and discriminator models
    generator = build_generator()
    discriminator = build_discriminator()

    # Compile discriminator model
    discriminator.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy', metrics=['accuracy'])

    # Print model summaries
    generator.summary()
    discriminator.summary()

