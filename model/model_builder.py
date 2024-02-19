from keras.layers import Input, Conv3D, BatchNormalization, Activation, MaxPooling3D, GlobalAveragePooling3D, Dense, Concatenate
from keras.models import Model
from cryptography.fernet import Fernet
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

def conv3d_bn(x, filters, kernel_size, strides=(1, 1, 1), padding='same'):
    x = Conv3D(filters=filters,
               kernel_size=kernel_size,
               strides=strides,
               padding=padding)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def SlowFast(input_shape=(None, 64, 64, 1), num_fusions=3):
    # Slow pathway
    input = Input(shape=input_shape)
    slow = conv3d_bn(input, 64, (5, 3, 3), strides=(1, 2, 2), padding='same')
    slow = MaxPooling3D(pool_size=(1, 3, 3), strides=(1, 2, 2), padding='same')(slow)
    slow = conv3d_bn(slow, 64, (3, 1, 1), padding='same')

    # Fast pathway
    # input_f = Input(shape=input_shape)
    fast = conv3d_bn(input, 8, (1, 7, 7), strides=(1, 2, 2), padding='same')
    fast = MaxPooling3D(pool_size=(1, 3, 3), strides=(1, 2, 2), padding='same')(fast)
    fast = conv3d_bn(fast, 32, (1, 3, 3), padding='same')

    # Multiple Fusions
    for _ in range(num_fusions):

        slow = Concatenate()([slow, fast])
        fast = conv3d_bn(fast, 64, (1, 3, 3), padding='same')  # Adjust channels for fusion
        slow = conv3d_bn(slow, 128, (3, 1, 1), padding='same')

    # Continue Slow Path
    slow = GlobalAveragePooling3D()(slow)
    fast = GlobalAveragePooling3D()(fast)
    output = Concatenate()([slow, fast])
    output = Dense(128, activation='relu')(output)
    output = Dense(64, activation='relu')(output)
    output = Dense(32, activation='relu')(output)
    output = Dense(1, activation='sigmoid')(output)

    model = Model(inputs=input, outputs=output)
    return model


def build_model():
    slowfast_model = SlowFast(num_fusions=3)  # Example: Three fusion steps
    return slowfast_model

# Encrypt the model weights
def encrypt_model(model_name, key=b'J17tdv3zz2nemLNwd17DV33-sQbo52vFzl2EOYgtScw='):
    # Read the entire model file
    with open(f"{model_name}.h5", "rb") as file:
        model_bytes = file.read()

    # Use the provided key to create a cipher
    cipher = Fernet(key)

    # Encrypt the entire model
    encrypted_model = cipher.encrypt(model_bytes)

    # Save the encrypted weights to a file
    with open(f"{model_name}.bin", "wb") as file:
        file.write(encrypted_model)


if __name__ == '__main__':
    model = build_model()
    model.summary()
