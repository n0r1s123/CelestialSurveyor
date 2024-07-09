from keras.layers import ConvLSTM2D, Flatten, Input, Conv3D, BatchNormalization, Activation, MaxPooling3D, GlobalAveragePooling3D, Dense, Concatenate
from keras.models import Model
from cryptography.fernet import Fernet
import os
from resnet18 import convolutional_block, identity_block



# os.environ["CUDA_VISIBLE_DEVICES"] = ""
# os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"


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
    input_s = Input(shape=input_shape)
    slow = conv3d_bn(input_s, 32, (1, 3, 3), strides=(1, 2, 2), padding='same')
    # slow = MaxPooling3D(pool_size=(1, 3, 3), strides=(1, 2, 2), padding='same')(slow)
    slow = conv3d_bn(slow, 32, (1, 3, 3))

    # Fast pathway
    input_f = Input(shape=input_shape)
    fast = conv3d_bn(input_f, 8, (3, 3, 3), strides=(1, 2, 2), padding='same')
    # fast = MaxPooling3D(pool_size=(1, 3, 3), strides=(1, 2, 2), padding='same')(fast)
    fast = conv3d_bn(fast, 8, (3, 1, 1))
    new_fast = conv3d_bn(fast, 16, (1, 1, 1), strides=(4, 1, 1), padding='same')  # Adjust number of frames for fusion
    slow = Concatenate()([slow, new_fast])
    fast = convolutional_block(fast, filters=[16, 16], s=1)
    fast = identity_block(fast, [16, 16])
    slow = convolutional_block(slow, filters=[64, 64], s=1)
    slow = identity_block(slow, [64, 64])
    new_fast = conv3d_bn(fast, 32, (5, 3, 3), strides=(4, 1, 1), padding='same')  # Adjust number of frames for fusion
    slow = Concatenate()([slow, new_fast])

    slow = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 1, 1), padding='same')(slow)
    fast = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 1, 1), padding='same')(fast)
    fast = convolutional_block(fast, filters=[32, 32], s=2)
    fast = identity_block(fast, [32, 32])
    slow = convolutional_block(slow, filters=[128, 128], s=2)
    slow = identity_block(slow, [128, 128])
    new_fast = conv3d_bn(fast, 64, (1, 1, 1), strides=(4, 1, 1), padding='same')  # Adjust number of frames for fusion
    slow = Concatenate()([slow, new_fast])

    slow = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 1, 1), padding='same')(slow)
    fast = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 1, 1), padding='same')(fast)
    fast = convolutional_block(fast, filters=[64, 64], s=2)
    fast = identity_block(fast, [64, 64])
    slow = convolutional_block(slow, filters=[256, 256], s=2)
    slow = identity_block(slow, [256, 256])
    new_fast = conv3d_bn(fast, 128, (1, 1, 1), strides=(4, 1, 1), padding='same')  # Adjust number of frames for fusion
    slow = Concatenate()([slow, new_fast])
    #
    # slow = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 1, 1), padding='same')(slow)
    # fast = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 1, 1), padding='same')(fast)
    # fast = convolutional_block(fast, filters=[128, 128], s=2)
    # fast = identity_block(fast, [128, 128])
    # slow = convolutional_block(slow, filters=[256, 256], s=2)
    # slow = identity_block(slow, [256, 256])
    # new_fast = conv3d_bn(fast, 128, (1, 1, 1), strides=(4, 1, 1), padding='same')  # Adjust number of frames for fusion
    # slow = Concatenate()([slow, new_fast])



    # # Multiple Fusions
    # for _ in range(num_fusions):
    #     new_fast = conv3d_bn(fast, 64, (1, 1, 1), strides=(4, 1, 1), padding='same') # Adjust number of frames for fusion
    #     slow = Concatenate()([slow, new_fast])
    #     fast = conv3d_bn(fast, 64, (5, 3, 3))
    #     slow = conv3d_bn(slow, 128, (1, 5, 5))


    # Continue Slow Path
    slow = GlobalAveragePooling3D()(slow)
    fast = GlobalAveragePooling3D()(fast)
    # slow = Flatten(slow)
    # fast = Flatten(fast)
    # slow = ConvLSTM2D(32, (3, 3), return_sequences=False)(slow)
    # fast = ConvLSTM2D(32, (3, 3), return_sequences=False)(fast)
    output = Concatenate()([slow, fast])
    # output = Dense(128, activation='relu')(output)
    output = Dense(128, activation='relu')(output)
    output = Dense(32, activation='relu')(output)
    output = Dense(1, activation='sigmoid')(output)

    model = Model(inputs=[input_f, input_s], outputs=output)
    return model


# def build_model():
#     slowfast_model = SlowFast(num_fusions=3)  # Example: Three fusion steps
#     return slowfast_model
def build_model():
    from slow_fast_exp import SlowFast_body, bottleneck
    model = SlowFast_body([3, 4, 6, 3], bottleneck)
    return model
    # slowfast_model = SlowFast(num_fusions=3)  # Example: Three fusion steps
    # return slowfast_model

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
