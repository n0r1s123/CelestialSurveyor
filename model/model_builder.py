from cryptography.fernet import Fernet
from slow_fast_exp import SlowFast_body, bottleneck
from tensorflow.keras.models import Model


def build_model() -> Model:
    """
    Builds and returns a SlowFast model with specific configurations.
    You can update this function with your own model.

    Returns:
        Model: The built SlowFast model.
    """
    model = SlowFast_body([3, 4, 6, 3], bottleneck)
    return model


def encrypt_model(model_name: str, key: bytes = b'J17tdv3zz2nemLNwd17DV33-sQbo52vFzl2EOYgtScw=') -> None:
    """
    Encrypts the model weights and saves them to a binary file.
    Preliminary version of model decryption. It was done when I was thinking to make this project open source or not.

    Args:
        model_name: Name of the model file to be encrypted.
        key: Encryption key to be used. Default is a predefined key.

    Returns:
        None
    """
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
    build_model().summary()
