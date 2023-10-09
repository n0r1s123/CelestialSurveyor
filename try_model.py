from object_detector import Dataset
import tensorflow as tf


if __name__ == '__main__':
    dataset_class = Dataset('C:\\Users\\bsolomin\\Astro\\Iris_2023\\Pix\\cropped',
                            'C:\\git\\object_recognition\\star_samples')

    model = tf.keras.models.load_model(
        'model3.h5'
    )

    train_generator = dataset_class.generate_batch(20)
    series, expected_result = next(train_generator)
    for expected, actual in zip(expected_result, model.predict(series)):
        print(expected, actual)
