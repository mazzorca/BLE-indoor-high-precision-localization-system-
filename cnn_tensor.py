import tensorflow as tf

import pandas as pd

if __name__ == '__main__':
    data_dir = "datasets/cnn_dataset/20x20-10/BLE2605r/"

    df = pd.read_csv(f"{data_dir}RSSI_images.csv")
    image_paths = df['RSSI'].values
    labels = df['Label'].values

    train_dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(50, (3, 3), activation='relu', input_shape=(1, 24, 24), padding="same"))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding="same"))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(4608))
    model.add(tf.keras.layers.Dense(18))

    model.compile(
        optimizer=tf.keras.optimizers.SGD(
            learning_rate=0.01, momentum=0.9, nesterov=False, name='SGD'),
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['loss'])

    model.fit(
        train_dataset,
        validation_data=None,
        epochs=20
    )


    def read_image(image_path, label):
        image = tf.io.read_file(f"{data_dir}{image_path}")
        image = tf.image.decode_image(image, channels=1, dtype=tf.float32)
        normalization_layer = tf.keras.layers.Rescaling(1. / 255)
        image = normalization_layer(image)

        return image, label


    ds_train = train_dataset.map(read_image).batch(32)

    model.fit(ds_train, epochs=20)
