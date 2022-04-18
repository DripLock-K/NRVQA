# from matplotlib import image
import tensorflow as tf
from utils import *
import cv2
import numpy as np
import pandas as pd

def image_preprocess(image: tf.Tensor) -> tf.Tensor:
    image = tf.cast(image, tf.float32)
    image = tf.image.rgb_to_grayscale(image)
    image_low = gaussian_filter(image, 16, 7/6)
    image_low = rescale(image_low, 1/4, method=tf.image.ResizeMethod.BICUBIC)
    image_low = tf.image.resize(image_low, size=image_shape(
        image), method=tf.image.ResizeMethod.BICUBIC)

    return image-tf.cast(image_low, image.dtype)


def error_map(reference: tf.Tensor, distorted: tf.Tensor, p: float = 0.2):
    return tf.pow(tf.abs(reference-distorted), p)


def reliability_map(distorted: tf.Tensor, alpha: float) -> tf.Tensor:
    return 2/(1+tf.exp(-alpha*tf.abs(distorted)))-1


def average_reliability_map(distorted: tf.Tensor, alpha: float) -> tf.Tensor:
    r = reliability_map(distorted, alpha)
    return r/tf.reduce_mean(r)


def loss(model, x, y_true, r):
    y_pred = model(x)
    return tf.reduce_mean(tf.square((y_true-y_pred)*r))


def gradient(model, x, y_true, r):
    with tf.GradientTape() as tape:
        loss_value = loss(model, x, y_true, r)
        return loss_value, tape.gradient(loss_value, model.trainable_variables)


optimizer = tf.optimizers.Nadam(learning_rate=10**-4)

# ds = builder.as_dataset(shuffle_files=True)['train']
# ds = ds.shuffle(1024).batch(1)
data_dir = './imquality/datasets/kadid10k'
df = pd.read_csv(f'{data_dir}/dmos.csv')
df_dict = df.to_dict('records')
data = []
j = 0
for i in df_dict:
    frame = cv2.imread(f'{data_dir}/images/{i["dist_img"]}')
    # frame = np.expand_dims(frame, axis=0)
    im = tf.constant(frame)
    # im = image_preprocess(im)
    frame2 = cv2.imread(f'{data_dir}/images/{i["ref_img"]}')
    # frame2 = np.expand_dims(frame2, axis=0)
    im2 = tf.constant(frame2)
    # im2 = image_preprocess(im2)
    data.append({
        'dist_img': im,
        'ref_img': im2,
        'dmos': i['dmos'],
    })
    j+=1
    if j==100:
        break
    
def calculate_error_map(features):
    I_d = image_preprocess(features['dist_img'])
    I_r = image_preprocess(features['ref_img'])
    r = rescale(average_reliability_map(I_d, 0.2), 1/4)
    e_gt = rescale(error_map(I_r, I_d, 0.2), 1/4)

    return (I_d, e_gt, r)

train = map(calculate_error_map, data)
input = tf.keras.Input(shape=(None, None, 1),
                       batch_size=1, name='original_image')
f = tf.keras.layers.Conv2D(48, (3, 3), name='Conv1',
                           activation='relu', padding='same')(input)
f = tf.keras.layers.Conv2D(48, (3, 3), name='Conv2',
                           activation='relu', padding='same', strides=(2, 2))(f)
f = tf.keras.layers.Conv2D(64, (3, 3), name='Conv3',
                           activation='relu', padding='same')(f)
f = tf.keras.layers.Conv2D(64, (3, 3), name='Conv4',
                           activation='relu', padding='same', strides=(2, 2))(f)
f = tf.keras.layers.Conv2D(64, (3, 3), name='Conv5',
                           activation='relu', padding='same')(f)
f = tf.keras.layers.Conv2D(64, (3, 3), name='Conv6',
                           activation='relu', padding='same')(f)
f = tf.keras.layers.Conv2D(128, (3, 3), name='Conv7',
                           activation='relu', padding='same')(f)
f = tf.keras.layers.Conv2D(128, (3, 3), name='Conv8',
                           activation='relu', padding='same')(f)
g = tf.keras.layers.Conv2D(1, (1, 1), name='Conv9',
                           padding='same', activation='linear')(f)

objective_error_model = tf.keras.Model(input, g, name='objective_error_model')
objective_error_model.summary()


for epoch in range(10):
    print(epoch)
    epoch_accuracy = tf.keras.metrics.MeanSquaredError()
    step = 0
    for I_d, e_gt, r in train:
        loss_value, gradients = gradient(objective_error_model, I_d, e_gt, r)
        _ = optimizer.apply_gradients(
            zip(gradients, objective_error_model.trainable_weights))
        _ = epoch_accuracy(e_gt, objective_error_model(I_d))

        if(step % 100 == 0):
            print("epoch:%s step:%s loss= %s" %
                  (epoch, step, epoch_accuracy.result().numpy()))
        step += 1


v = tf.keras.layers.GlobalAveragePooling2D(data_format='channels_last')(f)
h = tf.keras.layers.Dense(128, activation='relu')(v)
h = tf.keras.layers.Dense(1)(h)
subjective_error_model = tf.keras.Model(
    input, h, name='subjective_error_model')
subjective_error_model.compile(optimizer=optimizer, loss=tf.losses.MeanSquaredError(
), metrics=[tf.metrics.MeanSquaredError()])
subjective_error_model.summary()


# def calculate_subjective_score(features):
#     # I_d = image_preprocess(features['dist_img'])
#     mos = features['dmos']
#     # return (I_d, mos)
#     return mos
def calculate_subjective_score(features, epochs):
    for _ in range(epochs):
        for i in features:
            yield image_preprocess(i['dist_img']), np.array([i['dmos']])

EPOCH = 10
history = subjective_error_model.fit(
    calculate_subjective_score(data, epochs=EPOCH), 
    epochs=EPOCH,
    steps_per_epoch=100,
    batch_size=1)
subjective_error_model.save_weights("models/")
tf.saved_model.save(subjective_error_model, "models2/")

# sample = next(iter(ds))
# sample = data[0]
# I_d = image_preprocess(sample['dist_img'])
# target = sample['dmos'][0]
# prediction = subjective_error_model.predict(I_d)[0][0]
# print(prediction)
