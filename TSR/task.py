import random
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from albumentations import (Compose, ElasticTransform, RandomGamma, GridDistortion, RGBShift, Rotate,
                            RandomBrightness)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, recall_score, precision_score)
from sklearn.utils.multiclass import unique_labels

from settings import (n_classes, channels, img_size, to_crop, elastic_params, brightness_params, gamma_params,
                      r_shift_params, rotate_params, all_other, shrink_size, to_shrink, to_augment, to_normalize)


def crop(img, upper_bounds, lower_bounds):
    x_top, y_top = upper_bounds
    x_bot, y_bot = lower_bounds
    cropped = Image.fromarray(img).crop((x_top, y_top, x_bot, y_bot))
    return np.array(cropped)


def read_train_data():
    path = 'GTSRB/Training'
    images = []  # images
    labels = []  # corresponding labels
    for c in range(n_classes):
        prefix = path + '/' + '{:05d}'.format(c) + '/'  # subdirectory for class
        df = pd.read_csv(prefix + 'GT-' + '{:05d}'.format(c) + '.csv', sep=';')
        images.append(np.array([plt.imread(prefix + item) for item in df['Filename'].to_numpy()]))
        if to_crop:
            borders_np = df[['Roi.X1', 'Roi.Y1', 'Roi.X2', 'Roi.Y2']].to_numpy()
            borders = np.array([((x1, y1), (x2, y2)) for x1, y1, x2, y2 in borders_np])
            for i, border in enumerate(borders):
                images[-1][i] = crop(images[-1][i], *border)
        labels.append(df['ClassId'].values)
    return images, labels


def transform_image(image):
    """
    1) Make image squared
    2) Add padding to squared image
    :rtype: ndarray
    :param image: image in matrix form
    :return: transformed image
    """
    (n, m, _) = image.shape  # extract height and width from image matrix
    if n > m:
        padding = ((1, 1), (1, n - m + 1))
        shape = list(image.shape)
        shape[0] += 2
        shape[1] += n - m + 2
    elif m > n:
        padding = ((1, m - n + 1), (1, 1))
        shape = list(image.shape)
        shape[0] += m - n + 2
        shape[1] += 2
    else:
        padding = ((1, 1), (1, 1))
        shape = list(image.shape)
        shape[0] += 2
        shape[1] += 2

    img = np.zeros(shape, dtype=np.uint8)
    shape_len = len(shape)
    if shape_len > 1:
        for c in range(shape_len):
            tmp = np.pad(image[:, :, c], padding, mode='edge')
            img[:, :, c] = tmp
    else:
        img = np.pad(image, padding, mode='edge')
    return np.array(Image.fromarray(img).resize(img_size, Image.ANTIALIAS))


def normalize(img):
    return img.astype(float) / 255


def augmentation(img, n):
    """
    Make random augmentations with image n times.
    :rtype: ndarray
    :param img: image in matrix form
    :param n: how many augmentations need to apply
    :return: list of augmented versions of image
    """
    methods = [ElasticTransform(**elastic_params),
               RandomGamma(**gamma_params),
               GridDistortion(**all_other),
               RGBShift(**r_shift_params),
               Rotate(**rotate_params),
               RandomBrightness(**brightness_params)
               ]
    for i in range(len(methods)):
        methods[i] = Compose([methods[i], ], p=1)
    chosen = np.random.choice(methods, replace=False, size=n)
    augmented = np.empty((n,), dtype=np.object)
    for i, method in enumerate(chosen):
        transformed = transform_image(method(image=img)["image"])
        if to_normalize:
            transformed = normalize(transformed)
        augmented[i] = transformed
    return augmented


def shrink(train):
    # shrinkage of big classes
    if train.flatten().shape[0] > shrink_size:
        k = int(np.ceil(train.flatten().shape[0] / 750))
        tmp_train = []
        for j in range(train.shape[0]):
            shrunken = np.random.choice(train[j], replace=False, size=30 // k)
            tmp_train.append(shrunken)
        return np.array(tmp_train)
    return train


def augment_data(train):
    # augmentation starts here
    k = train[0].shape[0]  # number of images per track
    n = train.shape[0]  # number of tracks
    if not to_shrink:
        max_size = max([item.shape[0] for item in train])
    else:
        max_size = shrink_size
    diff = max_size - n * k
    n_aug_per_el = int(diff / (n * k))
    residual = int(diff - n_aug_per_el * n * k)

    tmp_train_aug = []
    for i in range(n):
        for j in range(k):
            transformed = transform_image(train[i][j])
            if to_normalize:
                transformed = normalize(transformed)
            tmp_train_aug.append(transformed)
    for i in range(n):
        if n_aug_per_el > 0:
            for j in range(k):
                augmented = augmentation(train[i][j], n_aug_per_el)
                tmp_train_aug.extend(list(augmented))

    if residual > 0:
        for item in range(residual):
            ind_track = random.randint(0, n) - 1
            ind_sample = random.randint(0, train[ind_track].shape[0]) - 1
            augmented = augmentation(train[ind_track][ind_sample], 1)
            tmp_train_aug.extend(list(augmented))
    return np.array(tmp_train_aug)


def split_data(x):
    test_x, test_y = [], []
    train_x, train_y = [], []
    # loop over classes
    for c in range(len(x)):
        n = x[c].shape[0]
        # loop over tracks and form tracks list
        tracks = []
        for j in range(n // 30):
            tracks.append(x[c][j * 30:(j + 1) * 30])
        tracks = np.array(tracks)
        np.random.shuffle(tracks)
        pareto = int(np.floor(tracks.shape[0] * 0.8))
        train_tracks, test_tracks = tracks[:pareto], tracks[pareto:]

        if to_shrink:
            train_tracks = shrink(train_tracks)

        # transform_data test
        tmp_test = []
        for track in test_tracks:
            for img in track:
                transformed = transform_image(img)
                if to_normalize:
                    transformed = normalize(transformed)
                tmp_test.append(transformed)
        test_tracks = np.array(tmp_test)
        test_x.append(test_tracks)
        curr_labels = np.full((test_tracks.shape[0],), c)
        test_y.append(curr_labels)
        if to_augment:
            train_tracks = augment_data(train_tracks)
        train_x.append(train_tracks)
        curr_labels = np.full((train_tracks.shape[0],), c)
        train_y.append(curr_labels)
    test_x = np.array(test_x)
    test_y = np.array(test_y)
    train_x = np.array(train_x)
    train_y = np.array(train_y)

    return test_x, test_y, train_x, train_y


def plot_bar(shapes, title):
    plt.xlabel('Class')
    plt.ylabel('Images')
    plt.bar(shapes.T[0], shapes.T[1])
    plt.savefig(f'{title}.png')
    plt.close()


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots(figsize=(20, 20))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


start = datetime.now()
images, labels = read_train_data()
diff = datetime.now() - start
print("time for reading: ", diff.total_seconds())
shapes_raw = [[i + 1, images[i].shape[0]] for i in range(n_classes)]
shapes_raw = np.array([np.array(item) for item in shapes_raw])
plot_bar(shapes_raw, 'raw_images_per_class')
start = datetime.now()
test_images, test_labels, train_images, train_labels = split_data(images)
diff = datetime.now() - start
print("time for splitting: ", diff.total_seconds())
shapes_test = [[i + 1, test_images[i].shape[0]] for i in range(n_classes)]
shapes_test = np.array([np.array(item) for item in shapes_test])
plot_bar(shapes_test, 'test_images_per_class')
shapes_train = np.array([[i + 1, shapes_raw[i][1] - shapes_test[i][1]] for i in range(n_classes)])
plot_bar(shapes_train, 'train_images_images_per_class')
shapes = [[i + 1, train_images[i].shape[0]] for i in range(n_classes)]
shapes = np.array([np.array(item) for item in shapes])
plot_bar(shapes_train, 'train_images_images_per_class_after_shrinkage_and_augmentation')
start = datetime.now()
model = RandomForestClassifier(max_depth=15, n_estimators=100, n_jobs=-1)
diff = datetime.now() - start
print("time for model initialization: ", diff.total_seconds())
n = np.array([item.shape[0] for item in train_images]).sum()
h, w = img_size
train_x_n = np.empty((n, h * w * channels), dtype=np.object)
train_y_n = np.empty((n,), dtype=np.int64)
n = np.array([item.shape[0] for item in test_images]).sum()
test_x_n = np.empty((n, h * w * channels), dtype=np.object)
test_y_n = np.empty((n,), dtype=np.int64)
last_len = 0
for i in range(train_images.shape[0]):
    m_train = train_images[i].shape[0]
    m_test = test_images[i].shape[0]
    for j, img in enumerate(train_images[i]):
        train_x_n[i * m_train + j] = img.flatten()
        train_y_n[i * m_train + j] = train_labels[i][j]
    for j, img in enumerate(test_images[i]):
        test_x_n[last_len + j] = img.flatten()
        test_y_n[last_len + j] = test_labels[i][j]
    last_len += m_test
start = datetime.now()
model.fit(train_x_n, train_y_n)
diff = datetime.now() - start
print("time for model fitting: ", diff.total_seconds())
start = datetime.now()
y_pred = model.predict(test_x_n)
print("time for model predicting: ", diff.total_seconds())
accuracy = accuracy_score(test_y_n, y_pred)
print(f'accuracy={accuracy * 100}%')
plt.figure(figsize=(20, 20))
labels = np.array(range(0, 43))
plot_confusion_matrix(test_y_n, y_pred, labels)
plt.savefig('confusion_matrix.png')
plt.close()
plt.figure(figsize=(20, 20))
labels = np.array(range(0, 43))
plot_confusion_matrix(test_y_n, y_pred, labels, normalize=True)
plt.savefig('confusion_matrix_norm.png')
plt.close()
plot_precision_recall(y_true=test_y_n, y_pred=y_pred)
