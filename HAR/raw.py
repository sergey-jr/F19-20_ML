# spot check on raw data
import numpy as np


def load_file(filepath):
    """
    load a single file as a numpy array
    :param filepath: a string representing the location of the file.
    :return: numpy array of the file.
    """
    dataframe = 0  # TODO: read the file from the filepath, keep in mind the file doesn't have a header and it is
    # separated with spaces.
    return dataframe.values


def load_group(filenames, prefix=''):
    """
    load a list of files into a 3D array of [samples, timesteps, features]
    :param filenames: list of filenames
    :param prefix: if you have located the file in a different folder
    :return:
    """
    loaded = list()
    for name in filenames:
        data = load_file(prefix + name)
        # TODO: add the data into loaded variable
    # stack group so that features are the 3rd dimension
    loaded = 0  # TODO: stack the loaded variable into a 3d array, hint numpy dstack
    return loaded


# load a dataset group, such as train or test
def load_dataset_group(group, prefix=''):
    filepath = prefix + group + '/Inertial Signals/'
    # load all 9 files as a single array
    filenames = list()
    # total acceleration
    filenames += ['total_acc_x_' + group + '.txt', 'total_acc_y_' + group + '.txt', 'total_acc_z_' + group + '.txt']
    # body acceleration
    filenames += ['body_acc_x_' + group + '.txt', 'body_acc_y_' + group + '.txt', 'body_acc_z_' + group + '.txt']
    # body gyroscope
    filenames += ['body_gyro_x_' + group + '.txt', 'body_gyro_y_' + group + '.txt', 'body_gyro_z_' + group + '.txt']
    # load input data
    X = load_group(filenames, filepath)
    # load class output
    y = load_file(prefix + group + '/y_' + group + '.txt')
    return X, y


def convert_sample(sample):
    """
    converts a sample of 128*9 into 36 values
    :param sample: a numpy array of (128,9) shape
    :return:
    """
    converted = 0  # TODO: convert that sample into an array of means,stds,max and mins
    # Hint: use hstack along with np.mean,std,max,min and check the axis parameter
    return converted


# load the dataset, returns train and test X and y elements
def extract_features(trainX):
    extracted_features = np.zeros((trainX.shape[0], trainX.shape[-1] * 4))
    for i in range(len(trainX)):
        extracted_features[i] = convert_sample(trainX[i])
    return extracted_features


def load_dataset(prefix=''):
    # load all train
    trainX, trainy = load_dataset_group('train', prefix + 'UCI HAR Dataset/')
    # load all test
    testX, testy = load_dataset_group('test', prefix + 'UCI HAR Dataset/')
    trainX = extract_features(trainX)
    testX = extract_features(testX)
    # flatten y
    trainy, testy = trainy[:, 0], testy[:, 0]
    return trainX, trainy, testX, testy


trainX, trainy, testX, testy = load_dataset()
assert trainX.shape == (7352, 36)
assert trainy.shape == (7352,)
assert testX.shape == (2947, 36)
assert testy.shape == (2947,)
