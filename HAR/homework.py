# spot check on engineered-features

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV


# load a single file as a numpy array


def load_file(filepath):
    """
    load a single file as a numpy array
    :param filepath: a string representing the location of the file.
    :return: numpy array of the file.
    """
    dataframe = 0  # TODO: read the file from the filepath, keep in mind the file doesn't have a header and it is
    # separated with spaces.
    return dataframe.values


def test_load_file():
    X_train = load_file("./UCI HAR Dataset/train/X_train.txt")
    assert X_train.shape == (7352, 561)
    assert X_train[0, 0] == 0.28858451


# load a dataset group, such as train or test
def load_dataset_group(group, prefix=''):
    # load input data
    X = load_file(prefix + group + '/X_' + group + '.txt')
    # load class output
    y = load_file(prefix + group + '/y_' + group + '.txt')
    return X, y


def test_load_dataset_group():
    trainX, trainy = load_dataset_group('train', 'UCI HAR Dataset/')
    assert trainX.shape == (7352, 561)
    assert trainy.shape == (7352, 1)
    assert trainX[0, 0] == 0.28858451
    assert trainy[0] == 5


def load_dataset(prefix=''):
    # load all train
    trainX, trainy = load_dataset_group('train', prefix + 'UCI HAR Dataset/')
    # load all test
    testX, testy = load_dataset_group('test', prefix + 'UCI HAR Dataset/')
    # flatten y
    trainy, testy = trainy[:, 0], testy[:, 0]
    test_reading(trainX.shape, trainy.shape, testX.shape, testy.shape)
    return trainX[:, :], trainy[:], testX[:, :], testy[:]


# load the dataset, returns train and test X and y elements
def test_reading(tr_x_shape, tr_y_shape, te_x_shape, te_y_shape):
    assert tr_x_shape == (7352, 561), "check the read csv function: parameters header and delim_whitespace"
    assert tr_y_shape == (7352,), "check the read csv function: parameters header and delim_whitespace"
    assert te_x_shape == (2947, 561), "check the read csv function: parameters header and delim_whitespace"
    assert te_y_shape == (2947,), "check the read csv function: parameters header and delim_whitespace"


# create a dict of standard models to evaluate {name:object}
def define_models(models=dict(), params=dict()):
    # nonlinear models
    models['sgd'] = SGDClassifier(max_iter=1000, tol=1e-3)
    params['sgd'] = {'penalty': ('l2', 'l1')}

    models['log_reg'] = 0  # TODO: create Log Reg model with parameter solver: lbfgs, multi_class auto, max_iter:500
    params['log_reg'] = 0  # TODO: create a dict of params with multi_class with two options ovr, multinomial

    models['svm'] = 0  # TODO: create SVC model with parameter gamma:scale
    params["svm"] = 0  # TODO: create a dict of params with kernel with two options linear, poly
    print('Defined %d models' % len(models))
    return models, params


def grid_search_models(models, params):
    grid_search_cv_models = dict()
    for key in models:
        grid_search_cv_models[key] = GridSearchCV(models[key], params[key], cv=10, iid=False)
    return grid_search_cv_models


# evaluate a single model
def evaluate_model(trainX, trainy, testX, testy, model, class_names, title):
    # fit the model
    # TODO: call the fit function with the training data
    # make predictions
    yhat = 0  # TODO: make predictions for the test data
    # evaluate predictions
    accuracy = 0  # TODO: evaluate predictions using accuracy on test data
    # Confusion Matrix
    plot_confusion_matrix(testy, yhat, class_names, normalize=True, title=title)
    return accuracy * 100.0


# evaluate a dict of models {name:object}, returns {name:score}
def evaluate_models(trainX, trainy, testX, testy, models, class_names):
    results = dict()
    for name, model in models.items():
        # evaluate the model
        results[name] = evaluate_model(trainX, trainy, testX, testy, model, class_names, name)
        # show process
        print('>%s: %.3f' % (name, results[name]))
    return results


# print and plot the results
def summarize_results(results, maximize=True):
    # create a list of (name, mean(scores)) tuples
    mean_scores = [(k, v) for k, v in results.items()]
    # sort tuples by mean score
    mean_scores = sorted(mean_scores, key=lambda x: x[1])
    # reverse for descending order (e.g. for accuracy)
    if maximize:
        mean_scores = list(reversed(mean_scores))
    print()
    for name, score in mean_scores:
        print(name, score)


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
    cm = 0  # TODO: calculate the confusion matrix
    # Only use the labels that appear in the data
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


def read_class_names():
    classes = []
    with open("UCI HAR Dataset/activity_labels.txt", "r") as f:
        for l in f:
            classes.append(l[2:].strip())
    return classes


if __name__ == '__main__':
    trainX, trainy, testX, testy = load_dataset()
    class_names = read_class_names()
    # get model list
    models, params = define_models()
    # Wrap the models with GridSearchCV
    models_cv = grid_search_models(models, params)
    # evaluate models
    results = evaluate_models(trainX, trainy, testX, testy, models_cv, class_names)
    # summarize results
    summarize_results(results)
    # plot Confusion Matrix
