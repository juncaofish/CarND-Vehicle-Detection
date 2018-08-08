import glob
import pickle

import cv2
import matplotlib.image as mpimg
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

from conf import feature_params
from utils import bin_spatial, color_hist, convert_color, get_hog_features


def extract_features_from_files(img_files, **kwargs):
    """
    extract features from a list of images
    :param img_files:
    :param kwargs:
    :return:
    """

    color_conversion = kwargs.get("color_conversion", 'RGB2YCrCb')
    orient = kwargs.get("orient", 9)
    pix_per_cell = kwargs.get("pix_per_cell", 8)
    cell_per_block = kwargs.get("cell_per_block", 2)
    hog_channel = kwargs.get("hog_channel", 0)
    spatial_size = kwargs.get("spatial_size", (32,32))
    hist_bins = kwargs.get("hist_bins", 32)

    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in img_files:
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        feature_image = convert_color(image, conv=color_conversion)

        # Call get_hog_features() with vis=False, feature_vec=True
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(feature_image[:, :, channel],
                                                     orient, pix_per_cell, cell_per_block,
                                                     visualize=False, feature_vec=True))
            hog_features = np.ravel(hog_features)
        else:
            hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
                                            pix_per_cell, cell_per_block, visualize=False, feature_vec=True)

        # Extract the image patch
        sub_img = cv2.resize(feature_image, (64, 64))

        # Get color features
        spatial_features = bin_spatial(sub_img, size=spatial_size)
        hist_features = color_hist(sub_img, nbins=hist_bins)
        image_features = np.hstack((hog_features, spatial_features, hist_features))

        # Append the new feature vector to the features list
        features.append(image_features)
    return features


if __name__=='__main__':
    cars = glob.glob('training_data/vehicles/*/*.png')
    not_cars = glob.glob('training_data/non-vehicles/*/*.png')

    car_features = extract_features_from_files(cars, **feature_params)
    notcar_features = extract_features_from_files(not_cars, **feature_params)

    # Create an array stack of feature vectors
    x = np.vstack((car_features, notcar_features)).astype(np.float64)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=rand_state)

    # Fit a per-column scaler
    x_scaler = StandardScaler().fit(x_train)

    # Apply the scaler to x
    x_train = x_scaler.transform(x_train)
    x_test = x_scaler.transform(x_test)

    parameters = {'kernel': ('linear', 'rbf'), 'C': [0.5, 1, 5, 10]}
    svc = SVC()
    clf = GridSearchCV(svc, parameters)
    # Check the training time for the SVC
    clf.fit(x_train, y_train)

    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(clf.score(x_test, y_test), 4))

    # Check the prediction time for a single sample
    n_predict = 10
    print('My SVC predicts: ', clf.predict(x_test[0:n_predict]))
    print('Params:', clf.get_params())
    print('For these', n_predict, 'labels: ', y_test[0:n_predict])

    dump_dict = dict()
    dump_dict["svc"] = clf
    dump_dict["scaler"] = x_scaler
    dump_dict["feature_params"] = feature_params

    # Dump the dict with svc, scaler and feature parameters to pickle file.
    with open("svc_pickle.p", "wb") as p:
        pickle.dump(dump_dict, p)
