import glob
import matplotlib.image as mpimg
import cv2
from skimage.feature import hog
import matplotlib.pyplot as plt
import numpy as np


def normalize_image(img):
    """
    Normalize image between 0 and 255 and cast to uint8
    (useful for visualization)
    """
    img = np.float32(img)
    img = img / img.max() * 255
    return np.uint8(img)


def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)


def bin_spatial(img, size=(32, 32)):
    color1 = cv2.resize(img[:, :, 0], size).ravel()
    color2 = cv2.resize(img[:, :, 1], size).ravel()
    color3 = cv2.resize(img[:, :, 2], size).ravel()
    return np.hstack((color1, color2, color3))


def color_hist(img, nbins=32):  # bins_range=(0, 256)
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                     visualize=False, feature_vec=True):
    """
    Extract HOG features and visualization.
    :param img:
    :param orient:
    :param pix_per_cell:
    :param cell_per_block:
    :param visualize:
    :param feature_vec:
    :return:
    """
    if visualize:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), block_norm='L2-Hys',
                                  transform_sqrt=True,
                                  visualise=visualize, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), block_norm='L2-Hys',
                       transform_sqrt=True,
                       visualise=visualize, feature_vector=feature_vec)
        return features


if __name__ == "__main__":
    cars = glob.glob('training_data/vehicles/*/*.png')
    not_cars = glob.glob('training_data/non-vehicles/*/*.png')

    from conf import feature_params

    orient = feature_params["orient"]
    pix_per_cell = feature_params["pix_per_cell"]
    cell_per_block = feature_params["cell_per_block"]
    nbins = feature_params["hist_bins"]
    color_conversion = feature_params["color_conversion"]

    f, axes = plt.subplots(2, 5)
    for i in range(5):
        rand_state = np.random.randint(0, len(cars) - 1)
        car = mpimg.imread(cars[rand_state])
        axes[0, i].imshow(car, cmap='gray')
        axes[0, i].set_axis_off()
        axes[0, 2].set_title("CAR")

        not_car = mpimg.imread(not_cars[rand_state])
        axes[1, i].imshow(not_car, cmap='gray')
        axes[1, i].set_axis_off()
        axes[1, 2].set_title("NOT CAR")

    plt.show()
    f.savefig("output_images/car_not_car.png", format='png', bbox_inches='tight', transparent=True)

    feature_image = convert_color(car, conv=color_conversion)
    f, axes = plt.subplots(1, 4)
    axes[0].imshow(feature_image[:, :, 0], cmap='gray')
    axes[0].set_title('Car')
    axes[0].set_axis_off()

    _, hog_img = get_hog_features(feature_image[:, :, 0], orient, pix_per_cell, cell_per_block,
                                  visualize=True, feature_vec=True)
    axes[1].imshow(hog_img, cmap='gray')
    axes[1].set_title('Car Hog')
    axes[1].set_axis_off()

    rand_state = np.random.randint(0, len(not_cars) - 1)
    not_car = mpimg.imread(not_cars[rand_state])
    feature_image = convert_color(not_car, conv=color_conversion)
    axes[2].imshow(feature_image[:, :, 0], cmap='gray')
    axes[2].set_title('Not Car')
    axes[2].set_axis_off()
    _, hog_img = get_hog_features(feature_image[:, :, 0], orient, pix_per_cell, cell_per_block,
                                  visualize=True, feature_vec=True)
    axes[3].imshow(hog_img, cmap='gray')
    axes[3].set_title('Not Car Hog')
    axes[3].set_axis_off()

    plt.show()
    # f.savefig("output_images/HOG_example.png", format='png', bbox_inches='tight', transparent=True)
