import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import glob
import pickle
import collections
from moviepy.editor import VideoFileClip

import scipy
from utils import *
from draw_detection import compute_heatmap_from_detections, draw_labeled_bounding_boxes, draw_boxes

time_window = 3 

hot_windows_history = collections.deque(maxlen=time_window)


def process_pipeline(frame, keep_state=True, visualize=False):
    global svc, feature_scaler, feature_params
    hot_windows = []

    for scale in np.arange(1, 3, 0.5):
        hot_windows += find_cars(frame, 400, 600, 300, 1290, scale, svc, feature_scaler, feature_params)

    if keep_state:
        if hot_windows:
            hot_windows_history.append(hot_windows)
            hot_windows = np.concatenate(hot_windows_history)

    # compute heatmaps positive windows found
    thresh = (time_window - 1) if keep_state else 1
    heatmap, heatmap_thresh = compute_heatmap_from_detections(frame, hot_windows, threshold=thresh)

    # label connected components
    labeled_frame, num_objects = scipy.ndimage.measurements.label(heatmap_thresh)

    # prepare images for blend
    img_hot_windows = draw_boxes(frame, hot_windows, color=(0, 255, 0), thick=2)                 # show pos windows
    img_heatmap = cv2.applyColorMap(normalize_image(heatmap), colormap=cv2.COLORMAP_HOT)         # draw heatmap
    img_labeling = cv2.applyColorMap(normalize_image(labeled_frame), colormap=cv2.COLORMAP_HOT)  # draw label
    img_detection = draw_labeled_bounding_boxes(frame.copy(), labeled_frame, num_objects)        # draw detected bboxes

    if visualize:
        f, axes = plt.subplots(2, 2)
        axes[0, 0].imshow(img_hot_windows)
        axes[0, 1].imshow(img_heatmap)
        axes[1, 0].imshow(img_labeling)
        axes[1, 1].imshow(img_detection)
        plt.show()

    return img_detection


def find_cars(img, ystart, ystop, xstart, xstop, scale, svc, scaler, feature_params):
    """
    extract features using hog sub-sampling and make predictions
    :param img:
    :param ystart:
    :param ystop:
    :param xstart:
    :param xstop:
    :param scale:
    :param svc:
    :param scaler:
    :param feature_params:
    :return:
    """

    orient, pix_per_cell, cell_per_block, spatial_size, hist_bins = \
        feature_params["orient"], feature_params["pix_per_cell"], feature_params["cell_per_block"], \
        feature_params["spatial_size"], feature_params["hist_bins"]

    draw_img = np.copy(img)
    img = img.astype(np.float32) / 255
    hot_windows = []

    img_tosearch = img[ystart:ystop, xstart:xstop, :]
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    ch1 = ctrans_tosearch[:, :, 0]
    ch2 = ctrans_tosearch[:, :, 1]
    ch3 = ctrans_tosearch[:, :, 2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
    nfeat_per_block = orient * cell_per_block ** 2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 4  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            image_features = np.hstack((hog_features, spatial_features, hist_features)).reshape(1, -1)

            # Scale features and make a prediction
            test_features = scaler.transform(image_features)
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)
                tl_corner_draw = (xbox_left + xstart, ytop_draw + ystart)
                br_corner_draw = (xbox_left + win_draw + xstart, ytop_draw + win_draw + ystart)
                cv2.rectangle(draw_img, tl_corner_draw,
                              br_corner_draw, (0, 0, 255), 5)
                hot_windows.append((tl_corner_draw, br_corner_draw))

    return hot_windows


if __name__ == '__main__':
    dist_pickle = pickle.load(open("svc_pickle.p", "rb"))

    # get attributes of our svc object
    svc = dist_pickle["svc"]
    feature_scaler = dist_pickle["scaler"]
    feature_params = dist_pickle["feature_params"]

    orient = feature_params["orient"]
    pix_per_cell = feature_params["pix_per_cell"]
    cell_per_block = feature_params["cell_per_block"]
    spatial_size = feature_params["spatial_size"]
    hist_bins = feature_params["hist_bins"]

    # test_img_dir = 'test_images'
    # images = glob.glob('test_images/*.jpg')
    # f, axes = plt.subplots(3, 2)
    # for idx, file in enumerate(images[::2]):
    #     img = mpimg.imread(file)
    #     img_out = process_pipeline(img, False, False)
    #
    #     axes[idx, 0].imshow(img)
    #     axes[idx, 0].set_axis_off()
    #     if idx == 0:
    #         axes[idx, 0].set_title("Original Image")
    #
    #     axes[idx, 1].imshow(img_out)
    #     axes[idx, 1].set_axis_off()
    #     if idx == 0:
    #         axes[idx, 1].set_title("Bounding Box")
    # plt.show()
    # f.savefig("output_images/bbox.png", format='png', bbox_inches='tight', transparent=True)

    selector = "project"
    clip = VideoFileClip('{}_video.mp4'.format(selector)).fl_image(process_pipeline)
    clip.write_videofile('output_{}.mp4'.format(selector), audio=False)
