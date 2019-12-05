import numpy as np
import math

import matplotlib.pyplot as plt
from scipy.misc import imread
from scipy.spatial import distance


def init_centroids(X, K):
    """
    Initializes K centroids that are to be used in K-Means on the dataset X.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Samples, where n_samples is the number of samples and n_features is the number of features.
    K : int
        The number of centroids.

    Returns
    -------
    centroids : ndarray, shape (K, n_features)
    """
    if K == 2:
        return np.asarray([[0., 0., 0.],
                           [0.07843137, 0.06666667, 0.09411765]])
    elif K == 4:
        return np.asarray([[0.72156863, 0.64313725, 0.54901961],
                           [0.49019608, 0.41960784, 0.33333333],
                           [0.02745098, 0., 0.],
                           [0.17254902, 0.16862745, 0.18823529]])
    elif K == 8:
        return np.asarray([[0.01568627, 0.01176471, 0.03529412],
                           [0.14509804, 0.12156863, 0.12941176],
                           [0.4745098, 0.40784314, 0.32941176],
                           [0.00784314, 0.00392157, 0.02745098],
                           [0.50588235, 0.43529412, 0.34117647],
                           [0.09411765, 0.09019608, 0.11372549],
                           [0.54509804, 0.45882353, 0.36470588],
                           [0.44705882, 0.37647059, 0.29019608]])
    elif K == 16:
        return np.asarray([[0.61568627, 0.56078431, 0.45882353],
                           [0.4745098, 0.38039216, 0.33333333],
                           [0.65882353, 0.57647059, 0.49411765],
                           [0.08235294, 0.07843137, 0.10196078],
                           [0.06666667, 0.03529412, 0.02352941],
                           [0.08235294, 0.07843137, 0.09803922],
                           [0.0745098, 0.07058824, 0.09411765],
                           [0.01960784, 0.01960784, 0.02745098],
                           [0.00784314, 0.00784314, 0.01568627],
                           [0.8627451, 0.78039216, 0.69803922],
                           [0.60784314, 0.52156863, 0.42745098],
                           [0.01960784, 0.01176471, 0.02352941],
                           [0.78431373, 0.69803922, 0.60392157],
                           [0.30196078, 0.21568627, 0.1254902],
                           [0.30588235, 0.2627451, 0.24705882],
                           [0.65490196, 0.61176471, 0.50196078]])
    else:
        print('This value of K is not supported.')
        return None


def KMeansIteration(X, k, centroids):
    new_centroids = []
    dict = {}  # the dict save the classification to group in the first
    for x in X:
        # for each point x in X
        #   for each centroid c in Centroids:
        #       calculate distance from x to c
        #   link x to closest centroid c
        distance_to_c = []
        for i in range(k):
            dst = distance.euclidean(x, centroids[i])
            distance_to_c.append(dst)
        index_min = np.argmin(distance_to_c)

        if index_min in dict:
            dict[index_min].append(x)
        else:
            dict[index_min] = [x]

    # now part 2 of algo
    for i in range(k):
        # find the mean point - so we can create new centroid
        mean = np.array(dict[i]).transpose().mean(axis=1)
        # assign to the centroid
        new_centroids.append(mean)
    return new_centroids


def centroid_to_string(centroids):
    arr = []
    for c in centroids:
        arr.append(str([math.floor(p * 100) / 100 for p in c]))

    s = ', '.join(arr)
    s = s.replace('0.0,','0.,').replace('0.0]','0.]')
    return s


def get_avg_loss(X, centroids):
    k = len(centroids)
    sum_dist = 0
    for x in X:
        distance_to_c = []
        for i in range(k):
            dst = distance.euclidean(x, centroids[i])
            distance_to_c.append(dst)
        sum_dist += (np.min(distance_to_c) ** 2)
    sum_dist /= len(X)
    return sum_dist


def kmeans(X, k):
    centroids = init_centroids(X, k)
    losses = []
    for i in range(10):
        print('iter {0}: {1}'.format(i, centroid_to_string(centroids)))
        losses.append(get_avg_loss(X, centroids))
        updated_centroids = KMeansIteration(X, k, centroids)
        centroids = updated_centroids
    print('iter 10: {0}'.format(centroid_to_string(centroids)))
    losses.append(get_avg_loss(X, centroids))
    return centroids, losses
#this code draw the image -you said in piazza that i dont need to submit
# this code so i save this code in comment

# def draw_image(X, centroids, img_shape):
# step 1: change all colors to centroids
#   newX = []
#  for x in X:
#     dist_to_c = []
#    for c in centroids:
#       dist_to_c.append(distance.euclidean(x,c))
#  index_min = np.argmin(dist_to_c)
# newX.append(centroids[index_min])

# now shape newX back into original size
# A = np.reshape(newX, img_shape)
# plt.imshow(A)
# plt.grid(False)
# plt.show()
#
# def draw_losses(k, losses):
#     X_vals = range(len(losses))
#     Y_vals = losses
#
#     plt.xlabel('Iterations')
#     plt.ylabel('Average Loss')
#     plt.title('K=' + str(k))
#     plt.plot(X_vals,Y_vals)
#     plt.show()

def main():
    # data preperation (loading, normalizing, reshaping)
    path = 'dog.jpeg'
    A = imread(path)
    A = A.astype(float) / 255.
    img_size = A.shape
    # print(A.shape)
    X = A.reshape(img_size[0] * img_size[1], img_size[2])
    k_arr = [2, 4, 8, 16]

    for k in k_arr:
        print('k={0}:'.format(k))
        centroids, losses = kmeans(X, k)
        #draw_losses(k,losses)
        #draw_image(X, centroids, A.shape)

    """
    for item in retVal:
        print(item)
    plt.imshow(A)
    plt.grid(False)
    plt.show()"""


if __name__ == '__main__':
    main()
