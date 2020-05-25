# Compressing an image
import argparse
import numpy as np
from scipy import misc
from sklearn import cluster
import matplotlib.pyplot as plt

def build_arg_parser():
    parser = argparse.ArgumentParser(description='Compress the input image \
    using clustering')
    parser.add_argument("--input-file", 
                        dest="input_file",
                        required=True,
                        help="Input image")
    parser.add_argument("--num-bits", 
                        dest="num_bits",
                        required=False,
                        type=int, 
                        help="Number of bits used to represent each pixel")
    return parser

def compress_image(img, num_clusters):
    # Convert input image into (num_samples, num_features)
    # array to run kmeans clustering algorithm
    X = img.reshape((-1, 1))
    # Run kmeans on input data
    kmeans = cluster.KMeans(n_clusters=num_clusters, 
                            n_init=4,
                            random_state=5)
    kmeans.fit(X)
    centroids = kmeans.cluster_centers_.squeeze()
    labels = kmeans.labels_
    # Assign each value to the nearest centroid and
    # reshape it to the original image shape
    input_image_compressed = np.choose(labels, centroids).reshape(img.shape)
    return input_image_compressed

def plot_image(img, title):
    vmin = img.min()
    vmax = img.max()
    plt.figure()
    plt.title(title)
    plt.imshow(img, cmap=plt.cm.gray, vmin=vmin, vmax=vmax)
    
if __name__=='__main__':
    args = build_arg_parser().parse_args()
    input_file = args.input_file
    num_bits = args.num_bits
    if not 1 <= num_bits <= 8:
         TypeError('Number of bits should be between 1 and 8')
    num_clusters = np.power(2, num_bits)
    # Print compression rate
    compression_rate = round(100 * (8.0 - args.num_bits) / 8.0, 2)
    print("The size of the image will be reduced by a factor of", 8.0/args.num_bits)
    print("Compression rate = " + str(compression_rate) + "%")    
    
# Load input image
import imageio
input_image = imageio.imread('flower_image.jpg')
input_image.shape
(512, 512, 3)
imageio.imwrite('flower_image1.jpg', input_image) 

# original image
plot_image(input_image, 'Original image')
# compressed image
input_image_compressed = compress_image(input_image, num_clusters)
plot_image(input_image_compressed, 'Compressed image; compression rate = '+ str(compression_rate) + '%')
plt.show()


    