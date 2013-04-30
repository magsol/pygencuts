import argparse
import numpy as np
import matplotlib.pyplot as plot
import matplotlib.cm as cm
import scipy.ndimage
import scipy.misc

parser = argparse.ArgumentParser(description = 'Random image generation', \
    epilog = 'lol moar clusterz', add_help = 'How to use', \
    prog = 'python occluder.py')
parser.add_argument('--size', '-s', required = True, type = int,
    help = 'Power of 2 to generate the dimensions of the image. This is (pow - 2) for occluder dimensions.')
parser.add_argument('--output', '-o', required = True,
    help = 'Output directory for the image.')

args = vars(parser.parse_args())

size = args['size']
if size < 4:
    quit("ERROR: Need a value for size greater than 3.")
output = args['output']

# Set up the dimensions and the scaling factor.
img_dim = np.power(2, size)
occ_dim = np.power(2, size - 2)
sigma = img_dim / 10.0

# First, generate a bunch of random pixels.
img = np.random.randn(img_dim, img_dim)
img = scipy.ndimage.filters.gaussian_filter(img, sigma = sigma)

# Second, create the occluder.
occ = np.random.randn(occ_dim, occ_dim)
occ = scipy.ndimage.filters.gaussian_filter(occ, sigma = sigma)

# Third, superimpose the occluder on the image.
startInd = (img_dim / 2) - (occ_dim / 2)
endInd = (img_dim / 2) + (occ_dim / 2)
img[startInd:endInd, startInd:endInd] = occ

# Show it and save it.
plot.imshow(img, cmap = cm.gray, interpolation = 'nearest')
plot.show()
scipy.misc.imsave("%soccluder.png" % args['output'], img)
