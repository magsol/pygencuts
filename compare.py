"""
Copyright 2013 University of Pittsburgh

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""
import argparse
import numpy as np
import scipy.misc
import scipy.sparse as sparse
import scipy.sparse.csgraph as csgraph
import sklearn.cluster as cluster
import sklearn.metrics.pairwise as pairwise
import matplotlib.pyplot as plot
import matplotlib.cm as cm
import eigencuts

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Comparative Spectral Clustering', \
        epilog = 'lol pygencuts', add_help = 'How to use', \
        prog = 'python compare.py <options>')
    parser.add_argument('-i', '--input', required = True,
        help = 'Path to input (image, cartesian, or mahout affinities).')
    parser.add_argument('-t', '--type', choices = ['img', 'txt', 'aff'], required = True,
        help = 'Specifies the type of input: comma-separated instances, or a PNG image.')

    # Optional arguments.
    parser.add_argument('-b', '--halflife', type = float, default = 40.0,
        help = 'Eigenflow threshold to Eigencuts.')
    parser.add_argument('-k', '--n_clusters', type = int, default = -1,
        help = 'Number of clusters to find in the data, or -1 to use number of clusters in Eigencuts. [DEFAULT: -1]')
    parser.add_argument('-o', '--original', default = None,
        help = 'Required if type == "aff". Specifies the path to the original data.')
    parser.add_argument('-n', '--numdims', type = int, default = 150,
        help = "Required if type is 'aff'. Number of points in the dataset.")
    parser.add_argument('-d', '--distance', type = float, default = 2.0,
        help = 'For "txt" data, the neighborhood distance threshold for computing affinities. [DEFAULT: 2.0]')
    parser.add_argument('-s', '--sigma', type = float, default = 1.0,
        help = 'For "txt" data, the standard deviation used to compute affinities. [DEFAULT: 1.0]')

    args = vars(parser.parse_args())

    # Read the input data.
    A = None
    data = None
    if args['type'] == "txt":
        # Comma-separated cartesian data.
        data = np.loadtxt(args['input'], delimiter = ",")
        A = eigencuts.cartesian_affinities(data, args['distance'], args['sigma'])
    elif args['type'] == "img":
        # PNG image.
        data = scipy.misc.imread(args['input'], flatten = True)
        A = eigencuts.image_affinities(data)
    else:
        # Mahout affinity data.
        data = np.loadtxt(args['original'], delimiter = ",")
        A = eigencuts.mahout_affinities(args['input'], args['numdims'])

    # Do the Eigencuts analysis.
    Ae, n = eigencuts.eigencuts(A, k = 10, b = args['halflife'])

    Abin = None
    if sparse.isspmatrix(Ae):
        Abin = sparse.csc_matrix(Ae).sign()
    else:
        Abin = np.sign(Ae)
    numConn, connMap = csgraph.connected_components(Abin, directed = False)
    print 'Found %s clusters.' % numConn

    # We can use spectral clustering. Use K-means only if it's text data.
    numClusters = args['n_clusters'] if args['n_clusters'] > 0 else numConn
    spectral = cluster.SpectralClustering(n_clusters = numClusters,
        affinity = "precomputed")
    y_spectral = spectral.fit_predict(A)

    if args['type'] == 'img':
        y_spectral = np.reshape(y_spectral, newshape = data.shape)
        connMap = np.reshape(connMap, newshape = data.shape)

        plot.figure(0)
        plot.imshow(np.reshape(y_spectral, newshape = data.shape), interpolation = 'nearest')
        #plot.imshow(data, cmap = cm.gray)
        #for i in range(0, numClusters):
        #    plot.contour(y_spectral == i, contours = 1,
        #        colors = [cm.spectral(i / float(numClusters)), ])
        plot.title('Spectral Clustering')

        plot.figure(1)
        plot.imshow(np.reshape(connMap, newshape = data.shape), interpolation = 'nearest')
        #plot.imshow(data, cmap = cm.gray)
        #for i in range(0, numConn):
        #    plot.contour(connMap == i, contours = 1,
        #        colors = [cm.spectral(i / float(numConn)), ])
        plot.title('Eigencuts')

        plot.figure(2)
        plot.imshow(data, cmap = cm.gray, interpolation = 'nearest')

        plot.show()
    else:
        # Cartesian data.
        kmeans = cluster.KMeans(n_clusters = numClusters)
        y_kmeans = kmeans.fit_predict(data)

        colormap = cm.get_cmap("spectral", numClusters)
        colorvals = colormap(np.arange(numClusters))

        # Plot spectral.
        plot.figure(0)
        colors = [colorvals[y_spectral[i]] for i in range(0, np.size(y_spectral))]
        for i in range(0, data.shape[0]):
            plot.plot(data[i, 0], data[i, 1], marker = 'o', c = colors[i])
        plot.title('Spectral Clustering')

        # Plot kmeans.
        plot.figure(1)
        colors = [colorvals[y_kmeans[i]] for i in range(0, np.size(y_kmeans))]
        for i in range(0, data.shape[0]):
            plot.plot(data[i, 0], data[i, 1], marker = 'o', c = colors[i])
        plot.title('KMeans')

        # Plot eigencuts.
        plot.figure(2)
        colormap = cm.get_cmap("spectral", numConn)
        colorvals = colormap(np.arange(numConn))
        colors = [colorvals[connMap[i]] for i in range(0, np.size(connMap))]
        for i in range(0, data.shape[0]):
            plot.plot(data[i, 0], data[i, 1], marker = 'o', c = colors[i])
        #plot.title('Eigencuts')

        plot.show()
