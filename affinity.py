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
import sklearn.metrics.pairwise as pairwise

def _grid_to_graph(nx, ny, return_as = scipy.sparse.coo_matrix, neighborhood = 8):
    """
    Drop-in replacement for scikit-learn's grid_to_graph method, as it can only
    compute neighborhoods of 4 pixels. We want the corners as well.

    Parameters
    ----------
    nx : integer
        Number of rows (image height).
    ny : integer
        Number of columns (image width).
    return_as : matrix type
        Can also be np.ndarray for a dense matrix.
    neighborhood : integer
        4 or 8 pixel neighborhood.

    Returns
    -------
    graph : array or sparse, shape (N, N)
        The connectivity graph for the image.
    """
    n_voxels = nx * ny
    vertices = np.arange(n_voxels).reshape((nx, ny))
    edges_right = np.vstack((vertices[:, :-1].ravel(), vertices[:, 1:].ravel()))
    edges_down = np.vstack((vertices[:-1].ravel(), vertices[1:].ravel()))

    edges = None
    if neighborhood == 4:
        edges = np.hstack((edges_right, edges_down))
    elif neighborhood == 8:
        edges_tc = np.vstack((vertices[1:, :-1].ravel(), vertices[:-1, 1:].ravel()))
        edges_bc = np.vstack((vertices[:-1, :-1].ravel(), vertices[1:, 1:].ravel()))
        edges = np.hstack((edges_right, edges_down, edges_tc, edges_bc))
    else:
        quit('ERROR: Unrecognized neighborhood size %s.' % neighborhood)

    weights = np.ones(edges.shape[1])
    diag = np.ones(n_voxels)
    diag_idx = np.arange(n_voxels)
    i_idx = np.hstack((edges[0], edges[1]))
    j_idx = np.hstack((edges[1], edges[0]))

    graph = sparse.coo_matrix((np.hstack((weights, weights, diag)),
        (np.hstack((i_idx, diag_idx)),
        np.hstack((j_idx, diag_idx)))),
        (n_voxels, n_voxels))
    return graph

def _differences(patch, i, j):
    """
    Computes the absolute gray-level difference between all the 
    pixels in the patch, relative to the current position. If the patch is
    not 3x3, i and j are used to determine where the current pixel is.
 
    Parameters
    ----------
    patch : array, shape (N, M)
        Image patch. N * M is always either 4, 6, or 8.
    i : integer
        Row of the current pixel (ignored if N == M == 3).
    j : integer
        Column of the current pixel (ignored if N == M == 3).
 
    Returns
    -------
    diffs : array, shape (P,)
        List of gray-level absolute differences.
    """
    if patch.shape[0] == 3: i = 1
    if patch.shape[1] == 3: j = 1
    differences = []
    for a in range(0, patch.shape[0]):
        for b in range(0, patch.shape[1]):
            if a == i and b == j: continue
            differences.append(np.abs(patch[a, b] - patch[i, j]))
    return np.array(differences)

def _median_difference(image):
    """
    Computes the median absolute gray-level difference between all the 
    pixels in the image.

    Parameters
    ----------
    image : array, shape (H, W)
        Matrix of gray-level intensities.

    Returns
    -------
    md : float
        Median absolute gray-level difference between all pixels.
    """
    D = []
    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            startrow = np.max([0, i - 1])
            endrow = np.min([image.shape[0], i + 2])
            startcol = np.max([0, j - 1])
            endcol = np.min([image.shape[1], j + 2])
            patch = image[startrow:endrow, startcol:endcol]
            locali = 0 if i - 1 < 0 else 1
            localj = 0 if j - 1 < 0 else 1

            D.extend(_differences(patch, locali, localj))
    return np.median(np.array(D))

def image_affinities(image, q = 1.5, gamma = 0.0):
    """
    Calculates a sparse affinity matrix from image data, where each pixel is
    connected only to its (at most) 8 neighbors. Furthermore, the sigma used
    is computed on a local basis.

    Parameters
    ----------
    image : array, shape (P, Q)
        Grayscale image.
    q : float
        Multiplier to compute gamma.
    gamma : float
        If specified and positive, this overrides the use of the multiplier q
        and of computing gamma on a per-neighborhood basis.

    Returns
    -------
    A : array, shape (P * Q, P * Q)
        Symmetric affinity matrix.
    """
    med = _median_difference(image)
    std = gamma
    if gamma <= 0.0:
        std = 1.0 / (2 * ((med * q) ** 2))
    numpixels = image.shape[0] * image.shape[1]
    graph = _grid_to_graph(image.shape[1], image.shape[0])
    connections = graph.nonzero()
    A = sparse.lil_matrix(graph.shape)

    # For each non-zero connection, compute the affinity.
    # We have to do this one at a time in a loop; rbf_kernel() doesn't have
    # a sparse mode, and therefore computing all the affinities at once--even
    # sparse ones--could overwhelm system memory.
    for k in xrange(0, (np.size(connections[0]) / 2) + 1):
        # Pixel IDs.
        i = connections[0][k]
        j = connections[1][k]

        # Where do the pixels reside?
        r1 = i / image.shape[1]
        c1 = i % image.shape[1]
        r2 = j / image.shape[1]
        c2 = j % image.shape[1]

        # Compute the RBF value.
        rbf = pairwise.rbf_kernel(image[r1, c1], image[r2, c2], gamma = std)[0, 0]
        A[i, j] = rbf
        A[j, i] = rbf
        A[i, i] = 1.0
        A[j, j] = 1.0
    #return A
    return np.array(A.todense())

def cartesian_affinities(data, distance = 2.0, sigma = 1.0):
    """
    Computes affinities between points using euclidean distance, and 
    sets to 0 all affinities for which the points are further than a certain
    threshold apart.

    Parameters
    ----------
    data : array, shape (N, M)
        N instances of M-dimensional data.
    distance : float
        Distance threshold, above which all affinities are set to 0.
    sigma : float
        Sigma used to compute affinities.

    Returns
    -------
    A : array, shape (N, N)
        Symmetric affinity matrix.
    """
    A = pairwise.rbf_kernel(data, data, gamma = (1.0 / (2 * (sigma ** 2))))
    if (distance > 0.0):
        distances = pairwise.pairwise_distances(data)
        A[np.where(distances > distance)] = 0.0
    return A

def mahout_affinities(infile, n):
    """
    Reads Mahout-style affinity file.

    Parameters
    ----------
    infile : string
        Path to a Mahout-style affinity file.
    n : integer
        Number of points.

    Returns
    -------
    A : array, shape (N, N)
        Symmetric affinity matrix.
    """
    A = np.zeros(shape = (n, n))
    for line in file(infile):
        row, col, val = map(float, line.strip().split(","))
        A[int(row), int(col)] = val
    return A

def write_mahout_affinity(A, outfile):
    """
    Writes the specified affinity matrix to an output file in Mahout format.

    Parameters
    ----------
    A : array or sparse matrix, shape (N, N)
        Symmetric affinity matrix.
    outfile : string
        Output text file where the matrix will be written.
    """
    f = open(outfile, "w")
    rows = None
    cols = None
    if sparse.isspmatrix(A):
        if not A.format == 'coo':
            A = A.tocoo()
        rows, cols = A.nonzero()
    else:
        rows, cols = np.nonzero(A)
    for row, col in zip(rows, cols):
        f.write("%s,%s,%s\n" % (row, col, A[row, col]))
    f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Eigencuts Affinity Generation', \
        epilog = 'lol pygencuts', add_help = 'How to use', \
        prog = 'python affinity.py <options>')
    parser.add_argument('-i', '--input', required = True,
        help = 'Path to input.')
    parser.add_argument('-t', '--type', choices = ['img', 'txt'], required = True,
        help = 'Specifies the type of input: comma-separated instances, or a PNG image.')
    parser.add_argument('-o', '--output', required = True,
        help = 'Path to output directory to write Mahout affinity data.')

    # Optional arguments for "txt" input type.
    parser.add_argument('-d', '--distance', type = float, default = 2.0,
        help = 'For "txt" data, the neighborhood distance threshold for computing affinities. [DEFAULT: 2.0]')
    parser.add_argument('-s', '--sigma', type = float, default = 1.0,
        help = 'For "txt" data, the standard deviation used to compute affinities. [DEFAULT: 1.0]')

    args = vars(parser.parse_args())
    A = None
    outfile = "%s.txt" % ".".join(args['input'].split("/")[-1].split(".")[:-1])
    if args['type'] == 'img':
        data = scipy.misc.imread(args['input'], flatten = True)
        A = image_affinities(data)
    else:
        data = np.loadtxt(args['input'], delimiter = ",")
        A = cartesian_affinities(data, args['distance'], args['sigma'])

    # We have the affinity matrix, now write it out!
    write_mahout_affinity(A, "%s%s" % (args['output'], outfile))
