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
import scipy.linalg as la
import scipy.sparse as sparse
import scipy.sparse.csgraph as csgraph
import scipy.sparse.linalg as sla
import matplotlib.pyplot as plot
import matplotlib.cm as cm
import affinity

def sensitivities(u, l, d, b, A):
    """
    Computes half-life sensitivities for the given eigenvector/eigenvalue pair.

    Parameters
    ----------
    u : array, shape (N,)
        Eigenvector.
    l : float
        Associated eigenvalue.
    d : array, shape (N,)
        Diagonal matrix.
    b : float
        Beta0.
    A : array, shape (N, N)
        Affinity matrix.

    Returns
    -------
    S : array, shape (N, N)
        Matrix of sensitivities.
    """
    S = np.zeros(shape = A.shape)
    for i in range(0, np.size(u)):
        for j in range(0, np.size(u)):
            if i == j: continue
            if A[i, j] == 0.0: continue

            # There is a connection here in the graph.
            prefix = np.log(2) / (l * np.log(l) * np.log((l ** b) / 2.0))
            firstterm = - ((u[i] / np.sqrt(d[i])) - (u[j] / np.sqrt(d[j]))) ** 2
            secondterm = ((u[i] ** 2) / d[i]) + ((u[j] ** 2) / d[j])

            S[i, j] = prefix * (firstterm + ((1 - l) * secondterm))
    return S

def suppression(S, t, d):
    """
    Performs non-maximal suppression. Sensitivities are suppressed (set to 0)
    if there is a sensitivity in the same row or column that is more negative.

    Parameters
    ----------
    S : array, shape (N, N)
        Matrix of sensitivities.
    t : float
        Tau.
    d : float
        Delta.

    Returns
    -------
    Ss : array, shape (N, N)
        Contains at most N non-suppressed sensitivities.
    """
    Ss = np.copy(S)
    for i in range(0, np.size(S, axis = 0)):
        for j in range(0, np.size(S, axis = 1)):
            if np.size(np.where(S[i, :] < S[i, j])) > 0 or np.size(np.where(S[:, j] < S[i, j])) > 0:
                Ss[i, j] = (t / d)
    return Ss

def update(A, S, t, d):
    """
    Parameters
    ----------
    A : array or sparse matrix, shape (N, N)
        Original affinity matrix.
    S : array, shape (N, N)
        Sensitivity matrix.
    t : float
        Tau.
    d : float
        Delta.

    Returns
    -------
    Ac : array, shape (N, N)
        Trimmed affinity matrix.
    n : integer
        Number of cuts made to the underlying graph.
    """
    Ac = A.copy()
    n = 0
    for i in range(0, np.size(A, axis = 0)):
        for j in range(i, np.size(A, axis = 1)):
            if i == j: continue
            if S[i, j] >= (t / d): continue

            # Cut the affinity matrix (symmetrically), and move the weight
            # into the diagonal.
            Ac[i, i] += Ac[i, j]
            Ac[i, j] = 0
            Ac[j, j] += Ac[j, i]
            Ac[j, i] = 0
            n += 2
    return [Ac, n]

def eigencuts(A, k = 5, b = 40.0, e = 0.25, t = -0.2, verbose = True):
    """
    Runs the Eigencuts algorithm on the affinity matrix.

    Parameters
    ----------
    A : array-like or sparse, shape (N, N)
        Pairwise affinity data, representing the graph stucture.
    k : integer
        Eigenrank of the system, or number of top eigenvectors/eigenvalues to keep.
    b : float
        Beta_0.
    e : float
        Epsilon.
    t : float
        Tau.
    verbose : boolean
        If True, prints out a few debug statements.

    Returns
    -------
    A : array-like or sparse, shape (N, N)
        Processed affinity matrix.
    n : integer
        Number of iterations performed.
    """
    # The diagonal matrix never changes, so let's just compute it now.
    # Note: since A is symmetric, summing across the rows vs columns should
    # give the same result (since a CSC matrix is optimized for columns).
    D = None
    if sparse.isspmatrix(A):
        D = np.array(A.sum(axis = 0))[0, :]
    else:
        D = np.sum(A, axis = 1)
    Ac = A.copy()
    delta = np.median(D)

    # Now, start the loop.
    numCuts = 1
    iterNum = 0
    while (numCuts > 0):
        # Compute the normalized graph Laplacian.
        # Viciously ripped the following four lines from scikit-learn.
        # Because this is bloody awesome.
        values = None
        vectors = None
        if sparse.isspmatrix(Ac):
            n_nodes = Ac.shape[0]
            if not Ac.format == 'coo':
                L = Ac.tocoo()
            else:
                L = Ac.copy()
            w = np.sqrt(np.asarray(L.sum(axis = 0)).squeeze())
            L.data /= w[L.row]
            L.data /= w[L.col]

            # Perform the SVD.
            out = sla.svds(L, k = k)
            vectors = out[0]
            values = out[1]
        else:
            L = Ac.copy()
            d = np.sqrt(D)
            L /= d
            L /= d[:, np.newaxis]

            # Perform the SVD.
            U, s, Vh = la.svd(L)
            values = s[:k]
            vectors = U[:, :k]
        
        # For each eigenvector with a half-life beyond the threshold,
        # compute sensitivities.
        numCuts = 0
        for i in range(0, np.size(vectors, axis = 1)):
            v = vectors[:, i]
            l = values[i]
            if l == 1.0: continue # would result in divide-by-zero error
            betak = -np.log(2) / np.log(l)
            if betak <= (e * b): continue
            Sk = sensitivities(v, l, D, b, Ac)
            
            # Comment out the following line to ignore non-maximal suppression.
            Sk = suppression(Sk, t, delta)

            # Cut edges.
            Ac, n = update(Ac, Sk, t, delta)
            numCuts += n

        iterNum += 1
        numEdges = np.size(np.nonzero(Ac)[0]) - Ac.shape[0]
        if verbose:
            print '%s: %s cuts made, %s edges left.' % (iterNum, numCuts, numEdges)
    return [Ac, numCuts]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Eigencuts Spectral Clustering', \
        epilog = 'lol pygencuts', add_help = 'How to use', \
        prog = 'python eigencuts.py <options>')
    parser.add_argument('-i', '--input', required = True,
        help = 'Path to input.')
    parser.add_argument('--type', choices = ['img', 'txt', 'aff'], required = True,
        help = 'Specifies the type of input: comma-separated instances, or a PNG image.')

    # Optional arguments.
    parser.add_argument('--mahout_data', default = None,
        help = 'Required if type == "aff". Specifies the path to the original data.')
    parser.add_argument('-n', '--numdims', type = int, default = 150,
        help = "Required if type is 'aff'. Number of points in the dataset.")
    parser.add_argument('--eigenrank', type = int, default = 10,
        help = 'Number of eigenvectors to use.')
    parser.add_argument('-b', '--halflife', type = float, default = 40.0,
        help = 'Value for beta0, the half-life threshold. [DEFAULT: 40.0]')
    parser.add_argument('-e', '--epsilon', type = float, default = 0.25,
        help = 'Multiplier for beta0. [DEFAULT: 0.25]')
    parser.add_argument('-t', '--tau', type = float, default = -0.2,
        help = 'Clamping thresholding on sensitivities. [DEFAULT: -0.2]')
    parser.add_argument('-d', '--distance', type = float, default = 2.0,
        help = 'For "txt" data, the neighborhood distance threshold for computing affinities. [DEFAULT: 2.0]')
    parser.add_argument('-s', '--sigma', type = float, default = 1.0,
        help = 'For "txt" data, the standard deviation used to compute affinities. [DEFAULT: 1.0]')

    args = vars(parser.parse_args())

    # Create the affinity matrix. This is the only step that is dependent
    # on the format of the input.
    A = None
    image = None
    data = None
    if args['type'] == 'txt':
        # Comma-delimited list, one instance per line.
        data = np.loadtxt(args['input'], delimiter = ",")
        A = affinity.cartesian_affinities(data, args['distance'], args['sigma'])
    elif args['type'] == 'img':
        # Image.
        image = scipy.misc.imread(args['input'], flatten = True)
        A = affinity.image_affinities(image)
    else:
        # Mahout-style affinity data.
        data = np.loadtxt(args['mahout_data'], delimiter = ",")
        A = affinity.mahout_affinities(args['input'], args['numdims'])

    # Run the algorithm.
    Ac, n = eigencuts(A, args['eigenrank'], args['halflife'], args['epsilon'], args['tau'])

    # Perform connected component analysis on A.
    Abin = None
    if sparse.isspmatrix(Ac):
        Abin = sparse.csc_matrix(Ac).sign().todense()
    else:
        Abin = np.sign(Ac)
    numConn, connMap = csgraph.connected_components(Abin, directed = False)
    print 'Found %s clusters.' % numConn

    if args['type'] == 'img':
        plot.figure(0)    
        plot.imshow(np.reshape(connMap, newshape = image.shape), interpolation = 'nearest')
        plot.figure(1)
        plot.imshow(image, interpolation = 'nearest', cmap = cm.gray)
        plot.show()
    else:
        plot.figure(0)
        colormap = cm.get_cmap("jet", numConn)
        colorvals = colormap(np.arange(numConn))
        colors = [colorvals[connMap[i]] for i in range(0, np.size(connMap))]
        for i in range(0, data.shape[0]):
            plot.plot(data[i, 0], data[i, 1], marker = 'o', c = colors[i])
        plot.show()
