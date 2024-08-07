import numpy as np
from copy import deepcopy
import networkx as nx
from tqdm import trange
import scipy as sp
from scipy.sparse.csgraph import laplacian
from scipy.stats import spearmanr
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lobpcg
import topcorr
import os


def align_data(data_dict, samplef, t):
    align_t = np.arange(0, t, samplef)
    aligned_dict = {}

    # Check for missing data
    for s, data in data_dict.items():
        if data.size == 0 or data.shape[0] == 3600:
            continue

        norm_to_start = data[:, 0] - data[0, 0]  # TODO: replace data[0,0] with ensemble start when implemented
        align_mapping = np.zeros(align_t.size)
        for ts in align_t:
            exists_within_tolerance = \
            np.where((norm_to_start >= ts - samplef / 2) & (norm_to_start <= ts + samplef / 2))[0]
            if len(exists_within_tolerance) > 0:
                align_mapping[int(ts / samplef)] = int(exists_within_tolerance[0])

        mask = align_mapping > 0
        mask[0] = True
        aligned_data = np.column_stack((align_t, np.zeros(align_t.size)))
        aligned_data[mask, 1] = data[align_mapping[mask].astype(int), 1]
        aligned_dict[s] = deepcopy(aligned_data)

    return aligned_dict


def win_shape(w, w_shape):
    if w_shape == "square":
        return np.ones((w, 1)) / w

    elif w_shape == "tapered":
        theta = np.round(w / 3)
        w0 = (1 - np.exp(-1 / theta)) / (1 - np.exp(-w / theta))
        return ((w0 * np.exp((np.array(range(-w + 1, 1)) / theta))).T).reshape(w, 1)


def weightedcorrs(X, w):
    dt, N = np.shape(X)
    temp = X - np.tile(np.dot(w.T, X), (dt, 1))
    temp = np.dot(temp.T, (temp * np.tile(w, (1, N))))
    temp = 0.5 * (temp + temp.T)
    R = np.diag(temp)
    R = R.reshape(len(R), 1)
    R = temp / np.sqrt(np.dot(R, R.T))
    return R


def rolling_window(X, w_size, w_shape, step, thr=0, type_thr=False, corr="pearsons"):
    data_len = np.shape(X)[0]
    r_dict = {}

    for w in range(0, int(data_len - (w_size))):
        if corr == "pearsons":
            r_dict[w] = weightedcorrs(X[w:w + w_size, :], win_shape(w_size, w_shape))

        elif corr == "spearmans":
            corr_matrix = np.zeros((X.shape[1], X.shape[1]))
            for i in range(X.shape[1]):
                for j in range(i + 1, X.shape[1]):
                    rho, _ = spearmanr(X[w:w + w_size, i], X[w:w + w_size, j])
                    corr_matrix[i, j] = rho
                    corr_matrix[j, i] = rho
            np.fill_diagonal(corr_matrix, 1)
            r_dict[w] = corr_matrix

        elif corr == "dcca":
            r_dict[w] = topcorr.dcca(X[w:w + w_size, :], 25)

        r_dict[w][np.where(np.isnan(r_dict[w]) == True)] = 0
        np.fill_diagonal(r_dict[w], 1)

        ## Thresholding
        if type_thr == False:
            r_dict[w][np.where((r_dict[w] < thr) & (r_dict[w] > -thr))] = 0
        elif type_thr == "neg":
            r_dict[w][np.where(r_dict[w] > -thr)] = 0
        elif type_thr == "pos":
            r_dict[w][np.where(r_dict[w] < thr)] = 0
        elif type_thr == "proportional":
            r_dict[w] = proportional_thr(r_dict[w], thr)
        elif type_thr == "pmfg":
            if thr != 0:
                r_dict[w][np.where(r_dict[w] > -thr)] = 0
            r_dict[w] = nx.to_numpy_array(topcorr.pmfg(r_dict[w]))
        elif type_thr == "tmfg":
            r_dict[w] = nx.to_numpy_array(topcorr.tmfg(r_dict[w], absolute=True))
    return r_dict


def claplacian(M, norm=True):
    if norm == True:
        L = np.diag(sum(M)) - M
        v = 1. / np.sqrt(sum(M))
        return np.diag(v) * L * np.diag(v)
    else:
        return np.diag(sum(M)) - M


def eigenspectrum(L):
    eigvals = np.real(np.linalg.eig(L)[0])
    return -np.sort(-eigvals)


def all_spectra_lobpcg(A_dict, norm=True):
    dict_keys = list(A_dict.keys())
    eigenspectrums = np.zeros((9, len(dict_keys)))
    i = 0
    for key in dict_keys:
        L = laplacian(A_dict[key], norm)
        L_sparse = csr_matrix(L)
        X = np.random.normal(size=(L_sparse.shape[0], 9))
        eigenspectrums[:, i], _ = lobpcg(L_sparse, X, largest=False)
        i += 1
    return eigenspectrums


def all_spectra(A_dict, norm=True):
    dict_keys = list(A_dict.keys())
    eigenspectrums = np.zeros((np.shape(A_dict[dict_keys[0]])[0], len(dict_keys)))
    i = 0
    for key in dict_keys:
        L = laplacian(A_dict[key], norm)
        eigenspectrums[:, i] = eigenspectrum(L)
        i += 1
    return eigenspectrums


def all_spectra_signed(A_dict, norm=True):
    dict_keys = list(A_dict.keys())
    eigenspectrums = np.zeros((np.shape(A_dict[dict_keys[0]])[0], len(dict_keys)))
    i=0
    for r in A_dict:
        R = A_dict[r]
        A_pos = np.maximum(0, R)
        A_neg = -np.minimum(0, R)

        D_pos = np.diag(np.sum(A_pos, axis=1))
        D_neg = np.diag(np.sum(A_neg, axis=1))

        D_pos_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D_pos)))
        D_neg_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D_neg)))

        D_pos_inv_sqrt[np.isinf(D_pos_inv_sqrt)] = 0
        D_neg_inv_sqrt[np.isinf(D_neg_inv_sqrt)] = 0

        L_pos = np.eye(R.shape[0]) - np.dot(np.dot(D_pos_inv_sqrt, A_pos), D_pos_inv_sqrt)
        L_neg = np.eye(R.shape[0]) - np.dot(np.dot(D_neg_inv_sqrt, A_neg), D_neg_inv_sqrt)

        L_signed = L_pos + L_neg

        eigenvalues, eigenvectors = np.linalg.eigh(L_signed)

        eigenspectrums[:,i] = eigenvalues

        i+=1

    return eigenspectrums

def snapshot_dist(eigenspectrums, norm=True):
    N = np.shape(eigenspectrums)[1]
    dist = np.zeros((N, N))
    for i in trange(N):
        for j in range(N):
            dist[i, j] = np.sqrt(np.sum(np.power((eigenspectrums[:, i] - eigenspectrums[:, j]), 2)))
            if norm == True:
                if max(max(eigenspectrums[:, i]), max(eigenspectrums[:, j])) > 1e-10:
                    dist[i, j] = dist[i, j] / np.sqrt(
                        max((np.sum(np.power(eigenspectrums[:, i], 2))), (np.sum(np.power(eigenspectrums[:, j], 2)))))
                else:
                    dist[i, j] = 0

    return dist


def landmark_snapshot_dist(eigenspectrums, lm_inds, norm=True):
    N = np.shape(eigenspectrums)[1]
    dist = np.zeros((N, len(lm_inds)))
    for i in trange(N):
        for j in range(len(lm_inds)):
            k = lm_inds[j]
            dist[i, j] = np.sqrt(np.sum(np.power((eigenspectrums[:, i] - eigenspectrums[:, k]), 2)))
            if norm == True:
                if max(max(eigenspectrums[:, i]), max(eigenspectrums[:, k])) > 1e-10:
                    dist[i, j] = dist[i, j] / np.sqrt(
                        max((np.sum(np.power(eigenspectrums[:, i], 2))), (np.sum(np.power(eigenspectrums[:, k], 2)))))
                else:
                    dist[i, j] = 0
    return dist


def LMDS(D, lands, dim):
    Dl = D[:, lands]
    n = len(Dl)

    # Centering matrix
    H = - np.ones((n, n)) / n
    np.fill_diagonal(H, 1 - 1 / n)
    # YY^T
    H = -H.dot(Dl ** 2).dot(H) / 2

    # Diagonalize
    evals, evecs = np.linalg.eigh(H)

    # Sort by eigenvalue in descending order
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]

    # Compute the coordinates using positive-eigenvalued components only
    w, = np.where(evals > 0)
    if dim:
        arr = evals
        w = arr.argsort()[-dim:][::-1]
        if np.any(evals[w] < 0):
            print('Error: Not enough positive eigenvalues for the selected dim.')
            return []
    if w.size == 0:
        print('Error: matrix is negative definite.')
        return []

    V = evecs[:, w]
    N = D.shape[1]
    Lh = V.dot(np.diag(1. / np.sqrt(evals[w]))).T
    Dm = D - np.tile(np.mean(Dl, axis=1), (N, 1)).T
    dim = w.size
    X = -Lh.dot(Dm) / 2.
    X -= np.tile(np.mean(X, axis=1), (N, 1)).T

    _, evecs = sp.linalg.eigh(X.dot(X.T))

    return (evecs[:, ::-1].T.dot(X)).T


def r_thr(r, thr, signed="pos"):
    if signed == False:
        r[np.where((r < thr) & (r > -thr))] = 0
    elif signed == "neg":
        r[np.where(r > -thr)] = 0
    else:
        r[np.where(r < thr)] = 0
    return r


def proportional_thr(r, p):
    thr_r = np.zeros((r.shape))
    r[np.where((-0.05 < r) & (r < 0.05))] = 0
    ut = np.triu(r, k=1)
    n = ((len(r) ** 2) / 2) - len(r)
    n = int(n * p)
    if len(np.where((0 < ut) | (ut < 0))[0]) < n:
        return r
    else:
        elem = [[x, y] for x, y in zip(np.where((0 < ut) | (ut < 0))[0], np.where((0 < ut) | (ut < 0))[1])]
        vals = ut[np.where((0 < ut) | (ut < 0))]
        vals = abs(vals)
        ind = np.argpartition(vals, -n)[-n:]
        for i in ind:
            thr_r[elem[i][0], elem[i][1]] = r[elem[i][0], elem[i][1]]
            thr_r[elem[i][1], elem[i][0]] = r[elem[i][0], elem[i][1]]
        np.fill_diagonal(thr_r, 1)
        return thr_r


def pmfg(corr_mat):
    """Construct a Planar Maximally Filtered Graph from a correlation matrix."""
    corr_mat[np.where(np.isnan(corr_mat) == True)] = 0
    n = corr_mat.shape[0]
    edges = [(corr_mat[i, j], i, j) for i in range(n) for j in range(i + 1, n) if corr_mat[i, j] != 0]
    # Sort edges based on weight in descending order
    edges.sort(reverse=True, key=lambda x: x[0])

    G = nx.Graph()
    G.add_nodes_from(list(range(n)))

    for _, i, j in edges:
        G.add_edge(i, j)
        if not nx.check_planarity(G)[0]:  # Check if the graph remains planar
            G.remove_edge(i, j)  # Remove the edge if adding it violates planarity

    return nx.to_numpy_array(G)


def sub_sample_zeros(X):
    zeros = list(np.where(X == 0)[0][1:])
    ind = 0
    while ind != len(zeros):
        intr = []
        curr = zeros[ind]
        intr.append(zeros[ind])
        ind += 1
        if ind != len(zeros):
            while zeros[ind] == curr:
                ind += 1
                curr = zeros[ind]
                intr.append(zeros[ind])
        if intr[-1] + 1 < X.shape[0]:
            st_val = X[intr[0] - 1]
            nd_val = X[intr[-1] + 1]
            n_val = len(intr)
            step = (nd_val - st_val) / (n_val + 1)
            for i in range(intr[0], intr[-1] + 1):
                X[i] = X[i - 1] + step
    return X


def overlap_ts(X, win, step):
    chunks = []
    y = []
    win_st = 0
    win_nd = win_st+win
    win_ind = 0
    while win_nd <= len(X):
        chunks.append(X[win_st:win_nd,:])
        y.extend([win_ind]*win)
        win_st+=step
        win_nd+=step
        win_ind+=1
    Xt = np.concatenate(chunks)
    return Xt, y


def export_mats_to_bins(r_dict, dir):
    os.mkdir(dir)

    for name, matrix in r_dict.items():
        flatten_matrix = matrix.flatten()

        with open(f'{dir}/{dir}-{name}.bin', 'wb') as f:
            flatten_matrix.tofile(f)


def import_par_tmfg_res(dir, n_files, n_nodes):
    r_dict = {}
    for n in range(n_files):
        filename = dir+f"-{n}-exact-P-1"
        mat = np.zeros((n_nodes,n_nodes))
        with open(filename, "r") as f:
            for line in f:
                i, j, value = line.split()
                i, j = int(i), int(j)
                value = float(value)
                mat[i, j] = value

        r_dict[n] = deepcopy(mat)

    return r_dict