import torch
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.special import erf
from scipy.stats import gaussian_kde
from math import sqrt, exp
from tqdm import tqdm

np.seterr(divide='ignore', invalid='ignore')


def min_ade(trgt, pred):
    '''
    pred: Batch, Samples, time, [x,y]
    trgt: Batch, time, [x,y] 
    '''
    b, t, _ = trgt.shape
    dist, _ = torch.min(torch.linalg.vector_norm(pred - trgt[:, None, ...],
                                                 dim=-1).sum(dim=-1),
                        dim=-1)
    return dist.sum() / (b * t)


def min_fde(trgt, pred):
    '''
    pred: Batch, Samples, time, [x,y]
    trgt: Batch, time, [x,y] 
    '''
    b, _, _ = trgt.shape
    dist, _ = torch.min(torch.linalg.vector_norm(pred[:, :, -1, :] -
                                                 trgt[:, None, -1, :],
                                                 dim=-1),
                        dim=-1)
    return dist.sum() / b


# (12, 168, n, 2)
# (12 ,168, 2)
# Fit a GMM on the n samples try 100 and 1000. If no difference pick 100
# both on rel and absolute
# compare weighted center of the best gmm
# (12,166,1000,2)

#AMD / AMV


def calc_amd_amv(gt, pred):
    '''
    pred:   Batch, Samples, time, [x,y] --> time, batch,samples,[x,y]
    gt:     Batch, time, [x,y] --> time,batch,[x,y]
    '''
    pred = np.transpose(pred.detach().cpu().numpy(), (2, 0, 1, 3))
    gt = np.transpose(gt.detach().cpu().numpy(), (1, 0, 2))
    total = 0
    accept_count = 0
    gmm_cov_all = 0
    pbar = tqdm(total=pred.shape[0] * pred.shape[1], desc="AMD/AMV:")
    for i in range(pred.shape[0]):  #per time step
        for j in range(pred.shape[1]):
            pbar.update(1)
            try:
                gmm = get_best_gmm2(pred[i, j, :, :])
                center = np.sum(np.multiply(gmm.means_,
                                            gmm.weights_[:, np.newaxis]),
                                axis=0)
                gmm_cov = 0
                for cnt in range(len(gmm.means_)):
                    gmm_cov += gmm.weights_[cnt] * (
                        gmm.means_[cnt] - center)[..., None] @ np.transpose(
                            (gmm.means_[cnt] - center)[..., None])
                gmm_cov = np.sum(
                    gmm.weights_[..., None, None] * gmm.covariances_,
                    axis=0) + gmm_cov

                dist, _ = mahalanobis_d(
                    center, gt[i, j], len(gmm.weights_), gmm.covariances_,
                    gmm.means_, gmm.weights_
                )  #assume it will be the true value, add parameters

                total += dist
                gmm_cov_all += np.abs(np.linalg.eigvals(gmm_cov)).max()
                accept_count += 1
            except:
                continue
            # m_collect.append(dist)
    pbar.close()
    print("AMD/AMV success:",
          accept_count / (pred.shape[0] * pred.shape[1]) * 100, '%')
    # gmm_cov_all = gmm_cov_all / accept_count
    # return total / accept_count, np.abs(np.linalg.eigvals(gmm_cov_all)).max()
    return total / accept_count, gmm_cov_all / accept_count


def mahalanobis_d(x, y, n_clusters, ccov, cmeans, cluster_p):  #ccov
    v = np.array(x - y)
    Gnum = 0
    Gden = 0
    for i in range(0, n_clusters):
        ck = np.linalg.pinv(ccov[i])
        u = np.array(cmeans[i] - y)
        val = ck * cluster_p[i]
        b2 = 1 / (v.T @ ck @ v)
        a = b2 * v.T @ ck @ u
        Z = u.T @ ck @ u - b2 * (v.T @ ck @ u)**2
        pxk = sqrt(np.pi * b2 / 2) * exp(-Z / 2) * (erf(
            (1 - a) / sqrt(2 * b2)) - erf(-a / sqrt(2 * b2)))
        Gnum += val * pxk
        Gden += cluster_p[i] * pxk
    G = Gnum / Gden
    mdist = sqrt(v.T @ G @ v)
    if np.isnan(mdist):
        # print(Gnum, Gden)
        '''
        print("is nan")
        print(v)
        print("Number of clusters", n_clusters)
        print("covariances", ccov)
        '''
        return 0, 0

    # print( "Mahalanobis distance between " + str(x) + " and "+str(y) + " is "+ str(mdist) )
    return mdist, G


def get_best_gmm(X):
    lowest_bic = np.infty
    bic = []
    n_components_range = range(
        1, 7)  ## stop based on fit/small BIC change/ earlystopping
    cv_types = ['full']
    best_gmm = GaussianMixture()
    for cv_type in cv_types:
        for n_components in n_components_range:
            # Fit a Gaussian mixture with EM
            gmm = GaussianMixture(n_components=n_components,
                                  covariance_type=cv_type)
            gmm.fit(X)
            bic.append(gmm.bic(X))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm

    bic = np.array(bic)
    return best_gmm


def get_best_gmm2(X):  #early stopping gmm
    lowest_bic = np.infty
    bic = []
    cv_types = ['full']  #changed to only looking for full covariance
    best_gmm = GaussianMixture()
    for cv_type in cv_types:
        p = 1  #Decide a value
        n_comps = 1
        j = 0
        while j < p and n_comps < 5:  # if hasn't improved in p times, then stop. Do it for each cv type and take the minimum of all of them
            gmm = GaussianMixture(n_components=n_comps,
                                  covariance_type=cv_type)
            gmm.fit(X)
            bic.append(gmm.bic(X))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm
                j = 0  #reset counter
            else:  #increment counter
                j += 1
            n_comps += 1

    bic = np.array(bic)
    return best_gmm


def kde_lossf(gt, pred):
    #(12, objects, samples, 2)
    # 12, 1600,1000,2

    pred = np.transpose(pred.detach().cpu().numpy(), (2, 0, 1, 3))
    gt = np.transpose(gt.detach().cpu().numpy(), (1, 0, 2))
    kde_ll = 0
    kde_ll_f = 0
    accept_count = 0
    pbar = tqdm(total=pred.shape[0] * pred.shape[1], desc="KDE:")
    for i in range(pred.shape[0]):
        for j in range(pred.shape[1]):
            pbar.update(1)
            try:
                temp = pred[i, j, :, :]
                n_unique = len(np.unique(temp, axis=0))
                kde = gaussian_kde(pred[i, j, :, :].T)
                t = np.clip(kde.logpdf(gt[i, j, :].T), a_min=-20,
                            a_max=None)[0]
                kde_ll += t
                if i == (pred.shape[0] - 1):
                    kde_ll_f += t
                accept_count += 1
            except:
                continue
    pbar.close()
    print("KDE success:", accept_count / (pred.shape[0] * pred.shape[1]) * 100,
          '%')
    return -kde_ll / accept_count