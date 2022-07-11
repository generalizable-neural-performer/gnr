import sys
import os

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# import torch
# torch.autograd.set_detect_anomaly(True)
# import torch.nn as nn
# import torch.nn.functional as F

import numpy as np
import trimesh
import cv2
from scipy.spatial import KDTree
import lpips

lpips_net = lpips.LPIPS(net='alex')
def chamfer(x_verts, gt_verts, x_normals=None, gt_normals=None):

    searcher = KDTree(gt_verts)
    dists, inds = searcher.query(x_verts)

    if x_normals is None or gt_normals is None:
        return dists
    elif x_normals is not None and gt_normals is not None:
        cosine_dists = 1 - np.sum(x_normals * gt_normals[inds], axis=1)
        return dists, cosine_dists
    else:
        raise Exception("provide normals for both point sets")



def fscore(dist1, dist2, threshold=1.0):
    """
    Calculates the F-score between two point clouds with the corresponding threshold value.
    :param dist1: N-Points
    :param dist2: N-Points
    :param th: float
    :return: fscore, precision, recall
    """
    # NB : In this depo, dist1 and dist2 are squared pointcloud euclidean distances, so you should adapt the threshold accordingly.
    precision_1 = np.mean((dist1 < threshold).astype(np.float32))
    precision_2 = np.mean((dist2 < threshold).astype(np.float32))
    fscore = 2 * precision_1 * precision_2 / (precision_1 + precision_2)
    return fscore, precision_1, precision_2


def psnr(x, gt):
    """
    x: np.uint8, HxWxC, 0 - 255
    gt: np.uint8, HxWxC, 0 - 255
    """
    x = (x / 255.).astype(np.float32)
    gt = (gt / 255.).astype(np.float32)
    mse = ((x - gt) ** 2).mean()
    psnr = 10. * np.log10(1. / mse)

    # return mse, psnr
    return psnr

def ssim_channel(x, gt):

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    x = x.astype(np.float32)
    gt = gt.astype(np.float32)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(x, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(gt, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(x ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(gt ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(x * gt, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def ssim(x, gt):
    '''calculate SSIM
    the same outputs as MATLAB's
    x: np.uint8, HxWxC, 0 - 255
    gt: np.uint8, HxWxC, 0 - 255
    '''
    if not x.shape == gt.shape:
        raise ValueError('Input images must have the same dimensions.')
    if x.ndim == 2:
        return ssim_channel(x, gt)
    elif x.ndim == 3:
        if x.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim_channel(x, gt))
            return np.array(ssims).mean()
        elif x.shape[2] == 1:
            return ssim_channel(np.squeeze(x), np.squeeze(gt))
    else:
        raise ValueError('input image dimension mismatch.')


def lpips(x, gt, net=lpips_net):
    x = torch.from_numpy(x).float() / 255. * 2 - 1.
    gt = torch.from_numpy(gt).float() / 255. * 2 - 1.
    x = x.permute([2, 0, 1]).unsqueeze(0)
    gt = gt.permute([2, 0, 1]).unsqueeze(0)
    with torch.no_grad():
        loss = net.forward(x, gt)
    return loss.item()




if __name__ == "__main__":
    import trimesh
    x = trimesh.load("./data/human/SMPL/10315_m_John/smpl.obj", process=False)
    y = trimesh.load("./data/human/SMPL/10316_m_John/smpl.obj", process=False)

    print(chamfer(x.vertices, y.vertices, x.vertex_normals, y.vertex_normals))
