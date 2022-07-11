import torch
import numpy as np

def rot2euler(R):
    phi = np.arctan2(R[1,2], R[2,2])
    theta = -np.arcsin(R[0,2])
    psi = np.arctan2(R[0,1], R[0,0])
    return np.array([phi, theta, psi])

def euler2rot(euler):
    sin, cos = np.sin, np.cos
    phi, theta, psi = euler[0], euler[1], euler[2]
    R1 = np.array([[1, 0, 0],
                [0, cos(phi), sin(phi)],
                [0, -sin(phi), cos(phi)]])
    R2 = np.array([[cos(theta), 0, -sin(theta)],
                [0, 1, 0],
                [sin(theta), 0, cos(theta)]])
    R3 = np.array([[cos(psi), sin(psi), 0],
                [-sin(psi), cos(psi), 0],
                [0, 0, 1]])
    R = R1 @ R2 @ R3
    return R

def batch_rodrigues(theta):
    """Convert axis-angle representation to rotation matrix.
    Args:
        theta: size = [B, 3]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    l1norm = torch.norm(theta + 1e-8, p = 2, dim = 1)
    angle = torch.unsqueeze(l1norm, -1)
    normalized = torch.div(theta, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * normalized], dim = 1)
    return quat_to_rotmat(quat)

def quat_to_rotmat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [B, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = quat
    norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:,0], norm_quat[:,1], norm_quat[:,2], norm_quat[:,3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).view(B, 3, 3)
    return rotMat


def index(feat, uv, mode='bilinear'):
    '''
    :param feat: [B, C, H, W] image features
    :param uv: [B, 2, N] uv coordinates in the image plane, range [-1, 1]
    :return: [B, C, N] image features at the uv coordinates
    '''
    uv = uv.transpose(1, 2)  # [B, N, 2]
    uv = uv.unsqueeze(2)  # [B, N, 1, 2]
    # NOTE: for newer PyTorch, it seems that training results are degraded due to implementation diff in F.grid_sample
    # for old versions, simply remove the aligned_corners argument.
    # if torch.__version__ >= "1.3.0":
    #     samples = torch.nn.functional.grid_sample(feat, uv, align_corners=True)  # [B, C, N, 1]
    # else:
    samples = torch.nn.functional.grid_sample(feat, uv, mode=mode)
    return samples[:, :, :, 0]  # [B, C, N]


def orthogonal(points, calibrations, transforms=None):
    '''
    Compute the orthogonal projections of 3D points into the image plane by given projection matrix
    :param points: [B, 3, N] Tensor of 3D points
    :param calibrations: [B, 4, 4] Tensor of projection matrix
    :param transforms: [B, 2, 3] Tensor of image transform matrix
    :return: xyz: [B, 3, N] Tensor of xyz coordinates in the image plane
    '''
    rot = calibrations[:, :3, :3]
    trans = calibrations[:, :3, 3:4]
    pts = torch.baddbmm(trans, rot, points)  # [B, 3, N]
    if transforms is not None:
        scale = transforms[:2, :2]
        shift = transforms[:2, 2:3]
        pts[:, :2, :] = torch.baddbmm(shift, scale, pts[:, :2, :])
    return pts



def perspective(points, w2c, camera):
    '''
    Compute the perspective projections of 3D points into the image plane by given projection matrix
    :param points: [Bx3xN] Tensor of 3D points
    :param calibrations: [Bx4/9] Tensor of projection matrix
    :param transforms: [Bx4x4] Tensor of image transform matrix
    :return: xy: [Bx2xN] Tensor of xy coordinates in the image plane
    '''
    rot = w2c[:, :3, :3]
    trans = w2c[:, :3, 3:4]
    points = torch.baddbmm(trans, rot, points)  # [B, 3, N]
    xy  = points[:,:2, :] / torch.clamp(points[:,2:3,:], 1e-9)
    if camera.shape[1] > 6:
        x2 = xy[:,0,:]*xy[:,0,:]
        y2 = xy[:,1,:]*xy[:,1,:]
        xy_= xy[:,0,:]*xy[:,1,:]
        r2 = x2 + y2
        c = (1 + r2*(camera[:,4:5]+r2*(camera[:,5:6]+r2*camera[:,8:9])))
        xy = c.unsqueeze(1)*xy + torch.cat([ \
                  (camera[:,6:7]*2*xy_+ camera[:,7:8]*(r2+2*x2)).unsqueeze(1),\
                  (camera[:,7:8]*2*xy_+ camera[:,6:7]*(r2+2*y2)).unsqueeze(1)],1)
    xy = camera[:,0:2,None]*xy + camera[:,2:4,None]
    points[:,:2, :] = xy
    return points
