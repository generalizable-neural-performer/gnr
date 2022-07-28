import os, sys
import numpy as np 
import cv2, imageio
from mesh import load_ply, load_obj_mesh, write_obj_mesh
import torch
from gender import genebody_gender

def image_cropping(mask, padding=0.1):
    """
    To better evaluate different metric on rendered images, we crop out the human performer and resize the cropped 
    image to the same resolution. This function provides returns the bound box of human performer given the mask.
    mask: np.ndarry of mask 
    padding: padding of the bounding box
    """
    a = np.where(mask != 0)
    h, w = list(mask.shape[:2])
    if len(a[0]) > 0:   # valid mask
        top, left, bottom, right = np.min(a[0]), np.min(a[1]), np.max(a[0]), np.max(a[1])
    else:               # mask failure
        return 0,0,mask.shape[0],mask.shape[1]
    bbox_h, bbox_w = bottom - top, right - left

    # padd bbox
    bottom = min(int(bbox_h*padding+bottom), h)
    top = max(int(top-bbox_h*padding), 0)
    right = min(int(bbox_w*padding+right), w)
    left = max(int(left-bbox_h*padding), 0)
    bbox_h, bbox_w = bottom - top, right - left
    bbox_h = min(bbox_h, h, w)
    bbox_w = min(bbox_w, h, w)

    if bbox_h >= bbox_w:
        w_c = (left+right) / 2
        size = bbox_h
        if w_c - size / 2 < 0:
            left = 0
            right = size
        elif w_c + size / 2 >= w:
            left = w - size
            right = w
        else:
            left = int(w_c - size / 2)
            right = left + size
        h_c = (top+bottom) / 2
        top = int(h_c - size / 2)
        bottom = top + size
    else:   # bbox_w >= bbox_h
        h_c = (top+bottom) / 2
        size = bbox_w
        if h_c - size / 2 < 0:
            top = 0
            bottom = size
        elif h_c + size / 2 >= h:
            top = h - size
            bottom = h
        else:
            top = int(h_c - size / 2)
            bottom = top + size
        w_c = (left+right) / 2
        left = int(w_c - size / 2)
        right = left + size

    return top, left, bottom, right

class GeneBodyReader():
    def __init__(self, rootdir, loadsize=512):
        self.rootdir = rootdir
        self.split = np.load(os.path.join(rootdir, 'genebody_split.npy'), allow_pickle=True).item()
        self.loadsize = loadsize
        # the default seting of GNR is to use these four source views of GeneBody
        self.sourceviews = ['01', '13', '25', '37']
        self.gender = genebody_gender

    def get_views(self, subject):
        """
        Returns valid camera views of each sequence. 
        Note that there are several view missing subjects in GeneBody. More specifically, 
        "Tichinah_jervier" misses [32], 
        "wuwenyan" misses [34, 36], 
        "joseph_matanda" misses [39, 40, 42, 43, 44, 45, 46, 47]

        subject: name of subject
        all_views: all valid views of this subject
        """

        all_views = sorted(os.listdir(os.path.join(self.rootdir, subject, 'image')))
        ## alt
        # all_views = sorted(np.load(os.path.join(self.rootdir, subject, 'annots.npy'), allow_pickle=True).item()['cams'].keys())

        return all_views

    def get_frames(self, subject):
        frame_list = []
        frame_list = os.listdir(os.path.join(self.rootdir, subject, 'image', '00'))
        frame_list = sorted(frame_list)
        return frame_list

    def get_cameras(self, subject):
        return np.load(os.path.join(self.rootdir, subject, 'annots.npy'), allow_pickle=True).item()['cams']

    def get_smpl(self, subject, frame_list, frame_id):
        """
        Returns the smpl vertices and faces
        frame_list: all frames of the subject <- self.get_frames(subject)
        frame_id: [0-149]
        """
        smpl_path = os.path.join(self.rootdir, subject, 'smpl', frame_list[frame_id][:-4]+'.obj')
        vert, face = load_obj_mesh(smpl_path)
        return vert, face

    def get_smpl_param(self, subject, frame_list, frame_id):
        """
        Returns the smpl parameters and smpl scale
        frame_list: all frames of the subject <- self.get_frames(subject)
        frame_id: [0-149]
        """
        param_path = os.path.join(self.rootdir, subject, 'param', frame_list[frame_id][:-4]+'.npy')
        # global_orient and pose are Rodrigues rotation vector
        param = np.load(param_path, allow_pickle=True).item()

        # the smpl_param is a dictionary of smplx parameters which can be directory passed to a SMPLX forward pass
        # via SMPLXLayer(**smpl_param) if each value of it is converted to torch.Tensor
        smpl_param = param["smplx"]
        for key in smpl_param.keys():
            if isinstance(smpl_param[key], torch.Tensor):
                smpl_param[key] = smpl_param[key].numpy()
        # For GeneBody, we fit human performer in a wide age range, and SMPLx cannot fit well on kids and giants
        # we use a smplx_scale outside SMPLX model via direct scaling.
        # You can recover the smplx mesh in 'smpl' directory via SMPLX(**smpl_param) * smpl_scale
        smpl_scale = param["smplx_scale"]

        return smpl_param, smpl_scale

    def get_data(self, subject, frame_list, all_views, camera_params, frame_id, views):
        """
        Fetch one frame of multiview data from database with cropping
        subject: name of subject
        frame_list: all frames of the subject <- self.get_frames(subject)
        all_views: all views of subject <- self.get_views(subject)
        camera_params: camera parameters <- self.get_annot(subject)
        frame_id: eg. 1
        views: list of views to fetch, eg. load sourceviews through self.sourceviews,
               or all view through self.get_views(subject)
        """
        subject_dir = os.path.join(self.rootdir, subject)
        
        Ks, c2ws, Ds, images, masks = [], [], [], [], []
        for view in views:
            img = imageio.imread(os.path.join(subject_dir, 'image', view, frame_list[frame_id]))
            msk = imageio.imread(os.path.join(subject_dir, 'mask', view, f'mask{frame_list[frame_id][:-4]}.png'))
            # crop the human out from raw image            
            top, left, bottom, right = image_cropping(msk)
            img = img * (msk > 128)[...,None]
            # resize to uniform resolution
            img = cv2.resize(img[top:bottom, left:right].copy(), (self.loadsize, self.loadsize), cv2.INTER_CUBIC)
            images.append(img)
            msk = cv2.resize(msk[top:bottom, left:right].copy(), (self.loadsize, self.loadsize), cv2.INTER_NEAREST)
            masks.append(msk)

            # adjust the camera intrinsic parameter because of the cropping and resize
            # Note that there is no need to adjust extrinsic or distortation coefficents
            K, c2w, D = camera_params[view]['K'].copy(), camera_params[view]['c2w'].copy(), camera_params[view]['D'].copy()
            K[0,2] -= left
            K[1,2] -= top
            K[0,:] *= self.loadsize / float(right - left)
            K[1,:] *= self.loadsize / float(bottom - top)
            Ks.append(K)
            c2ws.append(c2w)
            Ds.append(D)

        return images, masks, Ks, c2ws, Ds

    def get_near_far(self, verts, c2w, pad=0.5):
        """
        Get near far plane of perspective project from SMPL estimation
        verts: SMPLx vertices
        c2w: Camera to world roation matrix
        pad: near far padding from SMPLx near far, set smaller if you want tighter bound, set larger if the accessory is huge.
        """
        w2c = np.linalg.inv(c2w)
        # Transform SMPLx to camera coordinate
        vp = verts.dot(w2c[:3,:3].T) + w2c[:3,3:].T
        vmin, vmax = vp.min(0), vp.max(0)
        # near far are minmax in z axis
        near, far = vmin[2], vmax[2]
        near, far = near-(far-near)*pad, far+(far-near)*pad
        return near, far

    def smpl_from_param(self, model_path, subject, smpl_param, smpl_scale):
        import smplx
        smpl = smplx.SMPLX(
            model_path=model_path,
            gender=self.gender[subject],
            use_pca=False,
        )
        smpl_param = smpl_param.copy()
        for key in smpl_param.keys():
            if isinstance(smpl_param[key], np.ndarray):
                smpl_param[key] = torch.from_numpy(smpl_param[key])
        output = smpl(
            **smpl_param,
            return_full_pose=True,
            use_hands=False,
            use_feet_keypoints=False,
        )
        verts = output['vertices'].numpy().reshape(-1,3) * smpl_scale

        # To align with keypoints3d saved in param, use the base keypoints only,
        # if you want to use the full keypoints3d with extra joints and landmarks,
        # please refer the the definition of joints in
        # vertex_joint_selector.py and landmarks in vertices2landmarks in lbs.py
        keypoints3d = output['joints'].numpy().reshape(-1,3)[:55] * smpl_scale
        
        return verts, smpl.faces, keypoints3d


if __name__ == "__main__":
    ## Here is a example
    # python genebody/genebody.py path_to_genebody fuzhizhi
    root = sys.argv[1]
    subject = sys.argv[2]

    genebody = GeneBodyReader(root)
    print(subject, ' is a ', 'training set' if subject in genebody.split['train'] else 'test set')

    views = genebody.get_views(subject)
    frames = genebody.get_frames(subject)
    camera_params = genebody.get_cameras(subject)

    imgs, msks, Ks, c2ws, Ds = genebody.get_data(subject, frames, views, camera_params, 0, genebody.sourceviews)
    print(f'loaded {len(imgs)} frames of images and masks in size of ', list(imgs[0].shape))

    verts, faces = genebody.get_smpl(subject, frames, 0)
    smpl_param, smpl_scale = genebody.get_smpl_param(subject, frames, 0)
    print('mesh with size ', verts.shape, ' body scale ', smpl_scale)

    near, far = genebody.get_near_far(verts, c2ws[0])
    print('the near far is ', near, far)

    # to test smplx parameter tor smplx mesh, please try the following command
    # python genebody/genebody.py path_to_genebody fuzhizhi path_to_smplx
    if len(sys.argv) == 4:
        import trimesh
        smplx_path = sys.argv[3]
        verts_from_param, faces_from_param, kpts_from_param = genebody.smpl_from_param(smplx_path, subject, smpl_param, smpl_scale)
        print('average error of parameter generated smplx is ', np.abs(verts_from_param-verts).mean())
        print('average error of parameter generated keypoints is ', np.abs(smpl_param['keypoints3d'].reshape(-1,3)-kpts_from_param).mean())
        smpl_mesh = trimesh.Trimesh(verts_from_param, faces_from_param)
        smpl_mesh.export(f'{subject}.obj')

