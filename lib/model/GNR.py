import torch
import torch.nn as nn
import torch.nn.functional as F
from .HGFilters import *
from ..net_util import init_weights
from .NeRF import NeRF
from .NeRFRenderer import NeRFRenderer
from .SRFilters import SRFilters
from ..geometry import index


class GNR(nn.Module):
    '''
    HG PIFu network uses Hourglass stacks as the image filter.
    It does the following:
        1. Compute image feature stacks and store it in self.im_feat_list
            self.im_feat_list[-1] is the last stack (output stack)
        2. Calculate calibration
        3. If training, it index on every intermediate stacks,
            If testing, it index on the last stack.
        4. Classification.
        5. During training, error is calculated on all stacks.
    '''

    def __init__(self, opt):

        super(GNR, self).__init__()
        self.name = 'gnr'

        self.opt = opt
        self.num_views = self.opt.num_views
        self.use_feat_sr = self.opt.use_feat_sr
        self.ddp = self.opt.ddp
        self.feat_dim = 64 if self.use_feat_sr else 256
        self.index = index
        self.error_term=nn.MSELoss()

        self.image_filter = HGFilter(opt)
        if self.use_feat_sr:
            self.sr_filter = SRFilters(order=2, out_ch=self.feat_dim)

        if not opt.train_encoder:
            for param in self.image_filter.parameters():
                param.requires_grad = False

        self.nerf = NeRF(opt, input_ch_feat=self.feat_dim)
        self.nerf_renderer = NeRFRenderer(opt, self.nerf)

        init_weights(self)
    
    def image_rescale(self, images, masks):
        if images.min() < -0.2:
            images = (images + 1) / 2
            images = images * (masks > 0).float()
        return images

    def get_image_feature(self, data):
        if 'feats' not in data.keys():
            images = data['images']
            im_feat = self.image_filter(images[:self.num_views])
            if self.use_feat_sr:
                im_feat = self.sr_filter(im_feat, images[:self.num_views])
            data['images'] = torch.cat([self.image_rescale(images[:self.num_views], data['masks'][:self.num_views]), \
                                        images[self.num_views:]], 0)
            data['feats'] = im_feat
        return data

    def forward(self, data, train_shape=False):
        '''
        Given 3D points, query the network predictions for each point.
        Image features should be pre-computed before this call.
        store all intermediate features.
        query() function may behave differently during training/testing.
        :param points: [B, 3, N] world space coordinates of points
        :param calibs: [B, 3, 4] calibration matrices for each image
        :param transforms: Optional [B, 2, 3] image space coordinate transforms
        :param labels: Optional [B, Res, N] gt labeling
        :return: [B, Res, N] predictions for each point
        '''
        data = self.get_image_feature(data)
        if train_shape:
            error = self.nerf_renderer.train_shape(**data)
        else:
            error = self.nerf_renderer.render(**data)

        return error

    def render_path(self, data):
        with torch.no_grad():
            rgbs = None
            data = self.get_image_feature(data)
            rgbs, depths = self.nerf_renderer.render_path(**data)

        return rgbs, depths

    def reconstruct(self, data):
        with torch.no_grad():
            data = self.get_image_feature(data)
            verts, faces, rgbs = self.nerf_renderer.reconstruct(**data)

        return verts, faces, rgbs