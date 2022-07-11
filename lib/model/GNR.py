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