import sys
import os
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path = sys.path[:-1]
import time
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

from lib.options import BaseOptions
from lib.data.GeneBodyDataset import GeneBodyDataset as MyDataset
from lib.mesh_util import *
from lib.net_ddp import create_network, worker_init_fn, ddpSampler, ddp_init, synchronize
from lib.model import GNR
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm, trange
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
import logging
import imageio
import lib.metrics_torch as metrics_torch

def loss_string(loss_dict):
    string = ''
    for key in loss_dict.keys():
        string += '| {}: {:.2e} '.format(key, loss_dict[key].item())
    return string

def print_write(file, string):
    file.write(string)
    if string[-1] == '\n': string = string[:-1]
    print(string)

def to8b(img):
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    if img.shape[0] == 3 and img.shape[-1] != 3:
        img = np.transpose(img, [1,2,0])
    if img.min() < -.2:
        img = (img + 1) * 127.5
    elif img.max() <= 2.:
        img = img * 255.
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)
# get options

def prepare_data(opt, data, local_rank=0):
    # retrieve the data
    image_tensor = data['img'][0].to(device=local_rank)
    calib_tensor = data['calib'][0].to(device=local_rank)
    mask_tensor = data['mask'][0].to(device=local_rank)
    bbox = list(data['bbox'][0].numpy().astype(np.int32))
    mesh_param = {'center': data['center'][0].to(device=local_rank), 
                    'body_scale': data['body_scale'][0].cpu().numpy().item()}
    if opt.train_shape:
        mesh_param['samples'] = data['samples'][0].to(device=local_rank)
        mesh_param['labels'] = data['labels'][0].to(device=local_rank)

    if any([opt.use_smpl_sdf, opt.use_t_pose]):
        smpl = { 'rot': data['smpl_rot'].to(device=local_rank) }
        if opt.use_smpl_sdf or opt.use_t_pose:
            smpl['verts'] = data['smpl_verts'][0].to(device=local_rank)
            smpl['faces'] = data['smpl_faces'][0].to(device=local_rank)
        if opt.use_t_pose:
            smpl['t_verts'] = data['smpl_t_verts'][0].to(device=local_rank)
            smpl['t_faces'] = data['smpl_t_faces'][0].to(device=local_rank)
        if opt.use_smpl_depth:
            smpl['depth'] = data['smpl_depth'][0].to(device=local_rank)[:,None,...]
            
    else:
        smpl = None

    if 'scan_verts' in data.keys():
        scan = [data['scan_verts'][0].to(device=local_rank), data['scan_faces'][0].to(device=local_rank)]
    else:
        scan = None

    persps = data['persps'][0].to(device=local_rank) if opt.projection_mode == 'perspective' else None
    
    return {
        'images': image_tensor,
        'calibs': calib_tensor,
        'bbox': bbox,
        'masks': mask_tensor,
        'mesh_param': mesh_param,
        'smpl': smpl,
        'scan': scan,
        'persps': persps
    }

def cal_metrics(metrics, rgbs, gts):
    x = rgbs.clone().permute((0, 3, 1, 2))
    out = {}
    for m_key in metrics.keys():
        out[m_key] = []
        for pred, gt in zip(x, gts):
            metric = metrics[m_key]
            out[m_key].append(metric(pred, gt))
        out[m_key] = torch.stack(out[m_key], dim=0)
    return out

def train(opt, rank=0, local_rank = 0):
    gpu_num = torch.cuda.device_count()
    train_dataset = MyDataset(opt, phase='train')
    test_dataset = MyDataset(opt, phase='test', move_cam=0)
    render_dataset = MyDataset(opt, phase='render', move_cam=opt.move_cam)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if opt.ddp else None

    # create data loader
    shuffle = not opt.ddp
    train_data_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=opt.batch_size, 
                                    num_workers=opt.num_threads, shuffle=False, worker_init_fn=worker_init_fn)
    logging.info(f'train data size: {len(train_data_loader)}')

    test_data_loader = DataLoader(test_dataset, batch_size=1)
    test_data_iter = iter(test_data_loader)
    logging.info(f'test data size: {len(test_data_loader)}')

    render_data_loader = DataLoader(render_dataset, batch_size=1)
    render_data_iter = iter(render_data_loader)
    logging.info(f'render data size: {len(render_data_loader)}')
    # create net
    net = GNR(opt)
    
    logging.info(f'Using Network: {net.name}')
    
    set_train = net.train
    set_eval = net.eval

    os.makedirs(opt.basedir, exist_ok=True)
    os.makedirs('%s/%s' % (opt.basedir, opt.name), exist_ok=True)

    net, start_epoch = create_network(opt, net, local_rank)
    global_step = start_epoch * len(train_dataset)

    lr = opt.lrate * (0.1 ** (start_epoch / opt.lrate_decay))
    # params = net.parameters() if opt.train_encoder else net.module.nerf.parameters()
    params_list = []
    for name, param in net.named_parameters():
        if 'occ_linears' in name:
            if opt.train_occlusion:
                params_list.append(param)
        elif 'image_filter' in name:
            if opt.train_encoder:
                params_list.append(param)
        else:
            params_list.append(param)

    optimizer = torch.optim.Adam(params=params_list, lr=lr, betas=(0.9, 0.999))

    is_summary = not opt.ddp or (opt.ddp and (rank == 0))
    if is_summary:
        from tqdm import tqdm, trange
        if opt.train:
            writer = SummaryWriter(os.path.join(opt.basedir, opt.name))
            opt_log = os.path.join(opt.basedir, opt.name, 'opt.txt')
            config_file = os.path.join(opt.basedir, opt.name, 'config.txt')
            with open(opt_log, 'w') as outfile:
                outfile.write(json.dumps(vars(opt), indent=2))
            os.system(f'cp {opt.config} {config_file}')
    else:
        tqdm = lambda x: x
        trange = range

    # evaluate, not demo
    # metrics_dict = {'lpips': [], 'psnr': [], 'ssim': []}
    metrics_dict = {}
    metrics = {'lpips': metrics_torch.LPIPS().to(local_rank), 'psnr': metrics_torch.psnr, 'ssim': metrics_torch.SSIM().to(local_rank)}
    
    # training
    if opt.train:
        for epoch in trange(start_epoch, opt.num_epoch):        

            set_train()
            if opt.ddp:
                train_data_loader.sampler.set_epoch(epoch)
                synchronize()
            
            pbar = tqdm(train_data_loader)
            if is_summary:
                pbar.set_description("epoch {}/{}".format(epoch, opt.num_epoch))

            for train_idx, train_data in enumerate(pbar):
                data = prepare_data(opt, train_data, local_rank)
                train_shape = opt.train_shape and train_idx % opt.train_shape_skips == 0
                loss_dict = net(data, train_shape=train_shape)
                loss = sum(loss_dict.values())

                optimizer.zero_grad()
                try:
                    loss.backward()
                except:
                    print(train_data['name'], train_data['sid'], train_data['vid'], flush=True)
                optimizer.step()

                if global_step % opt.freq_plot == 0 and is_summary:
                    tqdm.write(
                        '[{}] | epoch: {} | step: {:d} | loss:{:.2e} | lr: {:.2e} '.format(
                            opt.name, epoch, global_step, loss.item(), lr))
                    tqdm.write(f'[{opt.name}] {loss_string(loss_dict)}')

                if is_summary:
                    writer.add_scalar('loss', loss.item(), global_step)
                    for key in loss_dict.keys():
                        writer.add_scalar(key, loss_dict[key].item(), global_step)
                    pbar.update(1)
                global_step += 1 if not opt.ddp else gpu_num

            if opt.ddp and (rank == 0):
                torch.save({'network_state_dict': net.module.state_dict(), 'epoch': epoch+1}, 
                            '%s/%s/%04d.tar' % (opt.basedir, opt.name, epoch+1))
            elif not opt.ddp:
                torch.save({'network_state_dict': net.state_dict(), 'epoch': epoch+1}, 
                            '%s/%s/%04d.tar' % (opt.basedir, opt.name, epoch+1))

            # update learning rate
            lr = opt.lrate * (0.1 ** ((epoch+1) / opt.lrate_decay))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

    if opt.test:
        net.eval()
        idx = 0
        with torch.no_grad():
            for test_idx in trange(len(test_dataset)):
                ## test dataset
                test_data = next(test_data_iter)
                data = prepare_data(opt, test_data, local_rank)
                name = test_data['name'][0]
                subject = name.split('_')[0]
                 # fid = test_data['fid'][0]
                fid = test_data['sid'][0]
                vid = test_data['vid'][0]
                render_gt = test_data['render_gt'][0].to(local_rank)
                
                if opt.ddp:
                    # distribute query views to different GPUs if multiple GPU or multiple machines are available
                    src_calibs, tar_calibs = torch.split(data['calibs'], [opt.num_views, data['calibs'].shape[0]-opt.num_views], 0)
                    total_len = len(tar_calibs)
                    sampler = ddpSampler(tar_calibs)
                    indices = sampler.indices()
                    tar_calibs = tar_calibs[indices]
                    tar_gt = render_gt[indices]
                    data['calibs'] = torch.cat([src_calibs, tar_calibs], 0)
                    if opt.projection_mode == 'perspective':
                        src_persps, tar_persps = torch.split(data['persps'], [opt.num_views, data['persps'].shape[0]-opt.num_views], 0)
                        tar_persps = tar_persps[indices]
                        data['persps'] = torch.cat([src_persps, tar_persps], 0)

                rgbs, _ = net.module.render_path(data)
                rgbs = sampler.distributed_concat(rgbs, total_len) if opt.ddp else rgbs
                if opt.use_attention:
                    rgbs, att_rgbs = rgbs[...,:3], rgbs[...,3:6]
                else:
                    att_rgbs = rgbs[..., :3]
                m_dict = cal_metrics(metrics, att_rgbs, render_gt)
                
                if subject not in metrics_dict.keys():
                    metrics_dict[subject] = {'lpips': [], 'psnr': [], 'ssim': []}
                for key, value in m_dict.items():
                    # if opt.ddp:
                    #     value = sampler.distributed_concat(value, total_len) if opt.ddp else value
                    metrics_dict[subject][key].append(torch.mean(value).cpu().numpy())

                att_rgbs = [to8b(att_rgb) for att_rgb in att_rgbs.cpu().numpy()]
                render_gt = [to8b(gt) for gt in render_gt.permute(0,2,3,1).cpu().numpy()]
                target_dir = os.path.join(opt.basedir, opt.name, opt.eval_dir, name)
                os.makedirs(target_dir, exist_ok=True)
                if is_summary:
                    for vid, im in enumerate(att_rgbs):
                        imageio.imwrite(os.path.join(target_dir, f'{vid:02d}.png'), im)

                    fname = os.path.join(opt.basedir, opt.name, opt.eval_dir, 'eval.txt')
                    with open(fname, 'a') as file_:
                        print_write(file_, '******\n%s\n' % (name))
                        for k, v in metrics_dict[subject].items():
                            print_write(file_, '%s: %.5f\n'%(k, v[-1]))
                        file_.write('------')
                        for k, v in metrics_dict[subject].items():
                            print_write(file_, '[total] %s: %.5f\n'%(k, sum(v)/len(v)))

                if opt.output_mesh:
                    verts, faces, rgbs = net.module.reconstruct(data)
                    if is_summary:
                        save_obj_mesh_with_color(os.path.join(target_dir, "{}.obj".format(name)), verts, faces, rgbs)
                idx += 1

    if opt.render:
        net.eval()
        with torch.no_grad():
            imgs = []
            for ridx in trange(len(render_dataset)):
                # render dataset
                test_data = next(render_data_iter)
                data = prepare_data(opt, test_data, local_rank)
                name = test_data['name'][0]
                # fid = test_data['fid'][0]
                fid = test_data['sid'][0]
                vid = test_data['vid'][0]
                
                if opt.ddp:
                    # distribute query views to different GPUs if multiple GPU or multiple machines are available
                    src_calibs, tar_calibs = torch.split(data['calibs'], [opt.num_views, data['calibs'].shape[0]-opt.num_views], 0)
                    total_len = len(tar_calibs)
                    sampler = ddpSampler(tar_calibs)
                    indices = sampler.indices()
                    tar_calibs = tar_calibs[indices]
                    data['calibs'] = torch.cat([src_calibs, tar_calibs], 0)
                    if opt.projection_mode == 'perspective':
                        src_persps, tar_persps = torch.split(data['persps'], [opt.num_views, data['persps'].shape[0]-opt.num_views], 0)
                        tar_persps = tar_persps[indices]
                        data['persps'] = torch.cat([src_persps, tar_persps], 0)

                target_dir = os.path.join(opt.basedir, opt.name, opt.render_dir, name.split('_')[0])
                os.makedirs(target_dir, exist_ok=True)
                rgbs, depths = net.module.render_path(data)
                rgbs = sampler.distributed_concat(rgbs, total_len) if opt.ddp else rgbs
                depths = sampler.distributed_concat(depths, total_len) if opt.ddp else depths
                if opt.use_attention:
                    rgbs, att_rgbs = rgbs[...,:3], rgbs[...,3:6]
                else:
                    att_rgbs = rgbs[..., :3]
                att_rgbs = [to8b(att_rgb) for att_rgb in att_rgbs.cpu().numpy()]
                imgs += att_rgbs
                os.makedirs(target_dir, exist_ok=True)
                for vid, im in enumerate(att_rgbs):
                    imageio.imwrite(os.path.join(target_dir, f'{ridx:03d}_rgb.png'), im)
                    depth= np.clip(np.round(depths[vid].cpu().numpy()*1000), 0, 65535).astype(np.uint16)
                    depth[depth == 0] = 65535
                    imageio.imwrite(os.path.join(target_dir, f'{ridx:03d}_depth.png'), depth)
            if is_summary:
                imageio.mimwrite(os.path.join(target_dir, "render_{}.mp4".format(fid)), imgs, quality=8, fps=30)

            

if __name__ == '__main__':
    opt = BaseOptions().parse()
    if opt.ddp:
        rank, local_rank = ddp_init(opt)
        logging.basicConfig(level=logging.INFO if rank in [-1, 0] else logging.WARN)
        logging.info(vars(opt))
        train(opt, rank, local_rank)
    else:
        logging.basicConfig(level=logging.INFO)
        logging.info(vars(opt))
        train(opt)

