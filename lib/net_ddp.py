import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import logging
import os
import random
import numpy as np
import math

def worker_init_fn(worker_id):
    random.seed(worker_id+100)
    np.random.seed(worker_id+100)
    torch.manual_seed(worker_id+100)

def ddp_init(args):
    local_rank = args.local_rank

    # torch.set_default_tensor_type('torch.cuda.FloatTensor') # RuntimeError: Expected a 'N2at13CUDAGeneratorE' but found 'PN2at9GeneratorE'
    dist.init_process_group(backend = 'nccl')  # 'nccl' for GPU, 'gloo/mpi' for CPU

    torch.cuda.set_device(local_rank)

    rank = torch.distributed.get_rank()
    random.seed(rank)
    np.random.seed(rank)
    torch.manual_seed(rank)
    print(f"local_rank {local_rank} rank {rank} launched...")

    return rank, local_rank

def create_network(opt, net, local_rank=0):
    def load_network(opt, net, load=True):
        # init network from ckpts
        start_epoch, global_step = 0, 0
        ckpts = [os.path.join(opt.basedir, opt.name, f) for f in sorted(os.listdir(os.path.join(opt.basedir, opt.name))) if 'tar' in f]
        if len(ckpts) == 0:
            ckpts = [os.path.join(opt.basedir, opt.name, '..', f) for f in sorted(os.listdir(os.path.join(opt.basedir, opt.name, '..'))) if 'tar' in f]
        if len(ckpts) > 0:
            ckpt_path = ckpts[-1]
            logging.info(f'Reloading from {ckpt_path}')
            ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))  # load to cpu, otherwise this will occupy rank=0 's gpu memory
            if load:
                try:    # dp or ddp load itself
                    net.load_state_dict(ckpt['network_state_dict'])
                except:
                    try:    # dp_load_ddp
                        net.load_state_dict({'module.'+k: v for k, v in ckpt['network_state_dict'].items()})
                    except: # ddp_load_dp
                        net.load_state_dict({k[7:]: v for k, v in ckpt['network_state_dict'].items()})
            start_epoch = ckpt['epoch']
        # if no ckpts found, only init the encode from PIFu
        elif opt.load_netG_checkpoint_path is not None:
            logging.info(f'loading for net G ... {opt.load_netG_checkpoint_path}')
            pretrained_net = torch.load(opt.load_netG_checkpoint_path, map_location=torch.device('cpu'))
            if opt.ddp:
                pretrained_image_filter = {k: v for k, v in pretrained_net.items() if k.startswith('image_filter')}
            else:
                pretrained_image_filter = {'module.'+k: v for k, v in pretrained_net.items() if k.startswith('image_filter')}
            if load:
                net.load_state_dict(pretrained_image_filter, strict=False)
        return start_epoch
    
    # DDP: load parameters first (only on master node), then make ddp model
    if opt.ddp:
        logging.info("use Distributed Data Parallel...")
        net = net.to(local_rank) 
        start_epoch = load_network(opt, net, load=(dist.get_rank() == 0))
        net = DDP(net, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    # DP: make dp model, then load parameters to all devices 
    else:
        if torch.cuda.is_available():
            logging.info("use Data Parallel...")
            gpu_ids = [i for i in range(torch.cuda.device_count())]
            net = net.to(gpu_ids[0])
            net = torch.nn.DataParallel(net, device_ids=gpu_ids)
            start_epoch = load_network(opt, net)
            
    return net, start_epoch

def synchronize():
    if dist.get_world_size() > 1:
        dist.barrier()
    return

class ddpSampler:
    """
    ddp sampler for inference
    """
    def __init__(self, dataset, rank=None, num_replicas=None):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def indices(self):
        indices = list(range(len(self.dataset)))
        # add extra samples to make it evenly divisible
        indices += [indices[-1]] * (self.total_size - len(indices))
        # subsample
        indices = indices[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]
        return indices

    def len(self):
        return self.num_samples

    def distributed_concat(self, tensor, num_total_examples):
        output_tensors = [tensor.clone() for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(output_tensors, tensor)
        concat = torch.cat(output_tensors, dim=0)
        # truncate the dummy elements added by SequentialDistributedSampler
        return concat[:num_total_examples]