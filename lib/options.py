# import argparse
import configargparse
import os


class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument('--config', is_config_file=True, 
                            help='config file path')
        # Datasets related
        g_data = parser.add_argument_group('Data')
        g_data.add_argument('--dataroot', type=str, default='./data',
                            help='path to images (data folder)')

        g_data.add_argument('--loadSize', type=int, default=512, help='load size of input image')

        # Experiment related
        g_exp = parser.add_argument_group('Experiment')
        g_exp.add_argument('--name', type=str, default='smplnerf',
                           help='name of the experiment. It decides where to store samples and models')
        g_exp.add_argument('--debug', action='store_true', help='debug mode or not')
        g_exp.add_argument('--general', action='store_true', help='generalization or not')
        g_exp.add_argument('--output_mesh', action='store_true', help='output mesh or not')
        g_exp.add_argument('--train_shape', action='store_true', help='train shape')
        g_exp.add_argument('--train_encoder', action='store_true', help='train encoder')
        g_exp.add_argument('--angle_diff', action='store_true', help='train encoder')

        g_exp.add_argument('--train', action='store_true', help='only train')
        g_exp.add_argument('--render', action='store_true', help='only render')
        g_exp.add_argument('--test', action='store_true', help='only test')

        g_exp.add_argument('--num_views', type=int, default=4, help='How many views to use for multiview network.')
        g_exp.add_argument('--random_multiview', action='store_true', help='Select random multiview combination.')
        g_exp.add_argument('--use_feat_sr', action='store_true', help='use feature embedding super resolution')
        g_exp.add_argument('--use_fine', action='store_true', help='use fine nerf module')
        g_exp.add_argument('--use_attention', action='store_true', help='use multiview attention')
        g_exp.add_argument('--ddp', action='store_true', help='Distributed data parallel')
        g_exp.add_argument('--ddp_load_dp', action='store_true', help='Load dp checkpoint with ddp model')
        g_exp.add_argument('--dp_load_ddp', action='store_true', help='Load ddp checkpoint with dp model')
        g_exp.add_argument("--local_rank", default = -1, type = int)
        g_exp.add_argument('--use_nml', action='store_true', help='use volume normalization')
        g_exp.add_argument('--use_alpha_loss', action='store_true', help='use alpha loss')
        g_exp.add_argument('--weighted_pool', action='store_true', help='use volume normalization')
        g_exp.add_argument('--use_sh', action='store_true', help='use spherical hamonics')
        g_exp.add_argument('--use_vgg', action='store_true', help='use vgg perceptural loss')
        g_exp.add_argument('--regularization', action='store_true', help='use regularization loss')
        g_exp.add_argument('--use_occlusion', action='store_true', help='use occulsion aware')
        g_exp.add_argument('--use_occlusion_net', action='store_true', help='use occulsion aware network')
        g_exp.add_argument('--train_shape_skips', type=int, default=2, help='How many skips to train shape.')
        g_exp.add_argument('--move_cam', type=int, default='0', help='move camera for novel view rendering')

        g_exp.add_argument('--use_smpl_sdf', action='store_true', help='use smpl sdf')
        g_exp.add_argument('--use_t_pose', action='store_true', help='use smpl t-pose coordinates')
        g_exp.add_argument('--t_pose_path', type=str, default='./smpl_t_pose', help='path to smpl t pose obj')
        g_exp.add_argument('--use_skel_dist', action='store_true', help='use skeleton relative distance')
        g_exp.add_argument('--use_skel_dir', action='store_true', help='use skeleton relative direction')
        g_exp.add_argument('--use_smpl_betas', action='store_true', help='use smpl betas')
        g_exp.add_argument('--use_smpl_depth', action='store_true', help='use smpl depth')
        g_exp.add_argument('--smpl_type', type=str, default='smpl', help='smpl type, smpl/smplx/smli')
        g_exp.add_argument('--use_vh', action='store_true', help='use visual hull sampling')
        g_exp.add_argument('--vh_overhead', type=int, default=4, help='over head of visual hull sampling')
        g_exp.add_argument('--use_vh_free', action='store_true', help='use free space sampling')
        g_exp.add_argument('--use_bn', action='store_true', help='use batch normalization')
        g_exp.add_argument('--use_white_bkgd', action='store_true', help='use white background')
        g_exp.add_argument('--use_anchor', action='store_true', help='use anchor 3d points')
        g_exp.add_argument('--use_vh_sdf', action='store_true', help='use visual hull sdf')
        g_exp.add_argument('--use_vh_mc', action='store_true', help='use visual hull marching cube')

        g_exp.add_argument('--projection_mode', type=str, default='perspective', help='projection model, orthogonal or perspective')
        # g_exp.add_argument('--cpu_sample', action='store_true', help='use cpu sampels')

        g_exp.add_argument('--real_seq', type=str, default='s0', help='real sequence to process')
        g_exp.add_argument('--zju_seq', nargs='+', type=str, help='zju seq used')
        g_exp.add_argument('--zju_test_seq', nargs='+', type=str, help='zju test seq used')
        g_exp.add_argument('--zju_train_n_frame', type=int, help='zju use first n frame for training')
        g_exp.add_argument('--frames', nargs='+', default=[0, 2, 4, 6], type=int, help='frames used')
        g_exp.add_argument('--train_skip', type=int, default=5, help='training skip used')

        g_exp.add_argument('--ghr_seq', nargs='+', type=str, help='ghr seq used')
        g_exp.add_argument('--ghr_test_seq', nargs='+', type=str, help='ghr test seq used')
        g_exp.add_argument('--ghr_val', type=str, default='./ghr_val1.txt', help='ghr test val set')
        g_exp.add_argument('--ghr_train', type=str, default='./ghr_train1.txt', help='ghr train set')
        g_exp.add_argument('--ghr_load_list', action='store_true', help='ghr load from list')
        g_exp.add_argument('--ghr_render_path', type=str, default=None, help='ghr render path')
        # g_exp.add_argument('--ghr_render_merge', action='store_true', help='ghr render merge')
        # g_exp.add_argument('--ghr_merge_pose', nargs='+', type=int, help='ghr test seq used')
        g_exp.add_argument('--ghr_annot_path', type=str, default='./data/sensehuman/annots', help='ghr annotations load path')
        g_exp.add_argument('--ghr_smpl_depth_path', type=str, default='/mnt/lustre/chengwei/sensehuman', help='ghr annotations load path')
        g_exp.add_argument('--ghr_no_pretrain', action='store_true', help='ghr load from list')
        
        # Training related
        g_train = parser.add_argument_group('Training')
        g_train.add_argument('--gpu_id', type=int, default=0, help='gpu id for cuda')
        g_train.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2, -1 for CPU mode')

        g_train.add_argument('--num_threads', default=1, type=int, help='# sthreads for loading data')
        g_train.add_argument('--serial_batches', action='store_true',
                             help='if true, takes images in order to make batches, otherwise takes them randomly')
        g_train.add_argument('--pin_memory', action='store_true', help='pin_memory')
        
        g_train.add_argument('--batch_size', type=int, default=1, help='input batch size')
        g_train.add_argument('--lrate', type=float, default=5e-4, help='adam learning rate')
        g_train.add_argument('--lrate_decay', type=float, default=50, help='adam learning rate decay')
        g_train.add_argument('--learning_rateC', type=float, default=1e-3, help='adam learning rate')
        g_train.add_argument('--learning_rate', type=float, default=1e-3, help='adam learning rate')
        g_train.add_argument('--num_epoch', type=int, default=1000, help='num epoch to train')

        g_train.add_argument('--freq_plot', type=int, default=100, help='freqency of the error plot')
        g_train.add_argument('--freq_save', type=int, default=500, help='freqency of the save_checkpoints')
        g_train.add_argument('--freq_save_ply', type=int, default=100, help='freqency of the save ply')
       
        g_train.add_argument('--no_gen_mesh', action='store_true')
        g_train.add_argument('--no_num_eval', action='store_true')
        
        g_train.add_argument('--resume_epoch', type=int, default=-1, help='epoch resuming the training')
        g_train.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        g_train.add_argument('--replica', type=int, default=10, help='replication of training set')
        g_train.add_argument('--num_random_sh', type=int, default=5, help='How many random spherical hamonical lights')
        g_train.add_argument('--sel_data', action='store_true', help='if true use selected data')
        g_train.add_argument('--eval_dir', type=str, default='eval', help='if true use selected data')

        # Testing related
        g_test = parser.add_argument_group('Testing')
        g_test.add_argument('--resolution', type=int, default=256, help='# of grid in mesh reconstruction')
        g_test.add_argument('--test_folder_path', type=str, default=None, help='the folder of test image')
        g_test.add_argument('--test_list', type=str, default="./test.txt")
        g_train.add_argument('--num_variants', type=int, default=5)
        g_test.add_argument('--val_description', type=str, default="./val_description.json")
        g_test.add_argument('--test_description', type=str, default="./new_test_description.json")
        g_test.add_argument('--resdir', type=str, default="./test_results")
        g_test.add_argument('--eval_phase', type=str, default="test")
        g_test.add_argument('--laplacian', type=int, default=0, help='iters of laplacian smoothing')
        g_test.add_argument('--workers', type=int, default=1)

        # Sampling related
        g_sample = parser.add_argument_group('Sampling')
        g_sample.add_argument('--sigma', type=float, default=5.0, help='perturbation standard deviation for positions')

        g_sample.add_argument('--num_sample_inout', type=int, default=5000, help='# of sampling points')
        g_sample.add_argument('--num_sample_color', type=int, default=0, help='# of sampling points')

        g_sample.add_argument('--z_size', type=float, default=200.0, help='z normalization factor')

        # Model related
        g_model = parser.add_argument_group('Model')
        # General
        g_model.add_argument('--norm', type=str, default='group',
                             help='instance normalization or batch normalization or group normalization')
        g_model.add_argument('--norm_color', type=str, default='instance',
                             help='instance normalization or batch normalization or group normalization')

        # NeRF
        g_nerf = parser.add_argument_group('NeRF')
        g_nerf.add_argument('--use_viewdirs', action='store_true', help='use view direction')
        g_model.add_argument('--N_samples', type=int, default=64, help='# of samples per ray')
        g_model.add_argument('--N_rand', type=int, default=1024, help='# of random pixels (to render)')
        g_model.add_argument('--chunk', type=int, default=1024*64, help='MLP batch_size')
        g_model.add_argument('--N_rand_infer', type=int, default=1024, help='MLP batch_size at inference')
        g_model.add_argument('--N_grid', type=int, default=256, help='marching cube resolution')

        # hg filter specify
        g_model.add_argument('--num_stack', type=int, default=4, help='# of hourglass')
        g_model.add_argument('--num_hourglass', type=int, default=2, help='# of stacked layer of hourglass')
        g_model.add_argument('--skip_hourglass', action='store_true', help='skip connection in hourglass')
        g_model.add_argument('--hg_down', type=str, default='ave_pool', help='ave pool || conv64 || conv128')
        g_model.add_argument('--hourglass_dim', type=int, default='256', help='256 | 512')

        # Classification General
        g_model.add_argument('--mlp_dim', nargs='+', default=[257, 1024, 512, 256, 128, 1], type=int,
                             help='# of dimensions of mlp')
        g_model.add_argument('--mlp_dim_color', nargs='+', default=[513, 1024, 512, 256, 128, 3],
                             type=int, help='# of dimensions of color mlp')

        g_model.add_argument('--use_tanh', action='store_true',
                             help='using tanh after last conv of image_filter network')
        g_model.add_argument('--skips', nargs='+', default=[2, 4, 6], type=int,
                             help='# of dimensions of mlp')
        # for train
        parser.add_argument('--random_flip', action='store_true', help='if random flip')
        parser.add_argument('--random_trans', action='store_true', help='if random flip')
        parser.add_argument('--random_scale', action='store_true', help='if random flip')
        parser.add_argument('--no_residual', action='store_true', help='no skip connection in mlp')
        parser.add_argument('--schedule', type=int, nargs='+', default=[60, 80],
                            help='Decrease learning rate at these epochs.')
        parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
        parser.add_argument('--color_loss_type', type=str, default='l1', help='mse | l1')

        # for eval
        parser.add_argument('--val_test_error', action='store_true', help='validate errors of test data')
        parser.add_argument('--val_train_error', action='store_true', help='validate errors of train data')
        parser.add_argument('--gen_test_mesh', action='store_true', help='generate test mesh')
        parser.add_argument('--gen_train_mesh', action='store_true', help='generate train mesh')
        parser.add_argument('--all_mesh', action='store_true', help='generate meshs from all hourglass output')
        parser.add_argument('--num_gen_mesh_test', type=int, default=1,
                            help='how many meshes to generate during testing')
        parser.add_argument('--eval_skip', type=int, default=10, help='evaluation skips')


        # path
        parser.add_argument('--basedir', type=str, default='./logs', help='path to save logs')
        parser.add_argument('--checkpoints_path', type=str, default='./checkpoints', help='path to save checkpoints')
        parser.add_argument('--load_netG_checkpoint_path', type=str, default=None, help='path to save checkpoints')
        parser.add_argument('--load_netC_checkpoint_path', type=str, default=None, help='path to save checkpoints')
        parser.add_argument('--results_path', type=str, default='./results', help='path to save results ply')
        parser.add_argument('--load_checkpoint_path', type=str, help='path to save results ply')
        parser.add_argument('--single', type=str, default='', help='single data for training')
        parser.add_argument('--load_pretrained_filter', action='store_true', help='load pretrained filter or not')
        # for single image reconstruction
        parser.add_argument('--mask_path', type=str, help='path for input mask')
        parser.add_argument('--img_path', type=str, help='path for input image')

        # aug
        group_aug = parser.add_argument_group('aug')
        group_aug.add_argument('--aug_alstd', type=float, default=0.0, help='augmentation pca lighting alpha std')
        group_aug.add_argument('--aug_bri', type=float, default=0.0, help='augmentation brightness')
        group_aug.add_argument('--aug_con', type=float, default=0.0, help='augmentation contrast')
        group_aug.add_argument('--aug_sat', type=float, default=0.0, help='augmentation saturation')
        group_aug.add_argument('--aug_hue', type=float, default=0.0, help='augmentation hue')
        group_aug.add_argument('--aug_blur', type=float, default=0.0, help='augmentation blur')

        # special tasks
        self.initialized = True
        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            # parser = argparse.ArgumentParser(
            #     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = configargparse.ArgumentParser()
            parser = self.initialize(parser)

        self.parser = parser

        return parser.parse_args()

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

    def parse(self):
        opt = self.gather_options()
        return opt
