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
        g_exp.add_argument('--use_smpl_depth', action='store_true', help='use smpl depth')
        g_exp.add_argument('--smpl_type', type=str, default='smpl', help='smpl type, smpl/smplx/smli')
        g_exp.add_argument('--use_vh', action='store_true', help='use visual hull sampling')
        g_exp.add_argument('--vh_overhead', type=int, default=4, help='over head of visual hull sampling')
        g_exp.add_argument('--use_vh_free', action='store_true', help='use free space sampling')
        g_exp.add_argument('--use_bn', action='store_true', help='use batch normalization')
        g_exp.add_argument('--use_white_bkgd', action='store_true', help='use white background')
        g_exp.add_argument('--use_vh_sdf', action='store_true', help='use visual hull sdf')

        g_exp.add_argument('--projection_mode', type=str, default='perspective', help='projection model, orthogonal or perspective')

        # zjumocap dataset releate
        g_zju = parser.add_argument_group('Experiment')
        g_zju.add_argument('--zju_seq', nargs='+', type=str, help='zju seq used')
        g_zju.add_argument('--zju_test_seq', nargs='+', type=str, help='zju test seq used')
        g_zju.add_argument('--zju_train_n_frame', type=int, help='zju use first n frame for training')
        g_zju.add_argument('--frames', nargs='+', default=[0, 2, 4, 6], type=int, help='frames used')
        g_zju.add_argument('--train_skip', type=int, default=1, help='training skip used')
        
        # Training related
        g_train = parser.add_argument_group('Training')
        g_train.add_argument('--gpu_id', type=int, default=0, help='gpu id for cuda')
        g_train.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2, -1 for CPU mode')

        g_train.add_argument('--num_threads', default=1, type=int, help='# sthreads for loading data')
        g_train.add_argument('--batch_size', type=int, default=1, help='input batch size')
        g_train.add_argument('--lrate', type=float, default=5e-4, help='adam learning rate')
        g_train.add_argument('--lrate_decay', type=float, default=50, help='adam learning rate decay')
        g_train.add_argument('--num_epoch', type=int, default=1000, help='num epoch to train')
        g_train.add_argument('--freq_plot', type=int, default=100, help='freqency of the error plot')
        
        g_train.add_argument('--replica', type=int, default=10, help='replication of training set')
        g_train.add_argument('--num_random_sh', type=int, default=5, help='How many random spherical hamonical lights')
        g_train.add_argument('--sel_data', action='store_true', help='if true use selected data')


        # Testing related
        g_test = parser.add_argument_group('Testing')
        g_test.add_argument('--val_description', type=str, default="./val_description.json")
        g_test.add_argument('--test_description', type=str, default="./new_test_description.json")
        g_test.add_argument('--laplacian', type=int, default=0, help='iters of laplacian smoothing')
        g_test.add_argument('--workers', type=int, default=1)
        g_train.add_argument('--eval_dir', type=str, default='eval', help='if true use selected data')
        g_train.add_argument('--render_dir', type=str, default='render', help='if true use selected data')

        # Model related
        g_model = parser.add_argument_group('Model')

        # NeRF
        g_nerf = parser.add_argument_group('NeRF')
        g_nerf.add_argument('--use_viewdirs', action='store_true', help='use view direction')
        g_nerf.add_argument('--N_samples', type=int, default=64, help='# of samples per ray')
        g_nerf.add_argument('--N_rand', type=int, default=1024, help='# of random pixels (to render)')
        g_nerf.add_argument('--chunk', type=int, default=1024*64, help='MLP batch_size')
        g_nerf.add_argument('--N_rand_infer', type=int, default=1024, help='MLP batch_size at inference')
        g_nerf.add_argument('--N_grid', type=int, default=256, help='marching cube resolution')
        g_nerf.add_argument('--skips', nargs='+', default=[2, 4, 6], type=int,
                             help='# of dimensions of mlp')

        # PIFu hg filter specify
        g_pifu = parser.add_argument_group('PIFu')
        g_pifu.add_argument('--norm', type=str, default='group',
                             help='instance normalization or batch normalization or group normalization')
        g_pifu.add_argument('--num_stack', type=int, default=4, help='# of hourglass')
        g_pifu.add_argument('--num_hourglass', type=int, default=2, help='# of stacked layer of hourglass')
        g_pifu.add_argument('--skip_hourglass', action='store_true', help='skip connection in hourglass')
        g_pifu.add_argument('--hg_down', type=str, default='ave_pool', help='ave pool || conv64 || conv128')
        g_pifu.add_argument('--hourglass_dim', type=int, default='256', help='256 | 512')

        # for eval
        parser.add_argument('--eval_skip', type=int, default=10, help='evaluation skips')

        # path
        parser.add_argument('--basedir', type=str, default='./logs', help='path to save logs')
        # load netG from PIFu pretrained model
        parser.add_argument('--load_netG_checkpoint_path', type=str, default=None, help='path to save checkpoints')

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
