import argparse
import collections.abc
import os
import yaml
import torch

from gencd.utils.misc import make_dir_with_number, find_dir_with_number

class TrainOptions():
    def __init__(self):
        self.initialized = False

    @staticmethod
    def add_model_specific_args(parser):
        # network
        parser.add_argument('--net_module', type=str, help='network module. if None, use current models.networks')
        parser.add_argument('--net_config', type=str, help='path to network config.yaml')
        parser.add_argument('--load_pretrained_network', type=str, help='path to pretrained network')
        parser.add_argument('--load_pretrained_ddpm', type=str, help='path to pretrained ddpm')
        
        # model
        parser.add_argument('--model', type=str, default='cd_base', help='chooses which model to use.')
        parser.add_argument('--load_pretrained_model', type=str, help='path to pretrained model (gencd)')
        parser.add_argument('--loss', type=str, default='ce', help='refer to models.losses')
        return parser
    
    @staticmethod
    def get_model_specific_args():
        dummy = argparse.ArgumentParser()
        dummy = TrainOptions.add_model_specific_args(dummy)
        dummyopt = dummy.parse_args(args=[])
        return vars(dummyopt).keys()
        
    def initialize(self, parser):
        # basic
        parser.add_argument('--run_base_dir', type=str, default='./runs', help='models are saved here (train) or loaded from here (test)')
        parser.add_argument('--exp_name', type=str, default='exp', help='experiment name')
        parser.add_argument('--exp_number', type=int, default=0, help='if 0, make new folder under run_base_dir/exp_name. if >0, find existing folder or make new folder with the number. if folder exists, suffix is ignored.')
        parser.add_argument('--exp_suffix', type=str, default='', help='Becomes suffix of save_dir. e.g., run_base_dir/{exp_name}/{exp_number}_{exp_suffix}')
        parser.add_argument('--fold', type=int, nargs='+', help='-1 for no fold. if use, e.g. 0 1 2 in 5-fold')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--num_workers', default=4, type=int, help='# threads for loading data')
        parser.add_argument('--no_saveoptions', action='store_true')
        parser.add_argument('--seed_determinism', type=int, help='seed for set_determinism. None: no use. negative: use default value.')
        parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        
        # data
        parser.add_argument('--datadir', type=str, help='path to data')
        parser.add_argument('--dataset_split', type=str, help='path to dataset split.pkl')
        #parser.add_argument('--file_extension', type=str, default='png', help='case-sensitive image file extension')
        parser.add_argument('--dataset_mode', type=str, help='name of dataset class')
        parser.add_argument('--num_class', type=int, default=2, help='number of classes including background')
        parser.add_argument('--patch_size', default=256, type=int, help='input patch size')
        parser.add_argument('--patch_overlap', default=0, type=float, help='for patch-based dataloader. not used in random crop')
        parser.add_argument('--patch_overlap_inference', default=0.5, type=float, help='overlap for sliding window inference')
        parser.add_argument('--patch_resize_factor', default=1, type=float, help='resize patch. 2 means half')
        parser.add_argument('--batch_size', default=1, type=int, help='batch size for train')
        parser.add_argument('--batch_size_inference', default=1, type=int, help='batch size for inference')
        
        # model & network
        parser = TrainOptions.add_model_specific_args(parser)
        parser.add_argument('--metric', type=str, default='f1_iou', help='iou, f1')

        # training parameters
        parser.add_argument('--train_only', action='store_true')
        parser.add_argument('--max_epochs', type=int, default=100, help='number of epochs')        
        parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
        parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. CUT, cycleGAN default is linear. nnUNet default is poly [step | poly | linear | plateau | cosine]')
        parser.add_argument('--optimizer', type=str, default='adam', help='adam or sgd')
                
        # trainer parameters
        parser.add_argument('--resume_from_checkpoint', type=str, help='path to checkpoint. for resume training.')
        parser.add_argument('--callbacks', type=str, help='result, ckpt, lr, metricvalid')
        parser.add_argument('--result_dir', type=str, help='save validation results. if none, use run_dir/result')
        parser.add_argument('--loggers', type=str, help='csv, tb, wandb')
        parser.add_argument('--wandb_project', type=str, default='maicon')
        parser.add_argument('--wandb_name', type=str)
        parser.add_argument('--log_every_n_steps', type=int, default=10, help='logging frequency')
        parser.add_argument('--check_val_every_n_epoch', type=int, default=1, help='run valid loop every n epoch')
        parser.add_argument('--checkpoint_monitor', type=str, default='loss/val_loss', help='checkpoint monitor. passed to ModelCheckpoint')
        parser.add_argument('--checkpoint_monitor_mode', type=str, default='min', help='e.g. min for losses. max for accuracy. passed to ModelCheckpoint')
        parser.add_argument('--checkpoint_save_top_k', default=1, type=int, help='checkpoint save_top_k. refer to Lightning docs')
        parser.add_argument('--checkpoint_nooverwrite', action='store_true', help='deprecated. keep every saved checkpoint. passed to ModelCheckpoint')  
        parser.add_argument('--checkpoint_every_n_epochs', type=int, help='checkpoint every n epochs. passed to ModelCheckpoint')
        parser.add_argument('--checkpoint_filename', type=str, default='epoch={epoch}', help='checkpoint file format')
        parser.add_argument('--save_fullmodel', action='store_true', help='if True, save full model. Else, save weights only')
        parser.add_argument('--mixed_precision', action='store_true', help='if True, use AMP')
        parser.add_argument('--detect_anomaly', action='store_true', help='detect anomaly')
        
        self.initialized = True
        return parser
        
    def parse(self, args=None):
        opt = self.gather_options(args=args)
        opt = self.setup(opt)
        opt = self.print_options(opt)
        
        # parse gpu ids as used in Pytorch Lightning Trainer 
        # https://pytorch-lightning.readthedocs.io/en/stable/advanced/multi_gpu.html
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) == 0:
            opt.gpu_ids = None # CPU
            
        self.opt = opt
        return opt
        
    def gather_options(self, args=None):
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
            
        if args is None:
            opt = parser.parse_args()
        else:
            opt = parser.parse_args(args=args)        
            
        # save and return the parser
        self.parser = parser
        return opt

    def setup(self, opt):
        ### save dir
        run_dir = os.path.join(opt.run_base_dir, opt.exp_name)
        if opt.exp_number==0:
            save_dir = make_dir_with_number(run_dir, fname=opt.exp_suffix)
        else:
            try:
                save_dir = find_dir_with_number(opt.exp_number, run_dir)
                if save_dir is None:
                    save_dir = make_dir_with_number(run_dir, fname=opt.exp_suffix, num=opt.exp_number)
            except:
                save_dir = make_dir_with_number(run_dir, fname=opt.exp_suffix, num=opt.exp_number)    
        opt.save_dir = save_dir
        
        ### inference for later use
        opt.inference = False
        
        return opt
    
    def print_options(self, opt):
        '''Print options and save
        '''
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in vars(opt).items():
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)        
        
        if not opt.no_saveoptions:
            file_name = os.path.join(opt.save_dir, f'_train_opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write(message)
                opt_file.write('\n')        
            print(f'saved {file_name}')
        
        return opt