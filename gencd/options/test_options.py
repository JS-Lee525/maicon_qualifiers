import argparse
import collections.abc
import os
import yaml
import torch

from gencd.utils.misc import make_dir_with_number, find_dir_with_number

class TestOptions():
    def __init__(self):
        self.initialized = False
        
    def initialize(self, parser):
        # basic
        parser.add_argument('--run_base_dir', type=str, default='./results', help='models are saved here (train) or loaded from here (test)')
        parser.add_argument('--fold', type=int, nargs='+', help='-1 for no fold. if use, e.g. 0 1 2 in 5-fold')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--num_workers', default=4, type=int, help='# threads for loading data')
        parser.add_argument('--no_saveoptions', action='store_true')
        parser.add_argument('--mixed_precision', action='store_true', help='if True, use AMP')
        parser.add_argument('--save_temp', action='store_true', help='only used in ensemble. if True, save individual results')
        parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        
        # data
        parser.add_argument('--phase', type=str, help='name of dataset class')
        parser.add_argument('--dataset_mode', type=str, help='name of dataset class')
        parser.add_argument('--datadir', type=str, help='path to data')
        parser.add_argument('--dataset_split', type=str, help='path to dataset split.pkl')
        parser.add_argument('--num_class', type=int, default=2, help='number of classes including background')
        parser.add_argument('--patch_size', default=256, type=int, help='input patch size')
        parser.add_argument('--patch_resize_factor', default=1, type=float, help='resize patch. 2 means half')
        parser.add_argument('--patch_overlap_inference', default=0.5, type=float, help='overlap for sliding window inference')
        parser.add_argument('--batch_size', default=1, type=int, help='batch size for sliding window')
        parser.add_argument('--batch_size_inference', default=1, type=int, help='batch size for inference')
        
        # model
        parser.add_argument('--load_pretrained_model', type=str, nargs='+', help='path to pretrained model (gencd)')
        parser.add_argument('--metric', type=str, default='f1_iou', help='iou, f1')
                        
        # trainer
        parser.add_argument('--callbacks', type=str, help='result')
        parser.add_argument('--loggers', type=str, help='csv, tb, wandb')
        parser.add_argument('--wandb_project', type=str, default='maicon', help='csv, tb, wandb')
        parser.add_argument('--wandb_name', type=str)
        parser.add_argument('--log_every_n_steps', type=int, default=10, help='logging frequency')
            
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
        opt.save_results = True
        ### result dir
        if opt.save_results:
            opt.result_dir = opt.save_dir = opt.run_base_dir 
            os.makedirs(opt.result_dir, exist_ok=True)
        
        opt.inference = True
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
            file_name = os.path.join(opt.save_dir, f'_test_opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write(message)
                opt_file.write('\n')        
            print(f'saved {file_name}')
        
        return opt