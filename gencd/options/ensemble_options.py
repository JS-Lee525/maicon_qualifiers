import argparse
import collections.abc
import os
import yaml
import torch

from gencd.utils.misc import make_dir_with_number, find_dir_with_number
from .test_options import TestOptions

class EnsembleOptions(TestOptions):
    def initialize(self, parser):
        parser = TestOptions.initialize(self, parser)
        
        parser.set_defaults(callbacks='result')
        parser.add_argument('--save_temp_results', action='store_true', help='only used in ensemble. if True, save individual results')
                    
        self.initialized = True
        return parser