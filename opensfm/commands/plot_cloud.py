# pyre-unsafe
from opensfm.actions import plot_cloud

from . import command
import argparse
from opensfm.dataset import DataSet
import time
import os
import logging

logger: logging.Logger = logging.getLogger(__name__)

class Command(command.CommandBase):
    name = "plot_cloud"
    help = "Stiches the frames with optimized parameters"

    def run_impl(self, dataset: DataSet, args: argparse.Namespace) -> None:
        print('-------- Stitch_v2 -- (Commands) --------')
        print('Image-Len: ', len(dataset.images()))
        print('-----------------')
        ft = time.time()

        json_path = os.path.join('/home/datademon/Desktop/Alik/galax_spip_v2/data', args.dataset)
        json_path = os.path.join(json_path, 'reconstruction_sampled.json')
        
        plot_cloud.plot_3d_point_cloud(json_path)

        logger.info('Successfully Saved Point (V + H) Cloud!')

        lt = time.time()
        print('Execution-Time (UNIX): ', (lt - ft), ' sec')

    def add_arguments_impl(self, parser: argparse.ArgumentParser) -> None:
        pass
