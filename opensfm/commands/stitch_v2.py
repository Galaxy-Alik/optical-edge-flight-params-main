# pyre-unsafe
from opensfm.actions import stitch_v2

from . import command
import argparse
from opensfm.dataset import DataSet
import time

class Command(command.CommandBase):
    name = "stitch_v2"
    help = "Stiches the frames with optimized parameters"

    def run_impl(self, dataset: DataSet, args: argparse.Namespace) -> None:
        print('-------- Stitch_v2 -- (Commands) --------')
        print('Image-Len: ', len(dataset.images()))
        print('-----------------')
        ft = time.time()
        stitch_v2.main_plot_save(args.dataset)
        lt = time.time()
        print('Execution-Time (UNIX): ', (lt - ft), ' sec')

    def add_arguments_impl(self, parser: argparse.ArgumentParser) -> None:
        pass
