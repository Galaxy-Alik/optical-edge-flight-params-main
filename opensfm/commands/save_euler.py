# pyre-unsafe
from opensfm.actions import save_euler

from . import command
import argparse
from opensfm.dataset import DataSet
import time
import os

class Command(command.CommandBase):
    name = "save_euler"
    help = "Stiches the frames with optimized parameters"

    def run_impl(self, dataset: DataSet, args: argparse.Namespace) -> None:
        
        ft = time.time()
        save_euler.run_and_save(args.dataset)
        lt = time.time()

        print('Execution-Time (UNIX): ', (lt - ft), ' sec')

    def add_arguments_impl(self, parser: argparse.ArgumentParser) -> None:
        pass
