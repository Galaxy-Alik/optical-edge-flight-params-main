# pyre-unsafe
from timeit import default_timer as timer

from opensfm import io
from opensfm import matching
from opensfm.dataset_base import DataSetBase
import logging
import os
import numpy as np

logger: logging.Logger = logging.getLogger(__name__)

def run_dataset(data: DataSetBase, src_folder) -> None:
    """Match features between image pairs."""

    logger.info(' --- Running - Match - Features --- ')

    images = data.images()

    start = timer()
    pairs_matches, preport = matching.match_images(data, {}, images, images)
    
    print(' #################### ')
    print(' -------- pairs_matches -------- ')
    print(pairs_matches)
    print(' #################### ')

    save_path = r"/home/datademon/Desktop/Alik/galax_spip_v2/data"
    save_path = os.path.join(save_path, src_folder)
    save_path = os.path.join(save_path, 'all_matches')
    
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    
    save_path = os.path.join(save_path, f"{src_folder.split('-')[1]}_matches.npy")
    np.save(save_path, pairs_matches)
    matching.save_matches(data, images, pairs_matches)
    matching.clear_cache()
    end = timer()
    write_report(data, preport, list(pairs_matches.keys()), end - start)


def write_report(data: DataSetBase, preport, pairs, wall_time) -> None:
    report = {
        "wall_time": wall_time,
        "num_pairs": len(pairs),
        "pairs": pairs,
    }
    report.update(preport)
    data.save_report(io.json_dumps(report), "matches.json")
