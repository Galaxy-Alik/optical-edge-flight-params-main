# pyre-unsafe
import gzip
import json
import logging
import os
import pickle
from io import BytesIO
from typing import Dict, List, Tuple, Optional, IO, Any

import numpy as np
from opensfm import (
    config,
    features,
    geo,
    io,
    pygeometry,
    types,
    pymap,
    masking,
    rig
)
from opensfm.dataset_base import DataSetBase
from PIL.PngImagePlugin import PngImageFile
import random

logger: logging.Logger = logging.getLogger(__name__)


class DataSet(DataSetBase):
    """Accessors to the main input and output data.

    Data include input images, masks, and segmentation as well
    temporary data such as features and matches and the final
    reconstructions.

    All data is stored inside a single folder with a specific subfolder
    structure.

    It is possible to store data remotely or in different formats
    by subclassing this class and overloading its methods.
    """

    io_handler: io.IoFilesystemBase = io.IoFilesystemDefault()
    config = None
    image_files: Dict[str, str] = {}
    mask_files: Dict[str, str] = {}
    image_list: List[str] = []

    def __init__(self, data_path: str, io_handler=io.IoFilesystemDefault) -> None:
        """Init dataset associated to a folder."""
        self.io_handler = io_handler
        self.data_path = data_path
        self.load_config()
        self.load_image_list()
        self.load_mask_list()

    def _config_file(self) -> str:
        return os.path.join(self.data_path, "config.yaml")

    def load_config(self) -> None:
        config_file_path = self._config_file()
        if self.io_handler.isfile(config_file_path):
            with self.io_handler.open(config_file_path) as f:
                self.config = config.load_config_from_fileobject(f)
        else:
            self.config = config.default_config()

    def _image_list_file(self) -> str:
        return os.path.join(self.data_path, "image_list.txt")

    def load_image_list(self) -> None:
        """Load image list from image_list.txt or list images/ folder."""
        image_list_file = self._image_list_file()
        image_list_path = os.path.join(self.data_path, "images")

        if self.io_handler.isfile(image_list_file):
            with self.io_handler.open_rt(image_list_file) as fin:
                lines = fin.read().splitlines()
            self._set_image_list(lines)
            self._set_image_list(lines)
        else:
            self._set_image_path(image_list_path)

        if self.data_path and not self.image_list:
            raise IOError("No Images found in {}".format(image_list_path))

    # def key_fn(item):
    #     print(' ------------ Key_fn ------------ ')
    #     return item.split('_')[2]

    def images(self) -> List[str]:
        """List of file names of all images in the dataset."""

        # self.image_list = sorted(self.image_list, key=lambda k: int(k.split('_')[2]), reverse=False)
        # print(' ****************************** ')
        # print('Image-List (CMD-RUN): ', self.image_list)
        # print(' ****************************** ')
        return self.image_list

    def _image_file(self, image: str) -> str:
        """Path to the image file."""
        return self.image_files[image]

    def open_image_file(self, image: str) -> IO[Any]:
        """Open image file and return file object."""
        return self.io_handler.open(self._image_file(image), "rb")

    def load_image(
        self,
        image: str,
        unchanged: bool = False,
        anydepth: bool = False,
        grayscale: bool = False,
    ) -> np.ndarray:
        """Load image pixels as numpy array.

        The array is 3D, indexed by y-coord, x-coord, channel.
        The channels are in RGB order.
        """
        return self.io_handler.imread(
            self._image_file(image),
            unchanged=unchanged,
            anydepth=anydepth,
            grayscale=grayscale,
        )

    def image_size(self, image: str) -> Tuple[int, int]:
        """Height and width of the image."""
        return self.io_handler.image_size(self._image_file(image))

    def load_mask_list(self) -> None:
        """Load mask list from mask_list.txt or list masks/ folder."""
        mask_list_file = os.path.join(self.data_path, "mask_list.txt")
        if self.io_handler.isfile(mask_list_file):
            with self.io_handler.open_rt(mask_list_file) as fin:
                lines = fin.read().splitlines()
            self._set_mask_list(lines)
        else:
            self._set_mask_path(os.path.join(self.data_path, "masks"))

    def load_mask(self, image: str) -> Optional[np.ndarray]:
        """Load image mask if it exists, otherwise return None."""
        if image in self.mask_files:
            mask_path = self.mask_files[image]
            mask = self.io_handler.imread(mask_path, grayscale=True)
            if mask is None:
                raise IOError(
                    "Unable to load mask for image {} "
                    "from file {}".format(image, mask_path)
                )
        else:
            mask = None
        return mask

    def _instances_path(self) -> str:
        return os.path.join(self.data_path, "instances")

    def _instances_file(self, image: str) -> str:
        return os.path.join(self._instances_path(), image + ".png")

    def load_instances(self, image: str) -> Optional[np.ndarray]:
        """Load image instances file if it exists, otherwise return None."""
        instances_file = self._instances_file(image)
        if self.io_handler.isfile(instances_file):
            instances = self.io_handler.imread(instances_file, grayscale=True)
        else:
            instances = None
        return instances

    def _segmentation_path(self) -> str:
        return os.path.join(self.data_path, "segmentations")

    def _segmentation_file(self, image: str) -> str:
        return os.path.join(self._segmentation_path(), image + ".png")

    def segmentation_labels(self) -> List[Any]:
        return []

    def load_segmentation(self, image: str) -> Optional[np.ndarray]:
        """Load image segmentation if it exists, otherwise return None."""
        segmentation_file = self._segmentation_file(image)
        if self.io_handler.isfile(segmentation_file):
            with self.io_handler.open(segmentation_file, "rb") as fp:
                with PngImageFile(fp) as png_image:
                    # TODO: We do not write a header tag in the metadata. Might be good safety check.
                    data = np.array(png_image)
                    if data.ndim == 2:
                        return data
                    elif data.ndim == 3:
                        return data[:, :, 0]

                        # TODO we can optionally return also the instances and scores:
                        # instances = (
                        #     data[:, :, 1].astype(np.int16) + data[:, :, 2].astype(np.int16) * 256
                        # )
                        # scores = data[:, :, 3].astype(np.float32) / 256.0
                    else:
                        raise IndexError
        else:
            segmentation = None
        return segmentation

    def segmentation_ignore_values(self, image: str) -> List[int]:
        """List of label values to ignore.

        Pixels with these label values will be masked out and won't be
        processed when extracting and matching features.
        """
        return self.config.get("segmentation_ignore_values", [])

    def undistorted_segmentation_ignore_values(self, image: str) -> List[int]:
        """List of label values to ignore on undistorted images

        Pixels with these label values will be masked out and won't be
        processed when computing depthmaps.
        """
        return self.config.get(
            "undistorted_segmentation_ignore_values",
            self.segmentation_ignore_values(image),
        )

    def _is_image_file(self, filename: str) -> bool:
        extensions = {"jpg", "jpeg", "png", "tif", "tiff", "pgm", "pnm", "gif"}
        return filename.split(".")[-1].lower() in extensions

    def _set_image_path(self, path: str) -> None:
        """Set image path and find all images in there"""
        self.image_list = []
        self.image_files = {}
        if self.io_handler.exists(path):
            for name in self.io_handler.ls(path):
                if self._is_image_file(name):
                    self.image_list.append(name)
                    self.image_files[name] = os.path.join(path, name)

    def _set_image_list(self, image_list: List[str]) -> None:
        self.image_list = []
        self.image_files = {}
        for line in image_list:
            path = os.path.join(self.data_path, line)
            name = os.path.basename(path)
            self.image_list.append(name)
            self.image_files[name] = path

    def _set_mask_path(self, path: str) -> None:
        """Set mask path and find all masks in there"""
        self.mask_files = {}
        if self.io_handler.isdir(path):
            files = set(self.io_handler.ls(path))
            for image in self.images():
                mask = image + ".png"
                if mask in files:
                    self.mask_files[image] = os.path.join(path, mask)

    def _set_mask_list(self, mask_list_lines: List[str]) -> None:
        self.mask_files = {}
        for line in mask_list_lines:
            image, relpath = line.split(None, 1)
            path = os.path.join(self.data_path, relpath.strip())
            self.mask_files[image.strip()] = path

    def _exif_path(self) -> str:
        """Return path of extracted exif directory"""
        return os.path.join(self.data_path, "exif")

    def _exif_file(self, image: str) -> str:
        """
        Return path of exif information for given image
        :param image: Image name, with extension (i.e. 123.jpg)
        """
        return os.path.join(self._exif_path(), image + ".exif")

    def load_exif(self, image: str) -> Dict[str, Any]:
        """Load pre-extracted image exif metadata."""
        with self.io_handler.open_rt(self._exif_file(image)) as fin:
            return json.load(fin)

    def save_exif(self, image: str, data: Dict[str, Any]) -> None:
        self.io_handler.mkdir_p(self._exif_path())
        with self.io_handler.open_wt(self._exif_file(image)) as fout:
            io.json_dump(data, fout)

    def exif_exists(self, image: str) -> bool:
        return self.io_handler.isfile(self._exif_file(image))

    def feature_type(self) -> str:
        """Return the type of local features (e.g. AKAZE, SURF, SIFT)"""
        feature_name = self.config["feature_type"].lower()
        if self.config["feature_root"]:
            feature_name = "root_" + feature_name
        return feature_name

    def _feature_path(self) -> str:
        """Return path of feature descriptors and FLANN indices directory"""
        return os.path.join(self.data_path, "features")

    def _feature_file(self, image: str) -> str:
        """
        Return path of feature file for specified image
        :param image: Image name, with extension (i.e. 123.jpg)
        """
        return os.path.join(self._feature_path(), image + ".features.npz")

    def _feature_file_legacy(self, image: str) -> str:
        """
        Return path of a legacy feature file for specified image
        :param image: Image name, with extension (i.e. 123.jpg)
        """
        return os.path.join(self._feature_path(), image + ".npz")

    def _save_features(
        self, filepath: str, features_data: features.FeaturesData
    ) -> None:
        self.io_handler.mkdir_p(self._feature_path())
        with self.io_handler.open(filepath, "wb") as fwb:
            features_data.save(fwb, self.config)

    def features_exist(self, image: str) -> bool:
        return self.io_handler.isfile(
            self._feature_file(image)
        ) or self.io_handler.isfile(self._feature_file_legacy(image))

    def load_features(self, image: str) -> Optional[features.FeaturesData]:
        features_filepath = (
            self._feature_file_legacy(image)
            if self.io_handler.isfile(self._feature_file_legacy(image))
            else self._feature_file(image)
        )
        with self.io_handler.open(features_filepath, "rb") as f:
            return features.FeaturesData.from_file(f, self.config)

    def save_features(self, image: str, features_data: features.FeaturesData) -> None:
        self._save_features(self._feature_file(image), features_data)

    def _words_file(self, image: str) -> str:
        return os.path.join(self._feature_path(), image + ".words.npz")

    def words_exist(self, image: str) -> bool:
        return self.io_handler.isfile(self._words_file(image))

    def load_words(self, image: str) -> np.ndarray:
        with self.io_handler.open(self._words_file(image), "rb") as f:
            s = np.load(f)
            return s["words"].astype(np.int32)

    def save_words(self, image: str, words: np.ndarray) -> None:
        with self.io_handler.open(self._words_file(image), "wb") as f:
            np.savez_compressed(f, words=words.astype(np.uint16))

    def _matches_path(self) -> str:
        """Return path of matches directory"""
        return os.path.join(self.data_path, "matches")

    def _matches_file(self, image: str) -> str:
        """File for matches for an image"""
        return os.path.join(self._matches_path(), "{}_matches.pkl.gz".format(image))

    def matches_exists(self, image: str) -> bool:
        return self.io_handler.isfile(self._matches_file(image))

    def load_matches(self, image: str) -> Dict[str, np.ndarray]:
        # Prevent pickling of anything except what we strictly need
        # as 'pickle.load' is RCE-prone. Will raise on any class other
        # than the numpy ones we allow.
        class MatchingUnpickler(pickle.Unpickler):
            modules_map = {
                "numpy.core.multiarray._reconstruct": np.core.multiarray,
                "numpy.core.multiarray.scalar": np.core.multiarray,
                "numpy.ndarray": np,
                "numpy.dtype": np,
            }

            def find_class(self, module, name):
                classname = f"{module}.{name}"
                allowed_module = classname in self.modules_map
                if not allowed_module:
                    raise pickle.UnpicklingError(
                        "global '%s.%s' is forbidden" % (module, name)
                    )
                return getattr(self.modules_map[classname], name)

        with self.io_handler.open(self._matches_file(image), "rb") as fin:
            matches = MatchingUnpickler(BytesIO(gzip.decompress(fin.read()))).load()
        return matches

    def save_matches(self, image: str, matches: Dict[str, np.ndarray]) -> None:
        self.io_handler.mkdir_p(self._matches_path())

        with BytesIO() as buffer:
            with gzip.GzipFile(fileobj=buffer, mode="w") as fzip:
                pickle.dump(matches, fzip)
            with self.io_handler.open(self._matches_file(image), "wb") as fw:
                fw.write(buffer.getvalue())

    def find_matches(self, im1: str, im2: str) -> np.ndarray:
        if self.matches_exists(im1):
            im1_matches = self.load_matches(im1)
            if im2 in im1_matches:
                return im1_matches[im2]
        if self.matches_exists(im2):
            im2_matches = self.load_matches(im2)
            if im1 in im2_matches:
                if len(im2_matches[im1]):
                    return im2_matches[im1][:, [1, 0]]
        return np.array([])

    def _tracks_manager_file(self, filename: Optional[str] = None) -> str:
        """Return path of tracks file"""
        return os.path.join(self.data_path, filename or "tracks.csv")

    def load_tracks_manager(
        self, filename: Optional[str] = None
    ) -> pymap.TracksManager:
        """Return the tracks manager"""
        with self.io_handler.open(self._tracks_manager_file(filename), "r") as f:
            return pymap.TracksManager.instanciate_from_string(f.read())

    def tracks_exists(self, filename: Optional[str] = None) -> bool:
        return self.io_handler.isfile(self._tracks_manager_file(filename))

    def save_tracks_manager(
        self, tracks_manager: pymap.TracksManager, filename: Optional[str] = None
    ) -> None:
        with self.io_handler.open(self._tracks_manager_file(filename), "w") as fw:
            fw.write(tracks_manager.as_string())

    def _reconstruction_file(self, filename: Optional[str]) -> str:
        """Return path of reconstruction file"""
        return os.path.join(self.data_path, filename or "reconstruction.json")

    def _reconstruction_file_sampled(self, filename: Optional[str]) -> str:
        """Return path of sampled_reconstruction file"""
        return os.path.join(self.data_path, filename or "reconstruction_sampled.json")

    def reconstruction_exists(self, filename: Optional[str] = None) -> bool:
        return self.io_handler.isfile(self._reconstruction_file(filename))

    def load_reconstruction(
        self, filename: Optional[str] = None
    ) -> List[types.Reconstruction]:
        with self.io_handler.open_rt(self._reconstruction_file(filename)) as fin:
            reconstructions = io.reconstructions_from_json(io.json_load(fin))
        return reconstructions

    def save_reconstruction(
        self,
        reconstruction: List[types.Reconstruction],
        filename: Optional[str] = None,
        minify=False,
    ) -> None:
        
        recons_points = io.reconstructions_to_json(reconstruction)[0]['points'] 
        num_points_perc = 1.0 ## 100 perc
        num_points = round(num_points_perc * len(list(recons_points.keys())))
        random_keys = random.sample(list(recons_points.keys()), num_points)
        sampled_points =  {key: recons_points[key] for key in random_keys}
        sampled_points = [{'points' : sampled_points}]
        
        sample_file_path = '.'.join([self._reconstruction_file_sampled(filename).split('.')[0]]  + ['pkl'])
        
        print()
        print('************** FOR SAVING RECONSTRUCTION ************')
        print(sampled_points)
        print()
        print('Num-Points: ', num_points)
        print()
        print('sample-len: ', sample_file_path)
        print('File-Path: ', self._reconstruction_file(filename))
        print("********************************************")
        print()

        with self.io_handler.open_wt(self._reconstruction_file(filename)) as fout:
            io.json_dump(io.reconstructions_to_json(reconstruction), fout, minify)

        with self.io_handler.open_wt(self._reconstruction_file_sampled(filename)) as fout:
            io.json_dump(sampled_points, fout, minify)

        # with open(sample_file_path, 'wb') as f:
        #     pickle.dump(sampled_points, f)

        logger.info("Saved reconstruction.json and reconstruction_sampled.json successfully!")
        

    def _reference_lla_path(self) -> str:
        return os.path.join(self.data_path, "reference_lla.json")

    def init_reference(self, images: Optional[List[str]] = None) -> None:
        """Initializes the dataset reference if not done already."""
        if not self.reference_exists():
            reference = invent_reference_from_gps_and_gcp(self, images)
            self.save_reference(reference)

    def save_reference(self, reference: geo.TopocentricConverter) -> None:
        reference_lla = {
            "latitude": reference.lat,
            "longitude": reference.lon,
            "altitude": reference.alt,
        }

        with self.io_handler.open_wt(self._reference_lla_path()) as fout:
            io.json_dump(reference_lla, fout)

    def load_reference(self) -> geo.TopocentricConverter:
        """Load reference as a topocentric converter."""
        with self.io_handler.open_rt(self._reference_lla_path()) as fin:
            lla = io.json_load(fin)

        return geo.TopocentricConverter(
            lla["latitude"], lla["longitude"], lla["altitude"]
        )

    def reference_exists(self) -> bool:
        return self.io_handler.isfile(self._reference_lla_path())

    def _camera_models_file(self) -> str:
        """Return path of camera model file"""
        return os.path.join(self.data_path, "camera_models.json")

    def load_camera_models(self) -> Dict[str, pygeometry.Camera]:
        """Return camera models data"""
        with self.io_handler.open_rt(self._camera_models_file()) as fin:
            obj = json.load(fin)
            return io.cameras_from_json(obj)

    def save_camera_models(self, camera_models: Dict[str, pygeometry.Camera]) -> None:
        """Save camera models data"""
        with self.io_handler.open_wt(self._camera_models_file()) as fout:
            obj = io.cameras_to_json(camera_models)
            io.json_dump(obj, fout)

    def _camera_models_overrides_file(self) -> str:
        """Path to the camera model overrides file."""
        return os.path.join(self.data_path, "camera_models_overrides.json")

    def camera_models_overrides_exists(self) -> bool:
        """Check if camera overrides file exists."""
        return self.io_handler.isfile(self._camera_models_overrides_file())

    def load_camera_models_overrides(self) -> Dict[str, pygeometry.Camera]:
        """Load camera models overrides data."""
        with self.io_handler.open_rt(self._camera_models_overrides_file()) as fin:
            obj = json.load(fin)
            return io.cameras_from_json(obj)

    def save_camera_models_overrides(
        self, camera_models: Dict[str, pygeometry.Camera]
    ) -> None:
        """Save camera models overrides data"""
        with self.io_handler.open_wt(self._camera_models_overrides_file()) as fout:
            obj = io.cameras_to_json(camera_models)
            io.json_dump(obj, fout)

    def _exif_overrides_file(self) -> str:
        """Path to the EXIF overrides file."""
        return os.path.join(self.data_path, "exif_overrides.json")

    def exif_overrides_exists(self) -> bool:
        """Check if EXIF overrides file exists."""
        return self.io_handler.isfile(self._exif_overrides_file())

    def load_exif_overrides(self) -> Dict[str, Any]:
        """Load EXIF overrides data."""
        with self.io_handler.open_rt(self._exif_overrides_file()) as fin:
            return json.load(fin)

    def save_exif_overrides(self, exif_overrides: Dict[str, Any]) -> None:
        """Load EXIF overrides data."""
        with self.io_handler.open_wt(self._exif_overrides_file()) as fout:
            io.json_dump(exif_overrides, fout)

    def _rig_cameras_file(self) -> str:
        """Return path of rig models file"""
        return os.path.join(self.data_path, "rig_cameras.json")

    def load_rig_cameras(self) -> Dict[str, pymap.RigCamera]:
        """Return rig models data"""
        all_rig_cameras = rig.default_rig_cameras(self.load_camera_models())
        if not self.io_handler.exists(self._rig_cameras_file()):
            return all_rig_cameras
        with self.io_handler.open_rt(self._rig_cameras_file()) as fin:
            rig_cameras = io.rig_cameras_from_json(json.load(fin))
            for rig_camera_id, rig_camera in rig_cameras.items():
                all_rig_cameras[rig_camera_id] = rig_camera
        return all_rig_cameras

    def save_rig_cameras(self, rig_cameras: Dict[str, pymap.RigCamera]) -> None:
        """Save rig models data"""
        with self.io_handler.open_wt(self._rig_cameras_file()) as fout:
            io.json_dump(io.rig_cameras_to_json(rig_cameras), fout)

    def _rig_assignments_file(self) -> str:
        """Return path of rig assignments file"""
        return os.path.join(self.data_path, "rig_assignments.json")

    def load_rig_assignments(self) -> Dict[str, List[Tuple[str, str]]]:
        """Return rig assignments  data"""
        if not self.io_handler.exists(self._rig_assignments_file()):
            return {}
        with self.io_handler.open_rt(self._rig_assignments_file()) as fin:
            assignments = json.load(fin)

        # Backward compatibility.
        # Older versions of the file were stored as a list of instances without id.
        if isinstance(assignments, list):
            assignments = {str(i): v for i, v in enumerate(assignments)}

        return assignments

    def save_rig_assignments(
        self, rig_assignments: Dict[str, List[Tuple[str, str]]]
    ) -> None:
        """Save rig assignments  data"""
        with self.io_handler.open_wt(self._rig_assignments_file()) as fout:
            io.json_dump(rig_assignments, fout)

    def append_to_profile_log(self, content: str) -> None:
        """Append content to the profile.log file."""
        path = os.path.join(self.data_path, "profile.log")
        with self.io_handler.open(path, "a") as fp:
            fp.write(content)

    def _report_path(self) -> str:
        return os.path.join(self.data_path, "reports")

    def load_report(self, path: str) -> str:
        """Load a report file as a string."""
        with self.io_handler.open_rt(os.path.join(self._report_path(), path)) as fin:
            return fin.read()

    def save_report(self, report_str: str, path: str) -> None:
        """Save report string to a file."""
        filepath = os.path.join(self._report_path(), path)
        self.io_handler.mkdir_p(os.path.dirname(filepath))
        with self.io_handler.open_wt(filepath) as fout:
            return fout.write(report_str)

    def _ground_control_points_file(self) -> str:
        return os.path.join(self.data_path, "ground_control_points.json")

    def _gcp_list_file(self) -> str:
        return os.path.join(self.data_path, "gcp_list.txt")

    def load_ground_control_points(self) -> List[pymap.GroundControlPoint]:
        """Load ground control points."""
        exif = {image: self.load_exif(image) for image in self.images()}

        gcp = []
        if self.io_handler.isfile(self._gcp_list_file()):
            with self.io_handler.open_rt(self._gcp_list_file()) as fin:
                gcp = io.read_gcp_list(fin, exif)

        pcs = []
        if self.io_handler.isfile(self._ground_control_points_file()):
            with self.io_handler.open_rt(self._ground_control_points_file()) as fin:
                pcs = io.read_ground_control_points(fin)

        return gcp + pcs

    def save_ground_control_points(
        self,
        points: List[pymap.GroundControlPoint],
    ) -> None:
        with self.io_handler.open_wt(self._ground_control_points_file()) as fout:
            io.write_ground_control_points(points, fout)

    def image_as_array(self, image: str) -> np.ndarray:
        logger.warning("image_as_array() is deprecated. Use load_image() instead.")
        return self.load_image(image)


def invent_reference_from_gps_and_gcp(
    data: DataSetBase, images: Optional[List[str]] = None
) -> geo.TopocentricConverter:
    """ Invent the reference from the weighted average of lat/lon measurements.
    Most of the time the altitude provided in the metadata is inaccurate, thus
    the reference altitude is set equal to 0 regardless of the altitude measurements.
    """
    lat, lon = 0.0, 0.0
    wlat, wlon = 0.0, 0.0
    if images is None:
        images = data.images()
    for image in images:
        d = data.load_exif(image)
        if "gps" in d and "latitude" in d["gps"] and "longitude" in d["gps"]:
            w = 1.0 / max(0.01, d["gps"].get("dop", 15))
            lat += w * d["gps"]["latitude"]
            lon += w * d["gps"]["longitude"]
            wlat += w
            wlon += w

    if not wlat and not wlon:
        for gcp in data.load_ground_control_points():
            if gcp.lla:
                lat += gcp.lla["latitude"]
                lon += gcp.lla["longitude"]
                wlat += 1.0
                wlon += 1.0

    if wlat:
        lat /= wlat
    if wlon:
        lon /= wlon

    return geo.TopocentricConverter(lat, lon, 0)

def save_point_cloud(
        self,
        points: np.ndarray,
        normals: np.ndarray,
        colors: np.ndarray,
        labels: np.ndarray,
        filename: str = "sparse_pc.ply",
    ) -> None:
        self.io_handler.mkdir_p(self._depthmap_path())
        with self.io_handler.open(self.point_cloud_file(filename), "w") as fp:
            io.point_cloud_to_ply(points, normals, colors, labels, fp)
