import math
import yaml
import json
import cv2
import numpy as np
from typing import Tuple
from abc import ABC, abstractmethod
from typing import Dict
from typing import Tuple
from dataclasses import dataclass, field

EARTH_RADIUS = 6371.0  # km
EARTH_CIRCUMFERENCE = 2 * math.pi * EARTH_RADIUS  # km
EQUTORIAL_CIRCUMFERENCE_METERS = 40075016.686  # meters
TILE_SIZE = 256  # pixels
MAX_ZOOM_LEVEL = 20  # maximum zoom level
MIN_ZOOM_LEVEL = 1  # minimum zoom level

# 计算给定纬度和缩放级别的地图分辨率（米/像素）
def resolution_at_zoom_level(lat: float, zoom_level: int, tile_size: int = TILE_SIZE) -> float:
    """
    Calculate the spatial resolution at a given latitude and zoom level.


    Parameters
    ----------
    lat : float
        latitude
    zoom_level : int
        zoom level
    tile_size : int
        size of the tile

    Returns
    -------
    float
        spatial resolution
    """

    equator_circumference = EQUTORIAL_CIRCUMFERENCE_METERS
    resolution_at_zoom_0 = equator_circumference / tile_size
    return resolution_at_zoom_0 * math.cos(math.radians(lat)) / 2**zoom_level

# 把瓦片坐标转换成全局像素坐标，常用于地图显示、坐标计算、影像拼接等应用
def get_xy_pixel_from_xy_tile(x: int, y: int, origin_x: int, origin_y: int, tile_size: int) -> Tuple[int, int]:
    """
    Convert x, y tile coordinates to x, y pixel coordinates. The origin is the top left corner of the tile.


    Parameters
    ----------
    x : int
        x tile coordinate
    y : int
        y tile coordinate
    origin_x : int
        x tile origin
    origin_y : int
        y tile origin
    tile_size : int
        size of the tile

    Returns
    -------
    Tuple[int, int]
        x, y pixel coordinates
    """

    x_pixel = (x - origin_x) * tile_size
    y_pixel = (y - origin_y) * tile_size
    return x_pixel, y_pixel

# 将地图的瓦片坐标转换成实际的经纬度坐标
def get_lat_long_from_tile_xy(x: int, y: int, zoom_level: int) -> Tuple[float, float]:
    """
    Convert x, y tile coordinates to lat, long coordinates.

    Parameters
    ----------
    x : int
        x tile coordinate
    y : int
        y tile coordinate
    zoom_level : int
        zoom level

    Returns
    -------
    Tuple[float, float]
        lat, long coordinates
    """

    n = 2.0**zoom_level
    lon_deg = x / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    lat_deg = math.degrees(lat_rad)

    assert -90 <= lat_deg <= 90, "The latitude must be between -90 and 90."
    assert -180 <= lon_deg <= 180, "The longitude must be between -180 and 180."

    return lat_deg, lon_deg

# 将地理坐标（纬度、经度）转换为瓦片坐标（x, y），用于地图分块
def get_tile_xy_from_lat_long(lat: float, long: float, zoom_level: int) -> Tuple[int, int]:
    """
    Convert lat, long coordinates to x, y tile coordinates.
    Parameters
    ----------
    lat : float
        latitude
    long : float
        longitude
    zoom_level : int
        zoom level
    Returns
    -------
    Tuple[int, int]
        x, y tile coordinates
    """

    x = int((long + 180) / 360 * 2**zoom_level)
    y = int(
        (
            1
            - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat)))
            / math.pi
        )
        / 2
        * 2**zoom_level
    )
    assert 1 <= zoom_level <= 20, "The zoom level must be between 1 and 20."

    MAX_XY = 2 ** zoom_level

    assert x < 0 or x >= MAX_XY or y < 0 or y >= MAX_XY, "The tile index must be between 0 and 2^zoom_level."

    return x, y


class Parser(ABC):
    """Abstract class for parsers."""

    @staticmethod
    @abstractmethod
    def load(file: str) -> Dict:
        """Load data from a file.

        Parameters
        ----------
        file : str
            file path

        Returns
        -------
        Dict
            data
        """
        pass

    @staticmethod
    @abstractmethod
    def dump(data: Dict, file: str) -> None:
        """Dump data to a file.

        Parameters
        ----------
        data : Dict
            data to dump
        file : str
            file path
        """
        pass

class YamlParser(Parser):
    """YAML parser."""

    @staticmethod
    def load_yaml(yaml_file: str) -> Dict:
        with open(yaml_file, "r") as file:
            return yaml.load(file, Loader=yaml.FullLoader)

    @staticmethod
    def dump_yaml(data: Dict, yaml_file: str) -> None:
        with open(yaml_file, "w") as file:
            yaml.dump(data, file)

class JsonParser(Parser):
    """JSON parser."""

    @staticmethod
    def load_json(json_file: str) -> Dict:
        with open(json_file, "r") as file:
            return json.load(file)

    @staticmethod
    def dump_json(data: Dict, json_file: str) -> None:
        with open(json_file, "w") as file:
            json.dump(data, file)
@dataclass
class CameraModel:
    """A CameraModel is a dataclass that represents the intrinsic parameters of a camera.

    Parameters
    ----------
    focal_length（焦距（毫米）） : float
        focal length of the camera in millimeters
    resolution_width（图像的分辨率（像素）） : int
        width of the image in pixels
    resolution_height（图像的分辨率（像素）） : int
        height of the image in pixels
    hfov_deg（水平视场角（度）） : float
        horizontal field of view in degrees
    principal_point_x（相机的主点坐标，默认为图像的中心） : float
        x coordinate of the principal point
    principal_point_y（相机的主点坐标，默认为图像的中心） : float
        y coordinate of the principal point

    Properties
    ----------
    hfov_rad（水平视场角（弧度），通过 hfov_deg 计算得出） : float
        horizontal field of view in radians
    resolution（图像的分辨率，作为一个 (width, height) 的元组） : Tuple
        resolution of the image
    aspect_ratio（图像的宽高比：resolution_width / resolution_height） : float
        aspect ratio of the image
    focal_length_px（焦距（像素）通过分辨率和视场角计算得出） : float
        focal length in pixels
    """

    focal_length: float
    resolution_width: int
    resolution_height: int
    hfov_deg: float
    hfov_rad: float = field(init=False)
    resolution: Tuple = field(init=False)
    aspect_ratio: float = field(init=False)
    focal_length_px: float = field(init=False)
    principal_point_x: float = None
    principal_point_y: float = None

    def __post_init__(self) -> None:
        self.hfov_rad = self.hfov_deg * (math.pi / 180)
        self.resolution = (self.resolution_width, self.resolution_height)
        self.aspect_ratio = self.resolution_width / self.resolution_height
        self.focal_length_px = self.resolution_width / (2 * math.tan(self.hfov_rad / 2))
        if self.principal_point_x is None:
            self.principal_point_x = self.resolution_width / 2
        if self.principal_point_y is None:
            self.principal_point_y = self.resolution_height / 2

    @staticmethod
    def from_yaml(yaml_file: str) -> "CameraModel":
        data = YamlParser.load_yaml(yaml_file)
        return CameraModel(
            focal_length=data["focal_length"],
            resolution_width=data["resolution_width"],
            resolution_height=data["resolution_height"],
            hfov_deg=data["hfov_deg"],
        )

    @staticmethod
    def from_json(json_file: str) -> "CameraModel":
        data = JsonParser.load_json(json_file)
        return CameraModel(
            focal_length=data["focal_length"],
            resolution_width=data["resolution_width"],
            resolution_height=data["resolution_height"],
            hfov_deg=data["hfov_deg"],
        )

# 获取相机内参
def get_intrinsics(camera_model: CameraModel, scale: float = 1.0) -> np.ndarray:
    """
    Get the intrinsics matrix of a camera model.

    Parameters
    ----------
    camera_model : CameraModel
        the camera model
    scale : float
        the scale factor to apply to the focal length, default is 1.0

    Returns
    -------
    np.ndarray
        the intrinsics matrix
    """
    intrinsics = np.array(
        [
            [camera_model.focal_length_px / scale, 0, camera_model.principal_point_x],
            [0, camera_model.focal_length_px / scale, camera_model.principal_point_y],
            [0, 0, 1],
        ]
    )
    return intrinsics

# 通过给定的滚转（roll）、俯仰（pitch）和偏航（yaw）角度，计算出对应的旋转矩阵
def rotation_matrix_from_angles(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Compute the rotation matrix from the roll, pitch, and yaw angles.

    Parameters
    ----------
    roll : float
        the roll angle
    pitch : float
        the pitch angle
    yaw : float
        the yaw angle

    Returns
    -------
    np.ndarray
        the rotation matrix
    """
    from scipy.spatial.transform import Rotation

    r = Rotation.from_euler("xyz", [roll, pitch, yaw], degrees=True).as_matrix()
    return r

# 计算一个缩放比例，用来将查询图像的分辨率调整到与卫星图像相同的分辨率
def compute_resize_scale(
    self,
    camera_model: CameraModel,
    altitude: float,
    satellite_resolution: float,
) -> float:
    """Compute the resize scale to resize the query image to the same resolution
    as the satellite image.

    Parameters
    ----------
    camera_model : CameraModel
        the camera model of the query image
    altitude : float
        the altitude of the query image
    satellite_resolution : float
        the resolution of the satellite image

    Returns
    -------
    float
        the resize scale
    """

    # hvof_m = 2 * altitude * np.tan(hfov / 2)
    hvof_m = 2 * altitude * np.tan(camera_model.hfov_rad / 2)
    # resolution = hvof_m / width
    drone_resolution = hvof_m / camera_model.resolution_width
    # Compute the resize scale
    resize_scale = drone_resolution / satellite_resolution
    return resize_scale

# Warp the query image so that it is aligned with the satellite image
def warp_image(query, camera_model, satellite_resolution, yaw, altitude):
    """
    Parameters
    ----------
    query:无人机图像
    camera_model：相机模型
    satellite_resolution：卫星图分辨率
    yaw：偏航角
    altitude: 高度

    Returns
    -------

    """
    # Check if the camera model is provided either in the query or externally
    if  camera_model is None:
        raise ValueError("Camera model is missing in the query and in the processor.")

    # Set the camera model
    query_camera_model= camera_model

    # Get the camera's intrinsic parameters
    K = get_intrinsics(query_camera_model)

    # Rotation matrices based on camera and drone orientation
    R_gimbal = rotation_matrix_from_angles(roll=0, pitch=0, yaw=yaw)
    R_drone = rotation_matrix_from_angles(roll=0, pitch=0, yaw=yaw + 15)
    R_target = rotation_matrix_from_angles(0, 0, 0)

    # Compute the scale factor if satellite resolution is provided
    if satellite_resolution is not None:
        scale = compute_resize_scale(query_camera_model, altitude, satellite_resolution)
    else:
        scale = 1.0

    # Get the scaled intrinsics matrix
    K_scale = get_intrinsics(query_camera_model, scale)

    # Compute the transformation matrix
    transformation_matrix = (
            K
            @ np.linalg.inv(R_gimbal)
            @ np.linalg.inv(R_target)
            @ R_drone
            @ R_gimbal
            @ np.linalg.inv(K_scale)
    )

    # Get the image dimensions
    height, width = query.shape[:2]

    # Define homogeneous coordinates for the image corners
    corners = np.array(
        [
            [0, 0, 1],
            [width - 1, 0, 1],
            [width - 1, height - 1, 1],
            [0, height - 1, 1],
        ]
    ).T

    # Apply the transformation to the corners
    warped_corners = transformation_matrix @ corners

    # Normalize the coordinates by the third component (homogeneous coordinates)
    warped_corners /= warped_corners[2]

    # Remove the third component (flatten to 2D)
    warped_corners = warped_corners[:2].T

    # Move corners to the origin
    warped_corners -= warped_corners.min(axis=0)

    # Remove the third component from original corners
    corners = corners[:2].T

    # Cast corners to float32 type for cv2 processing
    corners = corners.astype(np.float32)
    warped_corners = warped_corners.astype(np.float32)

    # Compute the transformation matrix for perspective warp
    dst = cv2.getPerspectiveTransform(corners, warped_corners)

    # Compute the new image size (max corner values)
    new_size = warped_corners.max(axis=0).astype(np.int32)

    # Warp the image based on the computed transformation matrix
    warped_image = cv2.warpPerspective(query, dst, tuple(new_size))

    query = warped_image
    return query

# 举一个例子
# camera_model = CameraModel(
#     focal_length=4.5 / 1000,  # 4.5mm
#     resolution_height=4056,
#     resolution_width=3040,
#     hfov_deg=82.9,
# )