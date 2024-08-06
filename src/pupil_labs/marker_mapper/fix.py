import cv2
from pupil_labs.camera import CameraRadial
import numpy as np


def undistort_image(image, camera: CameraRadial):
    return cv2.undistort(
        image,
        camera.camera_matrix,
        camera.distortion_coefficients,
        newCameraMatrix=camera.camera_matrix,
    )


def perspectiveTransform(points, transform):
    points = points.reshape(-1, 1, 2).astype(np.float32)
    points_trans = cv2.perspectiveTransform(points, transform)
    points_trans = points_trans.reshape(-1, 2)
    return points_trans
