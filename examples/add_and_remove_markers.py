import pathlib

import cv2
import numpy as np
import pupil_apriltags
import pupil_labs.neon_recording as nr
from pupil_labs.camera import CameraRadial
from pupil_labs.marker_mapper import Surface, fix, utils

import helpers

new_corner_pos = None


def main():
    recording_dir = pathlib.Path("/home/marc/Downloads/recording")
    camera, marker_detector, frames = helpers.setup(recording_dir)

    # Define an example surface based on markers detected in the first frame
    frame = next(iter(frames))
    markers = marker_detector.detect(frame.gray)
    surface = Surface.from_apriltag_detections("test surface", markers, camera)

    surface.remove_marker(24)
    img2surface, surface2image = surface.localize(markers, camera)
    surface.add_marker(markers[0], camera, img2surface)

    for frame in frames:
        markers = marker_detector.detect(frame.gray)
        localization = surface.localize(markers, camera)
        helpers.visualize_results(camera, frame, markers, surface, localization)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
