import pathlib

import cv2
import numpy as np
import pupil_apriltags
import pupil_labs.neon_recording as nr
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

    move_corner(camera, markers, surface)

    for frame in frames:
        markers = marker_detector.detect(frame.gray)
        localization = surface.localize(markers, camera)
        helpers.visualize_results(camera, frame, markers, surface, localization)
        cv2.waitKey(1)


def move_corner(camera, markers, surface):
    img2surface, surface2image = surface.localize(markers, camera)
    current_corner_pos_undist = fix.perspectiveTransform(
        np.array([1, 0]), surface2image
    )
    current_corner_pos_dist = camera.distort_points_on_image_plane(
        current_corner_pos_undist
    )
    new_corner_pos = current_corner_pos_dist + 200
    surface.move_corner(1, new_corner_pos, img2surface, camera)


if __name__ == "__main__":
    main()
