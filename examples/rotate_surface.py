import pathlib

import cv2
import helpers
from pupil_labs.marker_mapper import Surface


def main():
    recording_dir = pathlib.Path("/home/marc/Downloads/recording")
    camera, marker_detector, frames = helpers.setup(recording_dir)

    # Define an example surface based on markers detected in the first frame
    frame = next(iter(frames))
    markers = marker_detector.detect(frame.gray)
    surface = Surface.from_apriltag_detections("test surface", markers, camera)

    for frame in frames:
        markers = marker_detector.detect(frame.gray)
        localization = surface.localize(markers, camera)
        helpers.visualize_results(camera, frame, markers, surface, localization)

        key = cv2.waitKey(1)
        if key == ord("r"):
            surface.rotate()


if __name__ == "__main__":
    main()
