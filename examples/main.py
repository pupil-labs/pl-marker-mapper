import pathlib

import cv2
import numpy as np
import pupil_apriltags
import pupil_labs.neon_recording as nr
from pupil_labs.camera import CameraRadial
from pupil_labs.marker_mapper import Surface, fix, utils


def get_cam(rec: nr.neon_recording.NeonRecording):
    intrinsics = rec.scene_camera_calibration
    camera_matrix = intrinsics.camera_matrix[0]
    dist_coeffs = intrinsics.distortion_coefficients[0]
    return CameraRadial(1600, 1200, camera_matrix, dist_coeffs)


def main():
    recording_dir = pathlib.Path("/home/marc/Downloads/recording")
    recording = nr.load(recording_dir)
    camera = get_cam(recording)

    marker_detector = pupil_apriltags.Detector(
        families="tag36h11",
        nthreads=1,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0,
    )

    frames = recording.scene.sample(recording.scene.ts)

    # Define an example surface based on markers detected in the first frame
    frame = next(iter(frames))
    markers = marker_detector.detect(frame.gray)
    surface = Surface.from_apriltag_detections("test surface", markers, camera)

    for frame in frames:
        markers = marker_detector.detect(frame.gray)
        localization = surface.localize(markers, camera)

        orig_img = frame.bgr
        undist_image = fix.undistort_image(frame.bgr, camera)

        if localization is not None:
            img2surface, surface2image = localization

            vertices_dist = np.array([m.corners for m in markers]).reshape(-1, 2)
            for p in vertices_dist.astype(int):
                cv2.circle(orig_img, tuple(p), 3, (0, 255, 0), -1)

            vertices_undist = Surface._get_undist_vertices(markers, camera)
            for p in vertices_undist.astype(int):
                cv2.circle(undist_image, tuple(p), 3, (0, 255, 0), -1)

            surface_boundary_undist = utils.get_surface_boundary(surface2image)
            cv2.polylines(
                undist_image,
                [surface_boundary_undist.astype(int)],
                True,
                (0, 0, 255),
                2,
            )

            surface_boundary_dist = utils.get_surface_boundary(
                surface2image, distorted=True, camera=camera
            )
            cv2.polylines(
                orig_img, [surface_boundary_dist.astype(int)], True, (0, 0, 255), 2
            )

            crop = utils.crop_image(undist_image, surface2image, width=500, height=None)

            cv2.imshow("Cropped Image", crop)
        cv2.imshow("Undistorted Image", undist_image)
        cv2.imshow("Distorted Image", orig_img)
        cv2.waitKey(0)


if __name__ == "__main__":
    main()
