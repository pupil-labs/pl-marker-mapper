import pathlib

import cv2
import numpy as np
import pupil_apriltags
import pupil_labs.neon_recording as nr
from pupil_labs.camera import CameraRadial
from pupil_labs.marker_mapper import Surface, fix, utils


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

    surface.remove_marker(24)
    img2surface, surface2image = surface.localize(markers, camera)
    surface.add_marker(markers[0], camera, img2surface)

    for frame in frames:
        markers = marker_detector.detect(frame.gray)
        localization = surface.localize(markers, camera)

        orig_img = frame.bgr
        undist_img = fix.undistort_image(frame.bgr, camera)

        if localization is not None:
            img2surface, surface2image = localization

            vertices_dist = np.array([m.corners for m in markers])
            vertices_undist = Surface._get_undist_vertices(markers, camera).reshape(
                -1, 4, 2
            )

            marker_ids = [m.tag_id for m in markers]
            orig_img = draw_markers(orig_img, marker_ids, vertices_dist, surface)
            undist_img = draw_markers(undist_img, marker_ids, vertices_undist, surface)

            surface_boundary_undist = utils.get_surface_boundary(surface2image)
            draw_surface(undist_img, surface_boundary_undist)

            surface_boundary_dist = utils.get_surface_boundary(
                surface2image, distorted=True, camera=camera
            )
            draw_surface(orig_img, surface_boundary_dist)

            crop = utils.crop_image(undist_img, surface2image, width=500, height=None)

            cv2.imshow("Cropped Image", crop)
        cv2.imshow("Undistorted Image", undist_img)
        cv2.imshow("Distorted Image", orig_img)

        key = cv2.waitKey(0)
        if key == ord("r"):
            surface.rotate()


def get_cam(rec: nr.neon_recording.NeonRecording):
    intrinsics = rec.scene_camera_calibration
    camera_matrix = intrinsics.camera_matrix[0]
    dist_coeffs = intrinsics.distortion_coefficients[0]
    return CameraRadial(1600, 1200, camera_matrix, dist_coeffs)


def draw_markers(img, marker_ids, marker_verts, surface):
    included_color = (0, 255, 0)
    excluded_color = (0, 0, 255)

    overlay = img.copy()
    for m_id, vert in zip(marker_ids, marker_verts):
        color = included_color if m_id in surface.markers.keys() else excluded_color
        cv2.fillPoly(overlay, [vert.astype(int)], color)

    alpha = 0.3
    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    return img


def draw_surface(img, boundary_points):
    cv2.polylines(
        img,
        [boundary_points.astype(int)],
        True,
        (255, 0, 0),
        2,
    )
    cv2.polylines(
        img,
        [boundary_points[:10].astype(int)],
        False,
        (0, 0, 255),
        2,
    )
    top_center = boundary_points[:10].mean(axis=0).astype(int)
    cv2.putText(
        img,
        "Top",
        tuple(top_center),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )


if __name__ == "__main__":
    main()
