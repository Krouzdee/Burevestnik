import cv2
import time
import numpy as np
from djitellopy import Tello

camera_matrix = np.array([[920, 0, 480], [0, 920, 360], [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros((5, 1), dtype=np.float32)

marker_length = 0.1

tello = Tello()
tello.connect()
print(f"Battery: {tello.get_battery()}%")
tello.streamon()
time.sleep(2)

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

obj_points = np.array(
    [
        [-marker_length / 2,  marker_length / 2, 0],
        [ marker_length / 2,  marker_length / 2, 0],
        [ marker_length / 2, -marker_length / 2, 0],
        [-marker_length / 2, -marker_length / 2, 0],
    ],
    dtype=np.float32,
)


def find_markers(current_marker_id, frame):
    if frame is None:
        return None, None

    frame = cv2.flip(frame, 1)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    corners, ids, rejected = detector.detectMarkers(gray)

    marker_pose = None

    if ids is not None:
        ids = ids.flatten()

        for i, marker_id in enumerate(ids):
            if marker_id == current_marker_id:

                cv2.aruco.drawDetectedMarkers(
                    frame,
                    [corners[i]],
                    np.array([[marker_id]])
                )

                success, rvec, tvec = cv2.solvePnP(
                    obj_points,
                    corners[i],
                    camera_matrix,
                    dist_coeffs,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE
                )

                if success:
                    cv2.drawFrameAxes(
                        frame,
                        camera_matrix,
                        dist_coeffs,
                        rvec,
                        tvec,
                        length=marker_length * 0.5
                    )

                    x, y, z = tvec.flatten()
                    marker_pose = (float(x), float(y), float(z))

                break

    return frame, marker_pose


current_id = 0

while True:
    frame_raw = tello.get_frame_read().frame
    frame, pose = find_markers(current_id, frame_raw)
    if pose is not None:
        x, y, z = pose
        print(f"\r Нашли {current_id}", f"Координаты: x: {x}, y: {y}, z: {z}", end="", flush=True)

    if frame is not None:
        cv2.imshow("Tello ArUco", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

tello.streamoff()
tello.end()
cv2.destroyAllWindows()