# rvm/markers/aruco.py
import cv2
from typing import List

from rvm.core.types import Marker, QRCode


class ArucoDetector:
    """ArUco marker and QR code detector."""

    def __init__(self, dictionary=cv2.aruco.DICT_6X6_250):
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary)
        self.parameters = cv2.aruco.DetectorParameters()

    def detect(self, image) -> List[Marker]:
        """
        Detect ArUco markers in the image.

        Args:
            image (np.ndarray): Input BGR image.

        Returns:
            List[Marker]: List of detected markers with IDs and corners.
        """
        detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)
        corners, ids, _ = detector.detectMarkers(image)
        markers: List[Marker] = []

        if ids is not None:
            for i, corner in enumerate(corners):
                points = [(int(x), int(y)) for x, y in corner[0]]
                markers.append(Marker(id=int(ids[i][0]), corners=points))
        return markers

    def detect_qr(self, image) -> List[QRCode]:
        """
        Detect QR codes in the image.

        Args:
            image (np.ndarray): Input BGR image.

        Returns:
            List[QRCode]: List of QR codes with data and corners.
        """
        detector = cv2.QRCodeDetector()
        retval, decoded_info, points, _ = detector.detectAndDecodeMulti(image)
        qrs: List[QRCode] = []

        if retval and points is not None:
            for data, pts in zip(decoded_info, points):
                corners = [(int(x), int(y)) for x, y in pts]
                qrs.append(QRCode(data=data, corners=corners))
        return qrs
