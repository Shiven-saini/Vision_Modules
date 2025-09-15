import cv2 as cv

class ArucoDetector:
    def __init__(self, dictionary=cv.aruco.DICT_6X6_250):
        self.dictionary = cv.aruco.getPredefinedDictionary(dictionary)
        self.parameters = cv.aruco.DetectorParameters()

    def detect(self, image):
        detector = cv.aruco.ArucoDetector(self.dictionary, self.parameters)
        corners, ids, _ = detector.detectMarkers(image)
        return corners, ids
