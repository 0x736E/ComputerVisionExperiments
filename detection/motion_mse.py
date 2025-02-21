import cv2
import numpy as np

class MotionMaskMSE:

    """
    The MotionMaskMSE class is used for detection of motion by calculating the mean squared error (MSE)
    between consecutive video frames. It preprocesses the frames by resizing, converting to grayscale,
    and applying Gaussian blur before computing the MSE.

    This class is useful in applications requiring motion detection, such as surveillance systems or
    video analysis tools. It operates on frames provided during updates and calculates the difference
    in intensity between these processed frames for detecting motion.

    This approach is fast and efficient, but does not detect motion in all cases. It is well suited for
    determining whether a frame contains significant motion, such as surveillance systems or video analysis.
    Which can be used to trigger actions or perform other more computationally expensive tasks such as more
    advanced motion detection or object tracking.
    """


    def __init__(self):
        self._prev_frame = None
        self._mse_value = 0
        self._blur_kernel = (5,5)
        self._downsample_resolution = (480, 270)
        pass

    def get_difference(self):
        return self._mse_value

    def _calc_difference(self, frame):

        frame = self._downsample(frame).copy()

        self._mse_value = np.mean(
            cv2.absdiff(self._prev_frame, frame) ** 2
        )

        self._prev_frame = frame.copy()
        return None

    def _downsample(self, frame):
        return cv2.GaussianBlur(
            cv2.resize(
                cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2GRAY
                ), self._downsample_resolution
        ), self._blur_kernel, 0)

    def update(self, frame):

        if self._prev_frame is None:
            self._prev_frame = self._downsample(frame).copy()
            return None

        self._calc_difference(frame)
        return None

