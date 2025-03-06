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

    def __init__(self, threshold=None, region_multiplier=4, break_on_threshold=False):
        self._prev_frame = None
        self._avg_mse_value = 0
        self._mse_values = np.zeros(region_multiplier * region_multiplier)
        self._mse_threshold_values = np.array([])
        self._blur_kernel = (5, 5)
        self._downsample_resolution = (480, 270)
        self._region_multiplier = region_multiplier
        self._threshold = threshold
        self._break_on_threshold = break_on_threshold
        self._regions = np.zeros((self._region_multiplier * self._region_multiplier, 4), np.uint8)
        self._diff = np.zeros((self._region_multiplier * self._region_multiplier, 1), np.uint8)

    def get_average(self):
        return self._avg_mse_value

    def get_values(self):
        return self._mse_values

    def _calc_difference(self, frame):

        if self._prev_frame is None:
            self._prev_frame = self._downsample(frame)
            self._avg_mse_value = 0
            self._mse_values = np.zeros(self._region_multiplier * self._region_multiplier)

            # init regions
            height, width = self._prev_frame.shape
            region_height = height // self._region_multiplier
            region_width = width // self._region_multiplier
            self._regions = np.zeros((self._region_multiplier * self._region_multiplier, 4), np.uint8)
            self._diff = np.zeros((region_height, region_width), np.uint8)
            for index in range(len(self._mse_values)):
                i = index // self._region_multiplier
                j = index % self._region_multiplier

                self._regions[index] = [
                    i * region_height,          # y start
                    (i + 1) * region_height,    # y end
                    j * region_width,           # x start
                    (j + 1) * region_width      # x end
                ]
            return None

        frame = self._downsample(frame)

        for r_index, region in enumerate(self._regions):

            self._diff = cv2.absdiff(
                self._prev_frame[
                    region[0]:region[1],
                    region[2]:region[3]
                ],
                frame[
                    region[0]:region[1],
                    region[2]:region[3]
                ]
            )

            if self._diff is None:
                continue

            self._mse_values[r_index] = np.mean(self._diff ** 2)

            if (self._threshold is not None
                and self._mse_values[r_index] > self._threshold
                and self._break_on_threshold):
                break

        self._avg_mse_value = np.mean(self._mse_values)
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
        self._calc_difference(frame)
        return None

