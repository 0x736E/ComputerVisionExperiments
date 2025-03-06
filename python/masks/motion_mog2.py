import cv2
import numpy as np

'''
    This class is used for motion detection using the MOG2 background subtractor.
    It preprocesses the frames by resizing, converting to grayscale,
    and applying Gaussian blur before computing the MSE. It also applies
    adaptive thresholding to reduce noise and fill in the contours.
    The resulting mask is then used for further processing such as object detection.
    
    This approach is fast and efficient, but does not detect motion in all cases.
    It is well suited for determining whether a frame contains significant motion,
    such as surveillance systems or video analysis. Which can be used to trigger
    actions or perform other more computationally expensive tasks such as more
    advanced motion detection or object tracking.
    
    The motion mask is available and can be used for further processing, such as
    drawing overlays or applying masks to the source frames.
'''


class MotionMaskMOG2:

    def __init__(self, src=None, resolution=(480, 270)):
        self._mask = np.zeros((resolution[1], resolution[0]), np.uint8)
        self._downsample_resolution = resolution

        self._back_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=400,
            varThreshold=50,
            detectShadows=True
        )

        self._adapt_thresh_conf = [
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            21,
            16
        ]

        self._mask_history = np.array([])

        self._erode_kernel = np.ones((2, 2), np.uint8)
        self._dilate_kernel = np.ones((2, 2), np.uint8)
        self._motion_trails_erode_kernel = np.ones((5, 5), np.uint8)
        self._close_shapes_kernel = np.ones((5, 5), np.uint8)
        self._downsample_blur_kernel = (5,5)

        self._max_motion_trail = 10
        self._min_motion_area = 400
        self._contour_fill_color = (255, 255, 255)

        if src is not None:
            self.update(src)

    def get_mask(self):
        return self._mask

    def _downsample(self, src):
        np.copyto(self._mask, cv2.blur(
            cv2.cvtColor(
                cv2.resize(src,
                           self._downsample_resolution,
                           interpolation=cv2.INTER_LINEAR),
                cv2.COLOR_BGR2GRAY
            ),
            self._downsample_blur_kernel
        ))
        return None

    def _remove_background(self):

        np.copyto(self._mask,                                       # step 6 - numpy copy
          cv2.morphologyEx(                                         # step 5 - Enclosure
              cv2.dilate(                                           # step 4 - Dilation
                  cv2.erode(                                        # step 3 - Erosion
                      cv2.adaptiveThreshold(                        # step 2 - Adaptive Threshold
                        self._back_subtractor.apply(self._mask),    # step 1 - MOG2
                        self._adapt_thresh_conf[0],
                        self._adapt_thresh_conf[1],
                        self._adapt_thresh_conf[2],
                        self._adapt_thresh_conf[3],
                        self._adapt_thresh_conf[4]
                      ),
                      self._erode_kernel),
            self._dilate_kernel),
            cv2.MORPH_CLOSE,
            self._close_shapes_kernel
        ))

        return None

    def _draw_contour_fill(self):
        # Note: If the contours touch the end of the image
        # then the contours there wont be properly closed
        # TODO: close contours touching edge of image
        contours, _ = cv2.findContours(
            self._mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(
            self._mask,
            contours,
            -1,
            self._contour_fill_color,
            -1
        )
        return None

    def _draw_motion_trails(self):
        if self._max_motion_trail > 0:
            if len(self._mask_history) > self._max_motion_trail:
                self._mask_history = np.delete(self._mask_history, 0)
            np.append(self._mask_history, self._mask)
            for pmask in self._mask_history:
                np.copyto(cv2.erode(
                    pmask, self._motion_trails_erode_kernel
                ), self._mask)
                # mog_threshold = cv2.addWeighted(mog_threshold, 0.5, pmask, 1, 0)
        return None

    def update(self, src):

        if src is None:
            # skip without error
            return None

        self._downsample(src)
        self._remove_background()
        self._draw_contour_fill()
        # self._draw_motion_trails()
        return self._mask
