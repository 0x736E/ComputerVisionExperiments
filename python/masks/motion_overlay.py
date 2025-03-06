import cv2
import numpy as np

'''
    This class is used to draw bounding boxes around motion areas.
    It can be used to draw bounding boxes around areas of motion,
    overlay the motion mask onto the source image, or draw both.
    
    The bounding boxes can be used for further processing such as
    object detection, tracking, or other more advanced actions.
    
    The bounding boxes are available and can be used for further processing,
    such as drawing overlays or applying masks to the source frames.
'''


INVALID_TUPLE = (-1,-1)

class MotionOverlayMode:
    NONE = 0
    BOUNDING_BOX = 1
    OVERLAY = 2
    BOUNDING_BOX_OVERLAY = 3

class MotionOverlayMask:

    def __init__(self, mode=MotionOverlayMode.BOUNDING_BOX_OVERLAY, src=None, mask=None):

        self._overlay_mode = mode
        self._overlay_mask = None
        self._motion_mask = None
        self._overlay_mask = None
        self._rgb_mask = None
        self._total_area = 0
        self._src_resolution = INVALID_TUPLE
        self._mask_resolution = INVALID_TUPLE
        self._ratio_res = INVALID_TUPLE
        self._bounding_boxes = np.array([])
        self._contours = np.array([])
        self._set_images(src, mask)
        self._mask_overlay_color = (0, 0, 255)
        self._bounding_box_color = (0,255,0)
        self._bounding_box_thickness = 2
        self._min_motion_area = 200
        self._mask_overlay_config = [
            1.0,
            1.0,
            0
        ]

        if (self._overlay_mask is not None) and (self._motion_mask is not None):
            self.update(self._overlay_mask, self._motion_mask)

    def mode(self, mode=MotionOverlayMode.NONE):
        self._overlay_mode = mode
        return None

    def get_mask(self):
        return self._motion_mask

    def get_overlay_mask(self):
        return self._overlay_mask

    def get_bounding_boxes(self):
        return self._bounding_boxes

    def _set_images(self, src=None, mask=None):
        self._total_area = 0

        if self._overlay_mode == MotionOverlayMode.NONE:
            self._overlay_mask = None
            self._motion_mask = None
            self._rgb_mask = None
            self._src_resolution = INVALID_TUPLE
            self._mask_resolution = INVALID_TUPLE
            self._ratio_res = INVALID_TUPLE
            return None

        if (src is None) or (mask is None):
            return None

        # Create new array or resize existing one to match source dimensions
        if self._overlay_mask is None or self._overlay_mask.shape != src.shape:
            self._overlay_mask = np.zeros_like(src)
        np.copyto(self._overlay_mask, src)

        # Create new array or resize existing one to match mask dimensions
        if self._motion_mask is None or self._motion_mask.shape != mask.shape:
            self._motion_mask = np.zeros_like(mask)
        np.copyto(self._motion_mask, mask)

        self._rgb_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        self._bounding_boxes.resize(0, refcheck=False)
        self._src_resolution = (src.shape[1], src.shape[0])
        self._mask_resolution = (mask.shape[1], mask.shape[0])

        if self._ratio_res == INVALID_TUPLE:
            self._ratio_res = (
                self._src_resolution[0] / self._mask_resolution[0],
                self._src_resolution[1] / self._mask_resolution[1]
            )
        return None

    def _draw_overlay(self):

        if self._overlay_mode not in (
                MotionOverlayMode.BOUNDING_BOX_OVERLAY,
                MotionOverlayMode.OVERLAY
        ):
            return None

        # Apply color to the areas where motion was detected
        self._rgb_mask[self._motion_mask > 0] = self._mask_overlay_color # performance bottleneck (-45 fps)

        # Resize the mask to match source resolution
        self._rgb_mask = cv2.resize(self._rgb_mask,
                                    self._src_resolution,
                                    interpolation=cv2.INTER_LINEAR)

        # gaussian blur
        self._rgb_mask = cv2.GaussianBlur(self._rgb_mask, (31,31), 0, borderType=cv2.BORDER_CONSTANT)


        # Composite colored mask onto source image
        cv2.addWeighted(
            self._overlay_mask,
            self._mask_overlay_config[0],
            self._rgb_mask,
            self._mask_overlay_config[1],
            self._mask_overlay_config[2],
            dst=self._overlay_mask
        )
        return None

    def _calc_box_dimensions(self, contour):
        area = cv2.contourArea(contour)
        if area < self._min_motion_area:
            return None

        x, y, w, h = cv2.boundingRect(contour)
        dims = [
            int(x * self._ratio_res[0]),
            int(y * self._ratio_res[1]),
            int(w * self._ratio_res[0]),
            int(h * self._ratio_res[1]),
            area
        ]
        dims[2] = dims[0] + dims[2] # x + width
        dims[3] = dims[1] + dims[3] # y + height
        return dims

    def _draw_bounding_boxes(self, mask):

        self._contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL | cv2.CHAIN_APPROX_SIMPLE,
            cv2.CHAIN_APPROX_SIMPLE
        )
        self._bounding_boxes.resize(0, refcheck=False)

        box_index = 0
        for con in self._contours:

            dims = self._calc_box_dimensions(con)
            if dims is None:
                continue
            self._bounding_boxes = np.append(self._bounding_boxes, dims, axis=0)
            # self._bounding_boxes[box_index] = dims
            self._total_area += dims[4]

            if (self._overlay_mode != MotionOverlayMode.BOUNDING_BOX and
                self._overlay_mode != MotionOverlayMode.BOUNDING_BOX_OVERLAY):
                continue

            # TODO: figure out why this hack is necessary and remove
            if self._overlay_mode == MotionOverlayMode.BOUNDING_BOX:
                np.copyto(self._motion_mask, self._overlay_mask)

            cv2.rectangle(
                self._overlay_mask,
                (dims[0], dims[1]),
                (dims[2], dims[3]),
                self._bounding_box_color,
                self._bounding_box_thickness
            )
        return None

    def get_total_area(self):
        return self._total_area

    def update(self, src, mask):

        if (src is None) or (mask is None):
            # skip without error
            return None

        self._set_images(src, mask)
        self._draw_overlay()
        self._draw_bounding_boxes(mask)
        return self._overlay_mask