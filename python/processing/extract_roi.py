import numpy as np

'''
    The purpose of this class is to extract regions of interest (ROI) from a source image based on bounding boxes.
    These ROIs can then be used for further processing such as object detection or image segmentation.
    
    TODO: Make this work
'''

class RoiExtractor:
    def __init__(self):
        self._regions = np.array([])

    def get_regions(self):
        return self._regions

    def _extract(self, src, boxes):
        if src is None or boxes is None or len(boxes) == 0:
            return None

        self._regions = np.zeros(len(boxes), dtype=np.uint8)
        for box in boxes:
            region = np.ascontiguousarray(
                src[int(box[1]):int(box[1]+box[3]),
                    int(box[0]):int(box[0]+box[2]) ]
            )
            self._regions = np.append(self._regions, region)
        return None

    def update(self, src, boxes):
        self._extract(src, boxes)
        return None