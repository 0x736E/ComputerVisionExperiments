import time
import cv2
import numpy as np
import argparse

from detection.motion_mse import MotionMaskMSE

'''
    benchmark results compare to test/motion.py:

    [ test/motion.py (mse enabled) ]
    Min FPS: 219.45
    Max FPS: 2221.56
    Average FPS: 453.90
    
    [ quick.py (mse disabled) ]
    Min FPS: 248.08
    Max FPS: 580.04
    Average FPS: 445.13
    
    [ quick.py (mse enabled) ]
    Min FPS: 167.73
    Max FPS: 362.55
    Average FPS: 283.65
'''

parser = argparse.ArgumentParser(description="Process video for motion detection.")
parser.add_argument('-i', '--input', required=True, help="Path to the input video file.")
args = parser.parse_args()
input_path = args.input

# hardware acceleration
backend_name = cv2.CAP_ANY
if cv2.CAP_AVFOUNDATION in cv2.videoio_registry.getBackends():
    backend_name = cv2.CAP_AVFOUNDATION
    print("AVFoundation backend is supported")
elif cv2.CAP_FFMPEG in cv2.videoio_registry.getBackends():
    backend_name = cv2.CAP_FFMPEG
    print("FFMPEG backend is supported")
else:
    print("Default to any backend")

cap = cv2.VideoCapture(input_path, backend_name)
# cap.set(cv2.CAP_PROP_FORMAT, -1)  # Use native format
cap.set(cv2.CAP_PROP_HW_ACCELERATION, 1)  # Enable hardware acceleration

out_res = (1280,720)
# proc_res = ( int(out_res[0]/4) , int(out_res[1]/4) )
proc_res = ( 480 , 270 )

ratio_res = (out_res[0]/proc_res[0], out_res[1]/proc_res[1])

MOG2_BG_SUB = cv2.createBackgroundSubtractorMOG2(
    history = 400,
    varThreshold = 15,
    detectShadows = True
)

prev_mask = []
erode_kernel = np.ones((2,2),np.uint8)
dilate_kernel = np.ones((2,2),np.uint8)
blur_kernel = (5,5)
athresh_config = [
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    17,
    16
]
close_shapes_kernel = np.ones((5,5),np.uint8)
max_motion_trail = 10
min_motion_area = 400
bounding_box_color = (0,255,0)
bounding_box_thickness = 2
mask_overlay_color = (0,0,255)
mask_overlay_config = [
    1.0,
    1.0,
    0
]
color_white = (255,255,255)

mse_detect = MotionMaskMSE()

def down_sample(src):
    src = cv2.resize(src, out_res)
    mask = cv2.resize(src, proc_res)  # This will be RGB/BGR
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)  # Now convert to grayscale
    mask = cv2.blur(mask, blur_kernel)
    return src, mask

def compose_mask_overlay(src, mask):
    mask = cv2.merge([mask, mask, mask])
    mask = cv2.resize(mask, out_res) # enlarging to output resolution

    # colourise and composite mask as overlay
    rgb_mask = np.zeros_like(src)
    rgb_mask[mask[:, :, 0] == 255] = mask_overlay_color
    mask = cv2.addWeighted(
        src,
        mask_overlay_config[0],
        rgb_mask,
        mask_overlay_config[1],
        mask_overlay_config[2]
    )
    return mask


def display_frames(src, mask, show_mask=False, show_source=True):

    if show_source:
        src = compose_mask_overlay(src, mask)
        # src = cv2.resize(src, proc_res)
        cv2.imshow('Source', src)

    if show_mask:
        cv2.imshow('Mask', mask)


def display_bounding_boxes(src, mask):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = []
    for cnt in contours:
        cnt_area = cv2.contourArea(cnt)
        if cnt_area < min_motion_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        bounding_boxes.append([x, y, w, h])

        # draw rectangle which is sized to output resolution
        x = int(x * ratio_res[0])
        y = int(y * ratio_res[1])
        w = int(w * ratio_res[0])
        h = int(h * ratio_res[1])

        # draw the rectangle
        cv2.rectangle(src, (x, y), (x + w, y + h), bounding_box_color, bounding_box_thickness)
    return src, bounding_boxes

def mask_motion_trails(mask):
    if max_motion_trail > 0:
        if len(prev_mask) > max_motion_trail:
            prev_mask.pop(0)
        prev_mask.append(mask)
        for pmask in prev_mask:
            pmask = cv2.erode(pmask, np.ones((5,5),np.uint8))
            mask = cv2.add(mask, pmask)
            # mog_threshold = cv2.addWeighted(mog_threshold, 0.5, pmask, 1, 0)
    return mask

def motion_mask(mask):

    # process for MOG2
    mask = MOG2_BG_SUB.apply(mask)

    # reduce noise by Adaptive Threshold
    mask = cv2.adaptiveThreshold(
        mask,
        athresh_config[0],
        athresh_config[1],
        athresh_config[2],
        athresh_config[3],
        athresh_config[4]
    )

    # reduce noise in MOG mask
    mask = cv2.erode(mask, erode_kernel)
    mask = cv2.dilate(mask, dilate_kernel)

    # close shapes
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_shapes_kernel)

    # Fill in the contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(mask, contours, -1, color_white, -1)  # -1 means fill

    # apply motion-trails
    mask = mask_motion_trails(mask)

    return mask

def create_motion_mask(src):
    src, mask = down_sample(src)
    mask = motion_mask(mask)
    return src, mask

def display_visuals(src, mask, show_mask=False, show_source=True):
    display_bounding_boxes(src, mask)
    display_frames(src, mask, show_mask, show_source)

def process(show_mask=False, show_source=True):

    mode_headless = (not show_mask and not show_source)
    frame_count = 0
    total_duration = 0
    min_fps = 9999
    max_fps = 0
    while cap.isOpened():

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        # read input data
        ret, img_src = cap.read()

        # Check both return value and frame content
        if not ret or img_src is None:
            print("End of video reached")
            break

        start_process = time.time()
        mse_detect.update(img_src)
        if mse_detect.get_average() < 0.2:
            continue
        img_src, img_mask = create_motion_mask(img_src)
        duration = time.time() - start_process

        # FPS calculation
        frame_fps = 1 / duration
        print(f"FPS: {frame_fps:.2f}")
        frame_count = frame_count + 1
        total_duration = total_duration + duration
        min_fps = min(frame_fps, min_fps)
        max_fps = max(frame_fps, max_fps)

        if not mode_headless:
            display_visuals(img_src, img_mask, show_mask, show_source)

    if not mode_headless:
        # Ensure proper cleanup
        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    # print average FPS
    avg_fps = (frame_count or 1) / (total_duration or 1)
    print(f"Min FPS: {min_fps:.2f}")
    print(f"Max FPS: {max_fps:.2f}")
    print(f"Average FPS: {avg_fps:.2f}")

if __name__ == "__main__":
    process(False, True)