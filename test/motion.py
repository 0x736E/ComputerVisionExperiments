import argparse
import cv2
import time

import numpy as np

from masks.motion_mog2 import MotionMaskMOG2
from detection.motion_mse import MotionMaskMSE
from masks.motion_overlay import MotionOverlayMask, MotionOverlayMode
from processing.extract_roi import RoiExtractor

'''
    and the trogdor comes in the NIIGGHHHTTTTT!!!!!!!!
                                                 :::
                                             :: :::.
                       \/,                    .:::::
           \),          \`-._                 :::888
           /\            \   `-.             ::88888
          /  \            | .(                ::88
         /,.  \           ; ( `              .:8888
            ), \         / ;``               :::888
           /_   \     __/_(_                  :88
             `. ,`..-'      `-._    \  /      :8
               )__ `.           `._ .\/.
              /   `. `             `-._______m         _,
  ,-=====-.-;'                 ,  ___________/ _,-_,'"`/__,-.
 C   =--   ;                   `.`._    V V V       -=-'"#==-._
:,  \     ,|      UuUu _,......__   `-.__A_A_ -. ._ ,--._ ",`` `-
||  |`---' :    uUuUu,'          `'--...____/   `" `".   `
|`  :       \   UuUu:
:  /         \   UuUu`-._
 \(_          `._  uUuUu `-.
 (_3             `._  uUu   `._
                    ``-._      `.
                         `-._    `.
                             `.    \
                               )   ;
                              /   /
               `.        |\ ,'   /
                 ",_A_/\-| `   ,'
                   `--..,_|_,-'\
                          |     \
                          |      \__
                          |__


taken from http://github.com/duwanis/trogdor/
'''
'''
    yes, I do know this code is ugly, but it works.
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

# input_path = 'rtsp://XXX:7447/YYY'
# backend_name = cv2.CAP_FFMPEG

cap = cv2.VideoCapture(input_path, backend_name)
# cap.set(cv2.CAP_PROP_FORMAT, -1)  # Use native format
cap.set(cv2.CAP_PROP_HW_ACCELERATION, 1)

# Before the main loop
frame_width = 1280  # Since you're resizing to these dimensions
frame_height = 720
fps = cap.get(cv2.CAP_PROP_FPS)

# Try different codec combinations
# fourcc = cv2.VideoWriter_fourcc(*'H264')
# output_path = 'output.mp4'

# Create writer and verify it's initialized properly
# out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

motion_mask = MotionMaskMOG2()
overlay_mode = MotionOverlayMode.NONE
motion_overlay_mask = MotionOverlayMask(mode=overlay_mode)
roi_extractor = RoiExtractor()
mse_calc = MotionMaskMSE(threshold=0.2)

frame_count = 0
total_duration = 0
min_fps = 9999
max_fps = 0

def draw_label(img, point, label, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, thickness=2, padding=(20,20), bg_color=(255,0,0)):
    if overlay_mode == MotionOverlayMode.NONE:
        return None

    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(img, (x - int(padding[0]/2), y - size[1] - int(padding[1]/2)), (x + size[0] + padding[0], y + padding[1]), bg_color, cv2.FILLED)
    cv2.putText(img, label, point, font, font_scale, (255, 255, 255), thickness)

frame_number = 0
fps_label_value = 0
motion_area_label_value = 0
frame_changed = False
while cap.isOpened():

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    # read input data
    ret, img_src = cap.read()
    frame_number = frame_number + 1

    # Check both return value and frame content
    if not ret or img_src is None:
        print("End of video reached")
        break

    # downsample
    img_src = cv2.resize(img_src, (1280, 720))

    # update masks
    start_process = time.time()

    # MSE - Optimisation
    mse_calc.update(img_src)
    # frame_changed = (mse_calc.get_difference() > 0.2)

    frame_changed = mse_calc.get_average() > 0.2
    # mse_index = 0
    # changed_regions_index = []
    # for mse in mse_calc.get_mse_values():
    #     mse_index = mse_index + 1
    #     if mse > 0.2:
    #         frame_changed = True
    #         changed_regions_index.append(mse_index)
    #
    # # Calculate rectangle coordinates for each changed region
    # if len(changed_regions_index) > 0:
    #     height, width = img_src.shape[:2]
    #     downsample_height, downsample_width = mse_calc._downsample_resolution
    #     region_height = downsample_height // mse_calc._region_multiplier
    #     region_width = downsample_width // mse_calc._region_multiplier
    #     scaling_factor_h = height / downsample_height
    #     scaling_factor_w = width / downsample_width
    #
    #     rectangles = []
    #     for region_idx in changed_regions_index:
    #         row = (region_idx - 1) // mse_calc._region_multiplier
    #         col = (region_idx - 1) % mse_calc._region_multiplier
    #         x1 = int(col * region_width * scaling_factor_w)
    #         y1 = int(row * region_height * scaling_factor_h)
    #         x2 = int((col + 1) * region_width * scaling_factor_w)
    #         y2 = int((row + 1) * region_height * scaling_factor_h)
    #         rectangles.append((x1, y1, x2, y2))

    if not frame_changed:
        mask = img_src.copy()
        draw_label(mask, (50, 220), "MSE: STATIC", font_scale=1, thickness=2, padding=(10,10), bg_color=(0,255,0) )
    else:

        mask = motion_overlay_mask.update(
            img_src,
            motion_mask.update(img_src)
        )

        # Draw rectangles over detected motion regions
        # for (x1, y1, x2, y2) in rectangles:
        #     overlay = mask.copy()
        #     cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)  # Filled red rectangle
        #     cv2.addWeighted(overlay, 0.3, mask, 0.7, 0, mask)  # Blend the overlay with transparency
        # regions = roi_extractor.update(
        #     motion_overlay_mask.get_mask(),
        #     motion_overlay_mask.get_bounding_boxes()
        # )
        # change_index_str = ", ".join(map(str, changed_regions_index))
        change_label = f"MSE: CHANGED "
        print(change_label)
        draw_label(mask, (50, 220), change_label, font_scale=1, thickness=2, padding=(10,10), bg_color=(0,0,255) )

    duration = time.time() - start_process

    # FPS calculation
    frame_fps = 1 / duration
    # print(f"FPS: {frame_fps:.2f}")
    frame_count = frame_count + 1
    total_duration = total_duration + duration
    min_fps = min(frame_fps, min_fps)
    max_fps = max(frame_fps, max_fps)

    if frame_number % 5 == 0:
        fps_label_value = frame_fps
        motion_area_label_value = motion_overlay_mask.get_total_area() / 1000
        print(fps_label)

    fps_label = f"{fps_label_value:.2f} FPS"
    draw_label(mask, (50, 100), fps_label)


    # area
    total_area = motion_overlay_mask.get_total_area()
    motion_label = ""
    if total_area > 600:
        motion_label = f"Motion detected ({motion_area_label_value:.2f}) K/p2"
        if overlay_mode != MotionOverlayMode.NONE:
            draw_label(mask, (50, 170), motion_label, font_scale=1, thickness=2, padding=(10,10), bg_color=(0,0,255) )
        # draw_label(mask, (50, 220), f"Regions: ({len(roi_extractor.get_regions()):.0f})", font_scale=1, thickness=2, padding=(10,10), bg_color=(0,0,255) )
        print(motion_label)


    # show
    # cv2.imshow('Source', img_src)
    if overlay_mode != MotionOverlayMode.NONE and mask is not None:
        cv2.imshow('Output', mask)
        # cv2.imshow('Mask', motion_mask.get_mask()) # optionally show the mask

# print average FPS
avg_fps = (frame_count or 1) / (total_duration or 1)
print(f"Min FPS: {min_fps:.2f}")
print(f"Max FPS: {max_fps:.2f}")
print(f"Average FPS: {avg_fps:.2f}")

# Ensure proper cleanup
# out.release()
cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)