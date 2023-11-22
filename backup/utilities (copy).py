import math
import cv2
import numpy as np

def is_inside(box, rect):
    x1, y1, x2, y2 = box
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return rect[0][0] <= center_x <= rect[1][0] and rect[0][1] <= center_y <= rect[1][1]

def calculate_grid_dimensions(total_videos):
    # Calculate the width based on the 2:3 ratio
    width = round(math.sqrt(total_videos * 3 / 2))
    height = math.ceil(total_videos / width)
    return height, width

def black_frame_like(frame):
    # Return a black frame of the same shape and type as the input frame
    return np.zeros_like(frame)


def draw_detections_on_frame(frame, box_data, names):
    x1, y1, x2, y2, conf, cls = box_data
    label = f'{names[int(cls)]} {conf:.2f}'
    color = [int(c) for c in (255, 0, 0)]
    tl = round(0.002 * (frame.shape[0] + frame.shape[1]) / 2) + 1
    c1, c2 = (int(x1), int(y1)), (int(x2), int(y2))
    cv2.rectangle(frame, c1, c2, color, thickness=tl)
    tf = max(tl - 1, 1)
    t_size = cv2.getTextSize(label, 0, fontScale=tf / 3, thickness=tf)[0]
    c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
    cv2.rectangle(frame, c1, c2, color, -1)
    cv2.putText(frame, label, (c1[0], c1[1] - 2), 0, tf / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return frame

