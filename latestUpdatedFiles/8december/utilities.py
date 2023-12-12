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

def draw_sides(frame, region, sides, color):
    x_start, y_start, x_end, y_end = region[0][0], region[0][1], region[1][0], region[1][1]
    thickness = 2

    if 'left' in sides:
        cv2.line(frame, (x_start, y_start), (x_start, y_end), color, thickness)
    if 'right' in sides:
        cv2.line(frame, (x_end, y_start), (x_end, y_end), color, thickness)
    if 'top' in sides:
        cv2.line(frame, (x_start, y_start), (x_end, y_start), color, thickness)
    if 'bottom' in sides:
        cv2.line(frame, (x_start, y_end), (x_end, y_end), color, thickness)

def point_line_distance(point, line_start, line_end):
    """Calculate the minimum distance from a point to a line segment."""
    # Line vector
    line_vec = np.array(line_end) - np.array(line_start)
    # Point vector
    point_vec = np.array(point) - np.array(line_start)
    # Line length squared
    line_len2 = line_vec.dot(line_vec)
    # Project point onto the line using dot product
    projection = point_vec.dot(line_vec) / line_len2
    if projection < 0:
        projection = 0
    elif projection > 1:
        projection = 1
    # Find the closest point on the line segment
    closest_point = np.array(line_start) + projection * line_vec
    # Return the distance from the point to the closest point on the line
    return np.linalg.norm(closest_point - np.array(point))

def is_entering_from_side(box, region, sides, threshold=50):
    """Check if the center of a bounding box is within a threshold distance of a specified side of a region."""
    x_center, y_center = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    
    side_centers = {
        'left': ((region[0][0], region[0][1]), (region[0][0], region[1][1])),
        'right': ((region[1][0], region[0][1]), (region[1][0], region[1][1])),
        'top': ((region[0][0], region[0][1]), (region[1][0], region[0][1])),
        'bottom': ((region[0][0], region[1][1]), (region[1][0], region[1][1]))
    }
    
    for side in sides:
        line_start, line_end = side_centers[side]
        if point_line_distance((x_center, y_center), line_start, line_end) < threshold:
            return True
    return False
