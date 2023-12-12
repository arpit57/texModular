import math
import cv2
import numpy as np
from shapely.geometry import Point, Polygon, LineString


def is_inside(point, polygon):
    """Check if the point (x, y) is inside the given polygon."""
    shapely_point = Point(point)
    shapely_polygon = Polygon(polygon)
    return shapely_polygon.contains(shapely_point)

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

def draw_sides(frame, polygon, side_indices, color):
    """Draw the indicated sides of the polygon on the frame."""
    thickness = 2
    sorted_polygon = get_sorted_polygon_vertices(polygon)
    n = len(sorted_polygon)
    for index in side_indices:
        # Convert side numbers assuming 1-based index from user input to 0-based index for programming
        adjusted_index = index - 1
        start_vertex = tuple(sorted_polygon[adjusted_index % n])
        end_vertex = tuple(sorted_polygon[(adjusted_index + 1) % n])
        cv2.line(frame, start_vertex, end_vertex, color, thickness)

def get_sorted_polygon_vertices(polygon):
    # Find the vertex with the highest y-value (and the leftmost in case of ties)
    high_point = max(polygon, key=lambda point: (point[1], -point[0]))
    high_index = polygon.index(high_point)

    # Shift the polygon vertices list so that the highest point is the first
    shifted_polygon = polygon[high_index:] + polygon[:high_index]

    # Find the centroid
    centroid = Polygon(shifted_polygon).centroid.coords[0]

    # Sort the vertices in clockwise order starting from the highest point
    def sort_key(point):
        # Calculate the angle of each point relative to the centroid
        angle = np.arctan2(point[1] - centroid[1], point[0] - centroid[0])
        # Normalize the angle to be in the range [0, 2*pi]
        normalized_angle = (angle + 2 * np.pi) % (2 * np.pi)
        return normalized_angle

    # We sort the vertices according to their angle in clockwise direction
    sorted_vertices = sorted(shifted_polygon, key=sort_key)

    return sorted_vertices

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


def is_entering_from_side(point, polygon, side_numbers, threshold=50):
    """Check if the point is within a threshold distance of any indicated polygon side."""
    shapely_point = Point(point)
    sorted_polygon = get_sorted_polygon_vertices(polygon)
    for side_number in side_numbers:
        # Convert the side number to index in the sorted polygon vertex list
        line_start = sorted_polygon[side_number % len(sorted_polygon)]
        line_end = sorted_polygon[(side_number + 1) % len(sorted_polygon)]
        line = LineString([line_start, line_end])
        if shapely_point.distance(line) < threshold:
            return True
    return False