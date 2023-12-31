import cv2
import time
from datetime import datetime
from ultralytics import YOLO
from utilities import is_inside, draw_detections_on_frame, is_entering_from_side, draw_sides
from db_config import get_db_connection
import onnxruntime
import numpy as np

model = YOLO('handDetection.pt')

class State:
    WAIT_FOR_PICKUP = 1
    WAIT_FOR_DROP = 2

def process_video(video_url, pickup_coords, drop_coords, pickup_sides, drop_sides, out_frames, video_index):
    cap = cv2.VideoCapture(video_url)
    if not cap.isOpened():
        raise ValueError(f"Couldn't open video stream from URL: {video_url}")

    width = 854
    height = 480
    original_fps = int(cap.get(cv2.CAP_PROP_FPS))

    frame_count = 0
    state = State.WAIT_FOR_PICKUP
    count = 0
    hand_was_in_drop = False

    last_count = -1
    last_time = None
    cycle_time = 0.0

    start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                end_time = time.time()  # Record end time when stream ends
                print(f"Stream ended. Total processing time: {end_time - start_time:.2f} seconds.")
                break
            
            frame = cv2.resize(frame, (854, 480))

            if frame_count % (original_fps // 5) == 0:  # limiting to 5 fps
                results = model(frame, conf=0.5, iou=0.5)
                hands_in_frame = [list(map(int, box_data[:4])) for box_data in results[0].boxes.data.cpu().numpy() if len(box_data) >= 4]

                hand_detected_in_pickup = False
                hand_detected_in_drop = False
                entering_pickup = False
                entering_drop = False
                for hand in hands_in_frame:
                    hand_center = [(hand[0] + hand[2]) / 2, (hand[1] + hand[3]) / 2]
    
                    if is_inside(hand_center, pickup_coords):
                        hand_detected_in_pickup = True
                    if is_inside(hand_center, drop_coords):
                        hand_detected_in_drop = True
                    if is_entering_from_side(hand_center, pickup_coords, pickup_sides):
                        entering_pickup = True
                    if is_entering_from_side(hand_center, drop_coords, drop_sides):
                        entering_drop = True

                current_date = datetime.now().strftime('%Y-%m-%d')
                current_time = datetime.now().strftime('%H:%M:%S')

                if state == State.WAIT_FOR_PICKUP and hand_detected_in_pickup and entering_pickup:
                    state = State.WAIT_FOR_DROP

                if state == State.WAIT_FOR_DROP and hand_detected_in_drop and entering_drop:
                    hand_was_in_drop = True

                if hand_was_in_drop and hand_detected_in_pickup and entering_pickup:
                    count += 1
                    state = State.WAIT_FOR_DROP
                    hand_was_in_drop = False


                if last_count != count:
                    recent_time = time.time()  # Record the current time

                    if last_time is not None:
                        cycle_time = recent_time - last_time  # Calculate the cycle time
                    last_time = recent_time  # Update last_time for the next cycle
                    last_count = count  # Update last_count for the next cycle

                    # Insert data into MySQL
                    try:
                        db = get_db_connection()  # Get a database connection
                        cursor_thread = db.cursor()
                        insert_query = "INSERT INTO video_data (`current_date`, `current_time`, `video_index`, `cycle_count`, `cycle_time`) VALUES (%s, %s, %s, %s, %s)"
                        print("Executing SQL:", insert_query)
                        print("Data:", (current_date, current_time, video_index, last_count, cycle_time))
                        cursor_thread.execute(insert_query, (current_date, current_time, video_index, last_count, cycle_time))
                        db.commit()
                        cursor_thread.close()
                        db.close()
                    except Exception as e:
                        print(f"Failed to insert data for video index {video_index}. Error: {e}")
                        raise

                for box_data in results[0].boxes.data.cpu().numpy():
                    frame = draw_detections_on_frame(frame, box_data, results[0].names)

                cv2.putText(frame, f"Cycle Count: {count}", (width - 250, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Pickup: {hand_detected_in_pickup}", (width - 200, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(frame, f"Drop: {hand_detected_in_drop}", (width - 200, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(frame, f"Was in Drop: {hand_was_in_drop}", (width - 200, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(frame, f"Entering Pickup: {entering_pickup}", (width - 200, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(frame, f"Entering Drop: {entering_drop}", (width - 200, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(frame, f"State: {state}", (width - 200, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                pickup_polygon_np = np.array([pickup_coords], np.int32)
                pickup_polygon_np = pickup_polygon_np.reshape((-1, 1, 2)) # Reshape for polylines
                
                drop_polygon_np = np.array([drop_coords], np.int32)
                drop_polygon_np = drop_polygon_np.reshape((-1, 1, 2)) # Reshape for polylines
                
                cv2.polylines(frame, [pickup_polygon_np], isClosed=True, color=(0, 255, 0), thickness=2)
                cv2.polylines(frame, [drop_polygon_np], isClosed=True, color=(0, 255, 0), thickness=2)

                draw_sides(frame, pickup_coords, pickup_sides, (0, 0, 255))
                draw_sides(frame, drop_coords, drop_sides, (0, 0, 255))
                
                out_frames.append(frame)

            frame_count += 1

    finally:
        cap.release()
        cv2.destroyAllWindows()



