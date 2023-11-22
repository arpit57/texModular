import cv2
from time import time
from datetime import datetime
from ultralytics import YOLO
from utilities import is_inside, draw_detections_on_frame
from db_config import get_db_connection
# from dynamoDB_aws import Bucketdynamodb

# Creating and initialize the calss with dynamodb table name
# dynamodb = Bucketdynamodb("video_data_tex_arpit")
# Creating table
# dynamodb.create_table()

model = YOLO('handDetection.pt')

class State:
    WAIT_FOR_PICKUP = 1
    WAIT_FOR_DROP = 2

def process_video(video_url, pickup_coords, drop_coords, out_frames, video_index):
    cap = cv2.VideoCapture(video_url)
    if not cap.isOpened():
        raise ValueError(f"Couldn't open video stream from URL: {video_url}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_fps = int(cap.get(cv2.CAP_PROP_FPS))

    frame_count = 0
    state = State.WAIT_FOR_PICKUP
    count = 0
    hand_was_in_drop = False

    last_count = -1
    last_time = None
    cycle_time = 0.0


    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Stream ended.")
                break

            if frame_count % (original_fps // 5) == 0:  # limiting to 5 fps
                results = model(frame, conf=0.05, iou=0.5)
                hand_detected_in_pickup = any(is_inside(map(int, box_data[:4]), pickup_coords) for box_data in results[0].boxes.data.cpu().numpy())
                hand_detected_in_drop = any(is_inside(map(int, box_data[:4]), drop_coords) for box_data in results[0].boxes.data.cpu().numpy())

                current_date = datetime.now().strftime('%Y-%m-%d')
                current_time = datetime.now().strftime('%H:%M:%S')

                if state == State.WAIT_FOR_PICKUP and hand_detected_in_pickup:
                    state = State.WAIT_FOR_DROP

                if state == State.WAIT_FOR_DROP and hand_detected_in_drop:
                    hand_was_in_drop = True

                if hand_was_in_drop and hand_detected_in_pickup:
                    count += 1
                    state = State.WAIT_FOR_DROP
                    hand_was_in_drop = False

                if last_count != count:
                    recent_time = time()  # Record the current time

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

                        # db_daata ={"current_date":current_date,"current_time":current_time, "video_index":video_index, "cycle_count":last_count, "cycle_time":cycle_time}
                        # dynamodb.insert_data_to_table(db_daata)

                        cursor_thread.execute(insert_query, (current_date, current_time, video_index, last_count, cycle_time))
                        db.commit()
                        cursor_thread.close()
                        db.close()
                    except Exception as e:
                        print(f"Failed to insert data for video index {video_index}. Error: {e}")
                        raise

                for box_data in results[0].boxes.data.cpu().numpy():
                    frame = draw_detections_on_frame(frame, box_data, results[0].names)

                cv2.putText(frame, f"Count: {count}", (width - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Pickup: {hand_detected_in_pickup}", (width - 150, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(frame, f"Drop: {hand_detected_in_drop}", (width - 150, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(frame, f"Was in Drop: {hand_was_in_drop}", (width - 150, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(frame, f"State: {state}", (width - 150, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.rectangle(frame, pickup_coords[0], pickup_coords[1], (0, 255, 0), 2)
                cv2.rectangle(frame, drop_coords[0], drop_coords[1], (0, 255, 0), 2)
                
                out_frames.append(frame)

            frame_count += 1

    finally:
        cap.release()
        cv2.destroyAllWindows()



