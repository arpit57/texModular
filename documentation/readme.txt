This App is built using FastAPI, objective is to stream feeds from multiple cameras installed in a textile factory and to keep count of the number of times a worker picks up a cloth from the pickup bin and after working on it puts it in the drop bin. Yolo v8 model was trained on human hand annotated images for detection.

App has 2 major endpoints, 'video' and 'addVideo'.

addVideo: accepts a video URL from a HTML script and diplays the first frame of that video and prompts the user to draw bounding boxes for pickup and drop regions. also asks the user to select side(left, right, bottom or top) for that particular region. when the user clicks 'save', JSON file is updated with new URL, pickup and drop co-ordinates and sides.

Video: reads data from a JSON file, having details about video URL, pickup co-ordinates and drop co-ordinates for each video stream. Adjusts the grid size based on the number of video URLs. connects to a mySQL database running inside docker(script is running outside docker) to update the data. data has columns like 'video-id', 'count' and 'time'.

