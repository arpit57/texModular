import numpy as np
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import asyncio
import cv2
import threading
from utilities import calculate_grid_dimensions, black_frame_like
from video_processing import process_video
from time import sleep
import json
from pathlib import Path


app = FastAPI()

templates = Jinja2Templates(directory=".")

@app.get("/")
def read_root():
    return {"Hello": "World"}

# Directory for storing videos
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/addVideo/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("templates/index.html", {"request": request})

@app.post("/update_json/")
async def update_json(request: Request):
    data = await request.json()
    video_path = data.get('videos')[0]  # Assuming only one video path is sent at a time
    pickup_coords = data.get('pickup_coords', None)
    drop_coords = data.get('drop_coords', None)
    pickup_edges = data.get('pickup_edges', [])
    drop_edges = data.get('drop_edges', [])

    json_path = 'video_paths.json'
    try:
        if Path(json_path).exists():
            with open(json_path, 'r+') as file:
                json_data = json.load(file)
                json_data['videos'].append(video_path)
                if pickup_coords:
                    json_data['pickup_coords'].append(pickup_coords)
                    json_data['pickup_edges'].append(pickup_edges)
                if drop_coords:
                    json_data['drop_coords'].append(drop_coords)
                    json_data['drop_edges'].append(drop_edges)
                file.seek(0)
                json.dump(json_data, file, indent=4)
        else:
            initial_data = {
                "videos": [video_path],
                "pickup_coords": [pickup_coords] if pickup_coords else [],
                "drop_coords": [drop_coords] if drop_coords else []
            }
            with open(json_path, 'w') as file:
                json.dump(initial_data, file, indent=4)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"message": "JSON updated successfully"}


@app.get("/video/")
async def video_feed():
    stream = stream_videos()  # Call the generator function
    return StreamingResponse(stream, media_type="multipart/x-mixed-replace; boundary=frame")

async def stream_videos():
    with open('video_paths.json', 'r') as file:
        data = json.load(file)
        video_urls = data.get("videos")
        all_pickup_coords = [tuple(map(tuple, coords)) for coords in data.get("pickup_coords")]
        all_drop_coords = [tuple(map(tuple, coords)) for coords in data.get("drop_coords")]


    grid_height, grid_width = calculate_grid_dimensions(len(video_urls))

    out_frames_list = [[] for _ in video_urls]
    threads = []

    for idx, video_url in enumerate(video_urls):
        t = threading.Thread(target=process_video, args=(video_url, all_pickup_coords[idx], all_drop_coords[idx], out_frames_list[idx], idx))
        t.start()
        threads.append(t)


    # Yield frames as they become available
    while True:
        all_frames_available = all([len(frames) > 0 for frames in out_frames_list])
        if all_frames_available:
            grid_frames = []

            # Create rows for the grid
            for i in range(0, len(out_frames_list), grid_width):
                row_frames = [out_frames_list[j].pop(0) if j < len(out_frames_list) else black_frame_like(out_frames_list[0][0]) for j in range(i, i + grid_width)]
                grid_frames.append(np.hstack(row_frames))

            # Combine rows to form the grid
            combined_frame = np.vstack(grid_frames)
            _, buffer = cv2.imencode('.jpg', combined_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
            sleep(0.1)

        else:
            # Wait a short period before checking for frames again
            await asyncio.sleep(0.1)

    for t in threads:
        t.join()


