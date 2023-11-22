import numpy as np
from fastapi import FastAPI, Request, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
# from fastapi.staticfiles import StaticFiles
import asyncio
import cv2
import threading
from utilities import calculate_grid_dimensions, black_frame_like
from video_processing import process_video
from time import sleep
import json
from pathlib import Path
# from typing import List, Dict, Any
import logging


app = FastAPI()

templates = Jinja2Templates(directory=".")

@app.get("/")
def read_root():
    return {"Hello": "World"}

# Directory for storing videos
# app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/addVideo/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("templates/index.html", {"request": request})

@app.get("/get_paths/")
async def get_paths():
    with open('video_paths.json', 'r') as file:
        data = json.load(file)
        return {"videos": data['videos']}

@app.post("/addVideo")
async def add_video(request: Request, file: UploadFile = File(None)):
    json_path = 'video_paths.json'
    video_path = ""

    if file and file.filename:
        # Handle file upload
        file_location = f"/home/arpit/Testing/texModular/videos/{file.filename}"
        with open(file_location, "wb+") as file_object:
            file_object.write(file.file.read())
        video_path = file_location
    else:
        # Handle video URL from form data
        form_data = await request.form()
        video_url = form_data.get("video_url")
        if not video_url:
            raise HTTPException(status_code=400, detail="No video file or URL provided")
        video_path = video_url

    # Update the JSON file
    try:
        with open(json_path, 'r+') as json_file:
            data = json.load(json_file)
            data['videos'].append(video_path)
            json_file.seek(0)
            json.dump(data, json_file, indent=4)
        return {"info": f"Video path '{video_path}' saved successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/update_json/")
async def update_json(request: Request):
    data = await request.json()
    video_path = data.get('video_path')
    pickup_coords = data.get('pickup_coords', None)
    drop_coords = data.get('drop_coords', None)
    pickup_sides = data.get('pickup_sides', [])
    drop_sides = data.get('drop_sides', [])

    # Now save both the video path (or URL) and the coordinates to the JSON file
    json_path = 'video_paths.json'
    try:
        if Path(json_path).exists():
            with open(json_path, 'r+') as file:
                json_data = json.load(file)
                json_data['videos'].append(video_path)
                if pickup_coords:
                    json_data['pickup_coords'].append(pickup_coords)
                    json_data['pickup_sides'].append(pickup_sides)
                if drop_coords:
                    json_data['drop_coords'].append(drop_coords)
                    json_data['drop_sides'].append(drop_sides)
                file.seek(0)
                json.dump(json_data, file, indent=4)
        else:
            raise HTTPException(status_code=500, detail="JSON file not found")
        return {"message": "Coordinates, sides and video URL/path saved successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/remove_paths/")
async def remove_paths(request: Request):
    try:
        body = await request.json()
        indices = body.get('indices')

        if not isinstance(indices, list) or not all(isinstance(index, int) for index in indices):
            raise ValueError("Indices must be a list of integers.")

        with open('video_paths.json', 'r+') as file:
            data = json.load(file)
            indices.sort(reverse=True)  # Sort indices in descending order
            for index in indices:
                if index < 0 or index >= len(data['videos']):
                    raise IndexError("Index out of range.")
                # Remove elements by index from each list
                del data['videos'][index]
                del data['pickup_coords'][index]
                del data['drop_coords'][index]
                del data['pickup_sides'][index]
                del data['drop_sides'][index]

            file.seek(0)  # Reset file pointer to the beginning of the file
            json.dump(data, file, indent=4)
            file.truncate()  # Remove the rest of the original data

        return {"message": "Removed successfully"}
    except ValueError:
        raise HTTPException(status_code=422, detail="Indices must be a list of integers.")
    except IndexError:
        raise HTTPException(status_code=422, detail="Index out of range.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
        all_pickup_sides = data.get("pickup_sides")  
        all_drop_sides = data.get("drop_sides")      



    grid_height, grid_width = calculate_grid_dimensions(len(video_urls))

    out_frames_list = [[] for _ in video_urls]
    threads = []

    for idx, video_url in enumerate(video_urls):
        t = threading.Thread(target=process_video, args=(video_url, all_pickup_coords[idx], all_drop_coords[idx], all_pickup_sides[idx], all_drop_sides[idx], out_frames_list[idx], idx))
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


