from fastapi import FastAPI, HTTPException, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import json
from pathlib import Path
import uvicorn

app = FastAPI()

templates = Jinja2Templates(directory=".")

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

    json_path = 'video_paths.json'
    try:
        if Path(json_path).exists():
            with open(json_path, 'r+') as file:
                json_data = json.load(file)
                json_data['videos'].append(video_path)
                if pickup_coords:
                    json_data['pickup_coords'].append(pickup_coords)
                if drop_coords:
                    json_data['drop_coords'].append(drop_coords)
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


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
