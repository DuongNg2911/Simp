from fastapi import FastAPI
from PIL import Image

from routes import router
from engine.removebg import removebg

app = FastAPI()
app.include_router(router)

# POST
@app.post("/remove_bg")
async def remove_bg(image):
    return removebg(image)

# @app.post("/remove_object")
# async def remove_object(image):
