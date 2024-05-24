from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.security import APIKeyHeader
from fastapi.responses import JSONResponse
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, HttpUrl, Field, validator
from typing import Optional, Union, List
from PIL import Image
import requests
import io
import torch
from itertools import cycle
import re
from .generate_image import OmegaPicassoInference  # Ensure this module is correctly set up

app = FastAPI()

# Assume four GPUs available (indexed from 0 to 3)
gpus = [f'cuda:{i}' for i in range(torch.cuda.device_count())]
models = {gpu: OmegaPicassoInference(device=gpu) for gpu in gpus}
model_cycle = cycle(models.values())  # Cycle through model instances

# Check if all required environment variables are available, not null, and non-empty
required_env_vars = ["MODEL_PATH", "VAE_PATH", "IMAGE_UPLOAD_ENDPOINT", "IMAGE_UPLOAD_APIKEY", "TOKEN"]

for var in required_env_vars:
    value = os.getenv(var)
    if not value:
        raise RuntimeError(f"Environment variable {var} is not set or is empty")

# Retrieve API keys from environment variable, split by comma, filter valid keys
api_keys_raw = os.getenv("API_KEYS", "")
API_KEYS = {key: {'type': 'standard'} for key in api_keys_raw.split(',') if len(key) == 32 and re.match(r'^[a-zA-Z0-9]+$', key)}

api_key_header = APIKeyHeader(name="X-API-Key")

def get_api_key(api_key: str = Depends(api_key_header)):
    if API_KEYS and (api_key not in API_KEYS):
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key

model_path = os.getenv("MODEL_PATH", "")
vae_path = os.getenv("VAE_PATH", "")
image_upload_endpoint = os.getenv("IMAGE_UPLOAD_ENDPOINT", "")
image_upload_apikey = os.getenv("IMAGE_UPLOAD_APIKEY", "")
token = os.getenv("TOKEN", "")
print("Args: ", {"MODEL_PATH": model_path, "VAE_PATH": vae_path, "TOKEN": token, "API_KEY": "true" if API_KEYS else "false", "device": device})

class RequestBody(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = Field(default="", max_length=25)
    steps: Optional[int] = Field(default=50, ge=1, le=500)
    num_of_images: Optional[int] = Field(default=1, ge=1, le=4)
    resolution: Optional[str] = Field(default="512x768")
    seed: Optional[int] = Field(default_factory=lambda: random.randint(1, 2147483647), ge=0, le=2147483647)
    guidance_scale: Optional[float] = Field(default=5.0, ge=3.5, le=7.0)
    scheduler: Optional[Literal["v1", "v2"]] = "v1"

    @validator("resolution")
    def validate_resolution(cls, v):
        valid_resolutions = [
            "512x512", "768x768", "1024x1024", "1280x1280", "1536x1536", "2048x2048",
            "768x512", "1024x768", "1536x1024", "1280x960", "1536x1152", "2048x1536",
            "2048x1152", "1280x1024"
        ]
        if v not in valid_resolutions:
            raise ValueError(f"Resolution must be one of {valid_resolutions}")
        return v

@app.get("/health")
async def health_check():
    return {
        "status": "online",
    }

@app.post("/generate-images/")
async def generate_image(request_body: RequestBody, api_key: str = Depends(get_api_key) if API_KEYS else None):
     model = next(model_cycle)
    try:
        images = await run_in_threadpool(
            model.generate_image,
            request_body.prompt,
            negative_prompt=request_body.negative_prompt,
            num_of_images=request_body.num_of_images,
            scheduler=request_body.scheduler,
            seed=request_body.seed,
            resolution=request_body.resolution,
            guidance_scale=request_body.guidance_scale,
            steps=request_body.steps,
        )
        return JSONResponse(status_code=200, content={"status": 1, "message": "Image generated", "data": images, "code": 200})
    except Exception as e:
        error_details = {"status": 0, "message": "Something went wrong", "data": [], "code": 500}
        # if environment == 'testing': #Send Error only to testing Environment
        #     error_details['error'] = str(e)
        return JSONResponse(status_code=500, content=error_details)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
