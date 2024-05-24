import torch
import torch.nn as nn
from diffusers import OmegaSpectraPipeline, EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler, AutoencoderKL
import requests
from io import BytesIO
import os
import re
from pydantic import BaseModel, Field
from typing import Optional

class OmegaPicassoInference:
    def __init__(self, device='cuda:0'):
        """
        Initialize the inference class with specified device.
        Input:
            device (str): The device (e.g., 'cuda:0' or 'cpu') where the model will run.
        """
        model_path = os.getenv("MODEL_PATH")
        vae_path = os.getenv("VAE_PATH")
        image_upload_endpoint = os.getenv("IMAGE_UPLOAD_ENDPOINT")
        image_upload_apikey = os.getenv("IMAGE_UPLOAD_APIKEY")
        vae_path = os.getenv("VAE_PATH")
        print("Args: ", {"MODEL_PATH": model_path, "VAE_PATH": vae_path, "device": device})

        # Check for GPU availability and set up DataParallel if multiple GPUs are available
        self.device = device

        VAE_CACHE = "vae-cache"
        if os.path.exists(VAE_CACHE):
            shutil.rmtree(VAE_CACHE)
        os.makedirs(VAE_CACHE, exist_ok=True)

        MODEL_CACHE = "cache"
        if os.path.exists(MODEL_CACHE):
            shutil.rmtree(MODEL_CACHE)
        os.makedirs(MODEL_CACHE, exist_ok=True)
        
        self.vae = AutoencoderKL.from_single_file(
            vae_path,
            cache_dir=VAE_CACHE
        )

        self.schedulerEulerA = EulerAncestralDiscreteScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear"
        )

        self.schedulerDPMSolver = DPMSolverMultistepScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear"
        )

        self.model = OmegaSpectraPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            use_safetensors=True,
        )
        self.model.save_pretrained(save_directory=MODEL_CACHE, safe_serialization=True)

        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.device)
        self.model = self.model.to(device)
    
    def base(self, x):
        return int(8 * math.floor(int(x)/8))


    def validate_parameters(prompt, parameters):
        """
        Validates the parameters required for image generation.
        Input:
            prompt (str): Text prompt for image generation.
            parameters (dict): Dictionary containing optional parameters:
                - negative_prompt (str): Negative prompt to guide image generation away from certain features.
                - steps (int): Number of diffusion steps.
                - num_of_images (int): Number of images to generate.
                - resolution (str): Resolution of the output images.
                - seed (int): Seed for random number generator for reproducibility.
                - guidance_scale (float): Scale for guidance during the generation.
                - scheduler (str): Type of scheduler used in the diffusion process.
        Output:
            dict: Dictionary containing status, message, and validated parameters formatted for the model.
        """
        
        # Default values and constraints
        defaults = {
            "steps": 50,
            "num_of_images": 1,
            "resolution": "512x768",
            "seed": random.randint(1, 2147483647),
            "guidance_scale": 5.0,
            "scheduler": "v1",
            "negative_prompt": ""
        }
        
        constraints = {
            "steps": (1, 500),
            "num_of_images": (1, 4),
            "resolution": [
                "512x512", "768x768", "1024x1024", "1280x1280", "1536x1536", "2048x2048",
                "768x512", "1024x768", "1536x1024", "1280x960", "1536x1152", "2048x1536",
                "2048x1152", "1280x1024"
            ],
            "seed": (0, 2147483647),
            "guidance_scale": (3.5, 7.0),
            "scheduler": ["v1", "v2"],
            "negative_prompt": 25  # Max length in characters
        }
        
        validated_params = {}
        
        # Validate and set each parameter
        validated_params['prompt'] = prompt
        
        steps = parameters.get("steps", defaults["steps"])
        if not (constraints["steps"][0] <= steps <= constraints["steps"][1]):
            return {
                "status": 0,
                "message": f"Steps must be between {constraints['steps'][0]} and {constraints['steps'][1]}",
                "data": {},
                "code": 400
            }
        validated_params["steps"] = steps
        
        num_of_images = parameters.get("num_of_images", defaults["num_of_images"])
        if not (constraints["num_of_images"][0] <= num_of_images <= constraints["num_of_images"][1]):
            return {
                "status": 0,
                "message": f"Number of images must be between {constraints['num_of_images'][0]} and {constraints['num_of_images'][1]}",
                "data": {},
                "code": 400
            }
        validated_params["num_of_images"] = num_of_images
        
        resolution = parameters.get("resolution", defaults["resolution"])
        if resolution not in constraints["resolution"]:
            return {
                "status": 0,
                "message": f"Resolution must be one of {constraints['resolution']}",
                "data": {},
                "code": 400
            }
        validated_params["resolution"] = resolution
        
        seed = parameters.get("seed", defaults["seed"])
        if not (constraints["seed"][0] <= seed <= constraints["seed"][1]):
            return {
                "status": 0,
                "message": f"Seed must be between {constraints['seed'][0]} and {constraints['seed'][1]}",
                "data": {},
                "code": 400
            }
        validated_params["seed"] = seed
        
        guidance_scale = parameters.get("guidance_scale", defaults["guidance_scale"])
        if not (constraints["guidance_scale"][0] <= guidance_scale <= constraints["guidance_scale"][1]):
            return {
                "status": 0,
                "message": f"Guidance scale must be between {constraints['guidance_scale'][0]} and {constraints['guidance_scale'][1]}",
                "data": {},
                "code": 400
            }
        validated_params["guidance_scale"] = guidance_scale
        
        scheduler = parameters.get("scheduler", defaults["scheduler"])
        if scheduler not in constraints["scheduler"]:
            return {
                "status": 0,
                "message": f"Scheduler must be one of {constraints['scheduler']}",
                "data": {},
                "code": 400
            }
        validated_params["scheduler"] = scheduler
        
        negative_prompt = parameters.get("negative_prompt", defaults["negative_prompt"])
        if len(negative_prompt) > constraints["negative_prompt"]:
            return {
                "status": 0,
                "message": f"Negative prompt must be no more than {constraints['negative_prompt']} characters long",
                "data": {},
                "code": 400
            }
        validated_params["negative_prompt"] = negative_prompt
        
        return {
            "status": 1,
            "message": "Parameters validated successfully",
            "data": validated_params,
            "code": 200
        }
    
    def generate_image(self, prompt, negative_prompt,steps, num_of_images, resolution, seed, guidance_scale,scheduler):
        
        seed_generator = torch.Generator('cuda').manual_seed(seed)
        resolution_split = resolution.split("x")

        width = self.base(resolution_split[0])
        height = self.base(resolution_split[1])
        
        negative_prompt = f"{negative_prompt}"

        result = self.pipe(
            prompt=parameter['prompt'],
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
            scheduler = schedulerEulerA if scheduler == "v2" else schedulerDPMSolver,
            width=width,
            height=height,
            generator=seed_generator,
            censored=False
        )

        image = result.images[0]
        uploaded_image_data = self.upload_image(image)
        return [
                {
                    "index": 0,
                    "image": uploaded_image_data,
                }
            ]

    def upload_image(self, image_object):
        """
               Uploads a PIL Image object to a specified server and returns the URL.
               Input:
                   image_object (PIL.Image): The image to be uploaded.
               Output:
                   str: URL of the uploaded image.
        """
        headers = {
                   'accept': 'multipart/form-data',
                   'x-api-key': image_upload_apikey
                   }
        img_byte_arr = BytesIO()
        image_object.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        files = {'image': ('image.png', img_byte_arr, 'image/png')}
        response = requests.post(image_upload_endpoint, headers=headers, files=files)
        return response.json()['data']