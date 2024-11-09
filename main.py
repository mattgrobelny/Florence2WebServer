import io
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Security
from fastapi.responses import HTMLResponse
from fastapi.security.api_key import APIKeyHeader, APIKey
import logging
import os
from PIL import Image

from FlorenceModel import Florence2Model

# Define the API key in the environment or code
API_KEY = os.getenv("API_KEY", "mysecureapikey123")
API_KEY_NAME = "access-token"  # This will be the header name

# Initialize FastAPI app
app = FastAPI()

# Create an API key dependency
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# Access FastAPI's default logger
logger = logging.getLogger("uvicorn")
logger.info('Start loading model')
model = Florence2Model()

logger.info('Finished loading model')

async def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header == API_KEY:
        logger.info("API Key Authorization Successed")
        return api_key_header
    else:
        logger.error("Could not validate credentials")
        raise HTTPException(status_code=403, detail="Could not validate credentials")


# Endpoint to list available tasks
@app.get("/tasks")
async def list_tasks():

    return {"supported_tasks": str(model.get_available_tasks().items())}

    
# Endpoint to handle image prediction
@app.post("/predict")
async def predict(file: UploadFile, task_type: str = Form(...), text_input: str = Form(None), api_key: APIKey = Security(get_api_key)):
    # Check if the file provided is an image
    # if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
    #     raise HTTPException(
    #         status_code=400, detail="Invalid image format. Please upload a JPEG or PNG file.")
    

    #Check if the task provided is supported
    if task_type not in model.get_available_tasks().keys():
        raise HTTPException(
            status_code=400, detail=f"Unsupported task: {task_type}. Supported tasks are {model.get_available_tasks().keys()}.")


    image = Image.open(io.BytesIO(await file.read()))
    prediction = model.predict(image=image, task_type=task_type, text_input=text_input)

    return {"task_type": task_type, "text_input": text_input, "result": prediction}

@app.get("/")
async def root():

    welcome_message = """
    <html>
        <head>
            <title>Welcome to Florence-2 Web Service</title>
        </head>
        <body>
            <h1>Welcome to the Florence-2 Model API!</h1>
            <p>This API exposes several capabilities of the Florence-2 model. Below is an overview of the capabilities of the model:</p>
            <ul>
    """
    for capability, description in model.get_cababilities().items():
        welcome_message += f"<li><strong>{capability}</strong>: {description}</li>"

    welcome_message += """
            </ul>
            <p>Use the API to perform the tasks by passing the appropriate parameters.</p>
            <p>You can find all the available tasks by using the GET tasks method from the API.</p>
            <p>For more information on how to use the API, please visit the <a href="/docs">Swagger documentation</a></p>
        </body>
    </html>
    """
    return HTMLResponse(content=welcome_message)



