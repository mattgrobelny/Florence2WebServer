import io
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse

from PIL import Image

from FlorenceModel import Florence2Model

app = FastAPI()
model = Florence2Model()

# Endpoint to list available tasks
@app.get("/tasks")
async def list_tasks():
    welcome_message = """
    <html>
        <head>
            <title>List of available tasks</title>
        </head>
        <body>
            <h1>Available tasks</h1>
            <ul>
    """
    for task, description in model.get_available_tasks().items():
        welcome_message += f"<li><strong>{task}</strong>: {description}</li>"

    return HTMLResponse(content=welcome_message)

    
# Endpoint to handle image prediction
@app.post("/predict")
async def predict(file:UploadFile, task_type: str = Form(...), text_input: str = Form(None)):
    print(file.content_type)
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
        </body>
    </html>
    """
    return HTMLResponse(content=welcome_message)



