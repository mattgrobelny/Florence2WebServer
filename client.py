import requests
import argparse
from FlorenceModel import tasks
from fastapi import UploadFile, Form

class APIClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url

    def list_tasks(self):
        response = requests.get(f"{self.base_url}/tasks")
        print(response)
        return response.json()

    def predict(self, image_path:UploadFile, task_type:str = Form(...), text_input:str = Form(None)):
        # with open(image_path, "rb") as image_file:
        files = {"file": image_path}
        data = {"task_type": task_type, 
                "text_input": text_input}
        response = requests.post(
            f"{self.base_url}/predict", files=files, data=data)
        return response.json()


# Example usage:
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Provide the image path along with the task type and optional text input')
    parser.add_argument(
        '-i', '--image_path', help='Image Path', required=True, type=str)
    
    parser.add_argument(
        '-t', '--task_type', help='Task Type', required=True, type=str, choices=tasks)
    
    parser.add_argument(
        '-e', '--text_input', help='Optional text input', required=False, type=str)
    
    args = parser.parse_args()

    client = APIClient()

    # # List available tasks
    # tasks = client.list_tasks()
    # print("Available tasks:", tasks)

    # Predict using an image
    prediction = client.predict(
        args.image_path, task_type=args.task_type, text_input=args.text_input)
    
    print("Prediction:", prediction)


