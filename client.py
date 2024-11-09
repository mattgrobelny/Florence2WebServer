import requests
import argparse
# from FlorenceModel import tasks
from fastapi import UploadFile, Form
import os
class APIClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = os.environ.get("API_URL", base_url)
        self.api_key = os.environ.get("API_KEY")

    def list_tasks(self):
        response = requests.get(f"{self.base_url}/tasks", timeout=5, headers={"access-token": self.api_key})
        return response.json()

    def predict(self, image_path:str, task_type:str = Form(...), text_input:str = Form(None)):
        with open(image_path, "rb") as image_file:
            
            files = {"file": image_file}
            data = {"task_type": task_type, 
                    "text_input": text_input}
            response = requests.post(
                f"{self.base_url}/predict", files=files, data=data, timeout=60, headers={"access-token": self.api_key})
        return response.json()


# Example usage:
if __name__ == "__main__":
    
    client = APIClient()
    tasks = client.list_tasks()
    parser = argparse.ArgumentParser(description='Provide the image path along with the task type and optional text input')
    
    parser.add_argument(
        '-p', '--print_tasks', action='store_true', help='Print available tasks')

    parser.add_argument(
        '-i', '--image_path', help='Image Path', required=False, type=str)
    
    parser.add_argument(
        '-t', '--task_type', help='Task Type', required=False, type=str)
    
    parser.add_argument(
        '-e', '--text_input', help='Optional text input', required=False, type=str)
    
    args = parser.parse_args()

    if args.print_tasks:
        print("Available tasks:")
        print(tasks)
    else:
        # Check if required arguments are provided when not printing tasks
        if not args.image_path or not args.task_type:
            parser.error("The following arguments are required: -i/--image_path, -t/--task_type (unless using -p/--print_tasks)")
        
        # print(tasks)
        # Predict using an image
        prediction = client.predict(
            args.image_path, task_type=args.task_type, text_input=args.text_input)
        
        print("Prediction:", prediction)



# python .\client.py -i .\leaves.png -t DETAILED_CAPTION
# python .\client.py -i .\leaves.png -t DETAILED_CAPTION