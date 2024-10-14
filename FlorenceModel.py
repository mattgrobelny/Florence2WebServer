import torch
from transformers import  AutoModelForCausalLM, AutoProcessor
from PIL import Image
import supervision as sv
import numpy as np

tasks = ["CAPTION",
         "DETAILED_CAPTION",
         "MORE_DETAILED_CAPTION",
         "CAPTION_TO_PHRASE_GROUNDING",
         "REFERRING_EXPRESSION_SEGMENTATION",
         "REGION_TO_SEGMENTATION",
         "OD",
         "OPEN_VOCABULARY_DETECTION",
         "DENSE_REGION_CAPTION",
         "REGION_PROPOSAL",
         "OCR",
         "OCR_WITH_REGION",
         "REGION_TO_CATEGORY",
         "REGION_TO_DESCRIPTION"]

class Florence2Model:
    def __init__(self) -> None:

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self.processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base", torch_dtype=self.torch_dtype, trust_remote_code=True).to(self.device)

    def get_available_tasks(self):
        tasks = ["CAPTION",
                 "DETAILED_CAPTION",
                 "MORE_DETAILED_CAPTION",
                 "CAPTION_TO_PHRASE_GROUNDING",
                 "REFERRING_EXPRESSION_SEGMENTATION",
                 "REGION_TO_SEGMENTATION",
                 "OD",
                 "OPEN_VOCABULARY_DETECTION",
                 "DENSE_REGION_CAPTION",
                 "REGION_PROPOSAL",
                 "OCR",
                 "OCR_WITH_REGION",
                 "REGION_TO_CATEGORY",
                 "REGION_TO_DESCRIPTION"]
        
        descriptions = ["Generates a caption for the image",
                        "Generates a detailed caption for the image",
                        "Generates an even more detailed caption for the image",
                        "Generates specific captions to corresponding regions in the image",
                        "Segments regions based on a phrase",
                        "converts identified segmented regions into segmented areas ",
                        "Performs object detection",
                        "Detects objects in the image from an open set of categories not constrained from a vocabulary",
                        "Generates captions for multiple dense regions",
                        "Generates regions in the image taht might contain objects of interest",
                        "Extracts text from the image",
                        "Extracts text from the image with detections of where the text appears in the image",
                        "Classifies specific regions within the image",
                        "Generates descriptive text for specific regions in the image"]
        
        return {task:desc for task, desc in zip(tasks, descriptions)}


    def get_cababilities(self):

        model_capabilities = {
            "Object Detection": "Identifies and locates objects in an image.",
            "Image Segmentation": "Segments different regions of an image.",
            "Image Captioning": "Generates captions based on the content of an image.",
        }
        return model_capabilities
    
    def predict(self, image: Image, task_type: str, text_input: str):
        task_type = "<"+task_type+">"
        if text_input is None:
            prompt = task_type
        else:
            prompt = task_type + text_input

        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device, self.torch_dtype)

        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
            do_sample=False)
        
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = self.processor.post_process_generation(generated_text, task=task_type, image_size=(image.width, image.height))
        # print(parsed_answer)
        # if "bboxes" in parsed_answer[task_type].keys():
                
        #     bboxes = np.array(parsed_answer[task_type]["bboxes"])
        #     labels = parsed_answer[task_type]["labels"]

        #     print(bboxes, bboxes.shape)
        #     detections = sv.Detections(
        #         xyxy=bboxes)
            
        #     print(detections)
        
        return parsed_answer
    

