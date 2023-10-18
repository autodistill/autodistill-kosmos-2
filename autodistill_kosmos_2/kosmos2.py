import os
from dataclasses import dataclass

import torch

import supervision as sv
from autodistill.detection import CaptionOntology, DetectionBaseModel
import numpy as np

from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

HOME = os.path.expanduser("~")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class Kosmos2(DetectionBaseModel):
    ontology: CaptionOntology
    
    def __init__(self, ontology: CaptionOntology):
        self.ontology = ontology
        self.model = AutoModelForVision2Seq.from_pretrained("ydshieh/kosmos-2-patch14-224", trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained("ydshieh/kosmos-2-patch14-224", trust_remote_code=True)

    def predict(self, input: str) -> sv.Detections:
        prompt = "<grounding>An image of"

        prompts = self.ontology.prompts()

        image = Image.open(input).raw

        inputs = self.processor(text=prompt, images=image, return_tensors="pt")

        generated_ids = self.model.generate(
            pixel_values=inputs["pixel_values"],
            input_ids=inputs["input_ids"][:, :-1],
            attention_mask=inputs["attention_mask"][:, :-1],
            img_features=None,
            img_attn_mask=inputs["img_attn_mask"][:, :-1],
            use_cache=True,
            max_new_tokens=64,
        )

        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        processed_text = self.processor.post_process_generation(generated_text, cleanup_and_extract=False)

        processed_text, entities = self.processor.post_process_generation(generated_text)

        print(processed_text, entities)

        # entities[3] is the grounding
        xyxys = [e[3] for e in entities]
        class_ids = [prompts.index(e[0]) for e in entities]

        # filter out classes not in user provided prompts
        xyxys = [xyxy for xyxy, class_id in zip(xyxys, class_ids) if class_id != -1]
        class_ids = [class_id for class_id in class_ids if class_id != -1]

        return sv.Detections(
            xyxy=np.array(xyxys),
            class_id=np.array(class_ids),
            confidence=np.array([1.0] * len(class_ids)),
        )