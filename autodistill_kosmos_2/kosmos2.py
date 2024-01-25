import os
from dataclasses import dataclass

import numpy as np
import supervision as sv
import torch
from autodistill.detection import CaptionOntology, DetectionBaseModel
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

HOME = os.path.expanduser("~")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class Kosmos2(DetectionBaseModel):
    ontology: CaptionOntology

    def __init__(self, ontology: CaptionOntology):
        self.ontology = ontology
        self.model = AutoModelForVision2Seq.from_pretrained(
            "microsoft/kosmos-2-patch14-224", trust_remote_code=True
        ).to(DEVICE)
        self.processor = AutoProcessor.from_pretrained(
            "microsoft/kosmos-2-patch14-224", trust_remote_code=True
        )

    def predict(self, input: str) -> sv.Detections:
        prompt = "<grounding>An image of"

        prompts = self.ontology.prompts()

        image = Image.open(input)

        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(
            DEVICE
        )

        generated_ids = self.model.generate(
            pixel_values=inputs["pixel_values"],
            input_ids=inputs["input_ids"][:, :-1],
            attention_mask=inputs["attention_mask"][:, :-1],
            image_embeds=None,
            image_embeds_position_mask=inputs["image_embeds_position_mask"][:, :-1],
            use_cache=True,
            max_new_tokens=64,
        )

        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]

        processed_text = self.processor.post_process_generation(
            generated_text, cleanup_and_extract=False
        )

        processed_text, entities = self.processor.post_process_generation(
            generated_text
        )

        # entities[3] is the grounding
        xyxys = [e[2] for e in entities]
        class_ids = [prompts.index(e[0]) if e[0] in prompts else -1 for e in entities]

        # filter out classes not in user provided prompts
        xyxys = [xyxy for xyxy, class_id in zip(xyxys, class_ids) if class_id != -1]
        class_ids = [class_id for class_id in class_ids if class_id != -1]

        final_xyxys = []
        final_class_ids = []

        for i, item in enumerate(xyxys):
            for sublist in item:
                final_xyxys.append(sublist)
                final_class_ids.append(class_ids[i])

        # scale class_ids to image coords, not 0-1
        for i, xyxy in enumerate(final_xyxys):
            final_xyxys[i] = [
                xyxy[0] * image.width,
                xyxy[1] * image.height,
                xyxy[2] * image.width,
                xyxy[3] * image.height,
            ]

        if len(xyxys) == 0:
            return sv.Detections.empty()

        return sv.Detections(
            xyxy=np.array(final_xyxys),
            class_id=np.array(final_class_ids),
            confidence=np.array([1.0] * len(final_class_ids)),
        )
