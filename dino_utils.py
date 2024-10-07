import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

def load_dino_model():
    model_name = "IDEA-Research/grounding-dino-tiny"
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name)
    return model, processor

def apply_grounded_dino(image, model, processor, confidence_threshold=0.5):
    # Prepare input for DINO
    inputs = processor(images=image, text=["Plant"], return_tensors="pt")
    
    # Forward pass through the model
    outputs = model(**inputs)

    # Extract bounding boxes and logits
    pred_boxes = outputs['pred_boxes'][0].detach()
    pred_scores = torch.softmax(outputs['logits'][0], dim=-1)

    # Filter boxes based on confidence
    valid_boxes = []
    for box, score in zip(pred_boxes, pred_scores):
        max_score = score.max().item()
        if max_score > confidence_threshold:
            valid_boxes.append(box)
            
    return valid_boxes, pred_scores
