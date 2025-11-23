"""
VLM + SAM3 object counting with think-and-zoom.
"""

import torch
import re
from PIL import Image
from typing import Tuple, List, Dict
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


class VLMSAMCounter:
    def __init__(self, device="cuda"):
        self.device = device

        self.vlm = Qwen3VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen3-VL-2B-Instruct",
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.vlm.eval()
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")

        sam3_model = build_sam3_image_model()
        self.sam3 = Sam3Processor(sam3_model)

    def count(self, image: Image.Image) -> Dict:
        object_type, regions = self._plan_regions(image)

        total = 0
        region_results = []

        for region in regions:
            crop = image.crop(region["coords"])
            count = self._sam3_count(crop, object_type)
            total += count
            region_results.append({
                "region": region["desc"],
                "count": count
            })

        return {
            "count": total,
            "object_type": object_type,
            "regions": region_results
        }

    def _plan_regions(self, image: Image.Image) -> Tuple[str, List[Dict]]:
        prompt = """Analyze this image for object counting. You will use a segmentation tool on specific regions.

Your task:
1. Identify the main object type
2. Decide which regions to zoom into

Response format:
OBJECT_TYPE: [e.g., "bird", "peach", "grape"]
STRATEGY: [FULL or REGIONS]

If STRATEGY is REGIONS:
REGION_1: [e.g., "top-left", "bottom-right", "left half"]
REGION_2: [region description]"""

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]
        }]

        text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt").to(self.device)

        with torch.no_grad():
            output_ids = self.vlm.generate(**inputs, max_new_tokens=256, temperature=0.5)

        response = self.processor.batch_decode(
            output_ids[:, inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )[0]

        object_type = self._parse_object_type(response)
        regions = self._parse_regions(response, image.size)

        return object_type, regions

    def _parse_object_type(self, response: str) -> str:
        response_lower = response.lower()

        if "object_type:" in response_lower:
            lines = [l for l in response.split('\n') if 'object_type:' in l.lower()]
            if lines:
                return lines[0].split(':', 1)[1].strip().lower()

        for obj in ['bird', 'peach', 'grape', 'car', 'person', 'bottle']:
            if obj in response_lower:
                return obj

        return "object"

    def _parse_regions(self, response: str, img_size: Tuple[int, int]) -> List[Dict]:
        response_lower = response.lower()
        regions = []
        w, h = img_size

        for i in range(1, 5):
            key = f"region_{i}:"
            if key in response_lower:
                lines = [l for l in response.split('\n') if key in l.lower()]
                if lines:
                    desc = lines[0].split(':', 1)[1].strip()
                    coords = self._parse_region_coords(desc, w, h)
                    regions.append({"desc": desc, "coords": coords})

        if not regions:
            regions = [{"desc": "full", "coords": (0, 0, w, h)}]

        return regions

    def _parse_region_coords(self, desc: str, w: int, h: int) -> Tuple[int, int, int, int]:
        desc_lower = desc.lower()

        if "top" in desc_lower and "left" in desc_lower:
            return (0, 0, w//2, h//2)
        elif "top" in desc_lower and "right" in desc_lower:
            return (w//2, 0, w, h//2)
        elif "bottom" in desc_lower and "left" in desc_lower:
            return (0, h//2, w//2, h)
        elif "bottom" in desc_lower and "right" in desc_lower:
            return (w//2, h//2, w, h)
        elif "left" in desc_lower:
            return (0, 0, w//2, h)
        elif "right" in desc_lower:
            return (w//2, 0, w, h)
        elif "top" in desc_lower:
            return (0, 0, w, h//2)
        elif "bottom" in desc_lower:
            return (0, h//2, w, h)

        return (0, 0, w, h)

    def _sam3_count(self, image: Image.Image, object_type: str) -> int:
        try:
            state = self.sam3.set_image(image)
            output = self.sam3.set_text_prompt(state=state, prompt=object_type)
            return len(output["masks"])
        except:
            return 0


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python count.py <image_path>")
        sys.exit(1)

    counter = VLMSAMCounter()
    image = Image.open(sys.argv[1]).convert("RGB")
    result = counter.count(image)

    print(f"Count: {result['count']}")
    print(f"Object: {result['object_type']}")
    for r in result['regions']:
        print(f"  {r['region']}: {r['count']}")
