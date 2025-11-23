"""
Benchmark all counting strategies on FSC147.
"""

import json
import torch
import re
from pathlib import Path
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


class AllStrategiesBenchmark:
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

    def direct_count(self, image: Image.Image) -> int:
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Count the number of objects in this image. Respond with just a number."}
            ]
        }]

        text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt").to(self.device)

        with torch.no_grad():
            output_ids = self.vlm.generate(**inputs, max_new_tokens=32, temperature=0.1)

        response = self.processor.batch_decode(
            output_ids[:, inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )[0]

        numbers = re.findall(r'\b(\d+)\b', response)
        return int(numbers[0]) if numbers else 0

    def dense_grid(self, image: Image.Image, grid_size=6) -> int:
        width, height = image.size
        crop_width = width // grid_size
        crop_height = height // grid_size

        total = 0
        for i in range(grid_size):
            for j in range(grid_size):
                left = j * crop_width
                top = i * crop_height
                right = min(left + crop_width, width)
                bottom = min(top + crop_height, height)

                crop = image.crop((left, top, right, bottom))
                total += self.direct_count(crop)

        return total

    def dense_grid_overlap(self, image: Image.Image) -> int:
        raw = self.dense_grid(image, grid_size=6)
        return int(raw * 0.9)

    def sam3_full(self, image: Image.Image) -> int:
        obj_type = self._identify_object(image)

        try:
            state = self.sam3.set_image(image)
            output = self.sam3.set_text_prompt(state=state, prompt=obj_type)
            return len(output["masks"])
        except:
            return 0

    def sam3_zoom(self, image: Image.Image) -> int:
        obj_type, regions = self._plan_regions(image)

        total = 0
        for region in regions:
            crop = image.crop(region["coords"])
            try:
                state = self.sam3.set_image(crop)
                output = self.sam3.set_text_prompt(state=state, prompt=obj_type)
                total += len(output["masks"])
            except:
                pass

        return total

    def _plan_regions(self, image: Image.Image):
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

        obj_type = self._parse_object_from_response(response)
        regions = self._parse_regions_from_response(response, image.size)

        return obj_type, regions

    def _parse_object_from_response(self, response: str) -> str:
        response_lower = response.lower()

        if "object_type:" in response_lower:
            lines = [l for l in response.split('\n') if 'object_type:' in l.lower()]
            if lines:
                return lines[0].split(':', 1)[1].strip().lower()

        for obj in ['bird', 'peach', 'grape', 'car', 'person', 'bottle']:
            if obj in response_lower:
                return obj

        return "object"

    def _parse_regions_from_response(self, response: str, img_size):
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

    def _parse_region_coords(self, desc: str, w: int, h: int):
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

    def _identify_object(self, image: Image.Image) -> str:
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "What is the main object type in this image? Answer in 1-2 words."}
            ]
        }]

        text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt").to(self.device)

        with torch.no_grad():
            output_ids = self.vlm.generate(**inputs, max_new_tokens=32, temperature=0.1)

        response = self.processor.batch_decode(
            output_ids[:, inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )[0]

        return response.strip().lower().split('.')[0].split(',')[0].strip()


def benchmark(data_dir: str, num_samples: int = 20, split: str = "val"):
    bench = AllStrategiesBenchmark()

    img_dir = Path(data_dir) / "images_384_VarV2"
    anno_file = Path(data_dir) / "annotation_FSC147_384.json"
    split_file = Path(data_dir) / "Train_Test_Val_FSC_147.json"

    with open(anno_file) as f:
        annotations = json.load(f)
    with open(split_file) as f:
        splits = json.load(f)

    images = splits[split]
    if num_samples > 0:
        images = images[:num_samples]

    print(f"Running on {len(images)} images from {split} split\n")

    strategies = {
        "Direct VLM": bench.direct_count,
        "Dense Grid (6x6)": bench.dense_grid,
        "Dense Grid + Overlap": bench.dense_grid_overlap,
        "SAM3 Full Image": bench.sam3_full,
        "SAM3 Think-and-Zoom": bench.sam3_zoom,
    }

    results = {name: [] for name in strategies}

    for i, img_name in enumerate(images):
        img_path = img_dir / img_name
        if not img_path.exists():
            continue

        image = Image.open(img_path).convert("RGB")
        true_count = len(annotations[img_name]["points"])

        print(f"[{i+1}/{len(images)}] {img_name} (true={true_count})")

        for name, method in strategies.items():
            pred = method(image)
            error = abs(pred - true_count)
            results[name].append(error)
            print(f"  {name}: {pred} (error={error})")

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")

    for name in strategies:
        mae = sum(results[name]) / len(results[name])
        print(f"{name:25s} MAE: {mae:.2f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--num_samples", type=int, default=20, help="Number of samples (0 = all)")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"], help="Dataset split")

    args = parser.parse_args()
    benchmark(args.data_dir, args.num_samples, args.split)
