"""
Benchmark all counting strategies on FSC147.
"""

import json
import torch
import re
import csv
from pathlib import Path
from PIL import Image
from datetime import datetime
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

    def dense_grid_3x3(self, image: Image.Image) -> int:
        return self.dense_grid(image, grid_size=3)

    def dense_grid_3x3_overlap(self, image: Image.Image) -> int:
        raw = self.dense_grid(image, grid_size=3)
        return int(raw * 0.95)

    def dense_grid_12x12(self, image: Image.Image) -> int:
        return self.dense_grid(image, grid_size=12)

    def dense_grid_12x12_overlap(self, image: Image.Image) -> int:
        raw = self.dense_grid(image, grid_size=12)
        return int(raw * 0.85)

    def vlm_zoom(self, image: Image.Image) -> int:
        """VLM Think-and-Zoom: Use VLM to plan regions and count in each region"""
        obj_type, regions = self._plan_regions(image)

        total = 0
        for region in regions:
            crop = image.crop(region["coords"])
            count = self.direct_count(crop)
            total += count

        return total

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


def benchmark(data_dir: str, num_samples: int = 20, split: str = "val", output_csv: str = None,
              only_strategies: list = None, update_csv: str = None):
    bench = AllStrategiesBenchmark()

    img_dir = Path(data_dir) / "images_384_VarV2"
    anno_file = Path(data_dir) / "annotation_FSC147_384.json"
    split_file = Path(data_dir) / "Train_Test_Val_FSC_147.json"

    with open(anno_file) as f:
        annotations = json.load(f)
    with open(split_file) as f:
        splits = json.load(f)

    # All available strategies
    all_strategies = {
        "Direct VLM": bench.direct_count,
        "Dense Grid (3x3)": bench.dense_grid_3x3,
        "Dense Grid 3x3 + Overlap": bench.dense_grid_3x3_overlap,
        "Dense Grid (6x6)": bench.dense_grid,
        "Dense Grid + Overlap": bench.dense_grid_overlap,
        "Dense Grid (12x12)": bench.dense_grid_12x12,
        "Dense Grid 12x12 + Overlap": bench.dense_grid_12x12_overlap,
        "VLM Think-and-Zoom": bench.vlm_zoom,
        "SAM3 Full Image": bench.sam3_full,
        "SAM3 Think-and-Zoom": bench.sam3_zoom,
    }

    # If updating existing CSV, read it and get images from there
    existing_data = {}
    if update_csv:
        with open(update_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_data[row['image_name']] = row
        images = list(existing_data.keys())
        print(f"Updating {len(images)} images from existing CSV: {update_csv}\n")
    else:
        images = splits[split]
        if num_samples > 0:
            images = images[:num_samples]
        print(f"Running on {len(images)} images from {split} split\n")

    # Select strategies to run
    if only_strategies:
        strategies = {name: all_strategies[name] for name in only_strategies if name in all_strategies}
        print(f"Running only: {', '.join(strategies.keys())}\n")
    else:
        strategies = all_strategies

    results = {name: [] for name in strategies}

    # Create output CSV file
    if output_csv is None:
        if update_csv:
            output_csv = update_csv.replace('.csv', '_updated.csv')
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_csv = f"benchmark_results_{split}_{timestamp}.csv"

    csv_path = Path(output_csv)

    # Determine fieldnames based on whether we're updating
    if update_csv and existing_data:
        # Get existing fieldnames and add new strategy columns
        sample_row = list(existing_data.values())[0]
        existing_fieldnames = list(sample_row.keys())
        new_strategy_fields = []
        for name in strategies:
            if f"{name}_pred" not in existing_fieldnames:
                new_strategy_fields.extend([f"{name}_pred", f"{name}_error"])
        fieldnames = existing_fieldnames + new_strategy_fields
    else:
        # Create new fieldnames from scratch
        all_strategy_names = list(all_strategies.keys())
        fieldnames = ["image_name", "true_count"] + [f"{name}_pred" for name in all_strategy_names] + [f"{name}_error" for name in all_strategy_names]

    # Write CSV header
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()

    print(f"Saving results to: {csv_path.absolute()}\n")

    for i, img_name in enumerate(images):
        img_path = img_dir / img_name
        if not img_path.exists():
            continue

        image = Image.open(img_path).convert("RGB")
        true_count = len(annotations[img_name]["points"])

        print(f"[{i+1}/{len(images)}] {img_name} (true={true_count})")

        # Prepare row for CSV - start with existing data if updating
        if update_csv and img_name in existing_data:
            row = existing_data[img_name].copy()
        else:
            row = {"image_name": img_name, "true_count": true_count}

        # Add new strategy results
        for name, method in strategies.items():
            pred = method(image)
            error = abs(pred - true_count)
            results[name].append(error)
            row[f"{name}_pred"] = pred
            row[f"{name}_error"] = error
            print(f"  {name}: {pred} (error={error})")

        # Append to CSV after each image
        with open(csv_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
            writer.writerow(row)

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")

    for name in strategies:
        mae = sum(results[name]) / len(results[name])
        print(f"{name:25s} MAE: {mae:.2f}")

    print(f"\nDetailed results saved to: {csv_path.absolute()}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--num_samples", type=int, default=20, help="Number of samples (0 = all)")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"], help="Dataset split")
    parser.add_argument("--output_csv", type=str, default=None, help="Output CSV file path (auto-generated if not specified)")
    parser.add_argument("--update_csv", type=str, default=None, help="Update existing CSV file with new strategy results")
    parser.add_argument("--only_strategies", type=str, nargs='+', default=None,
                        help='Run only specific strategies (e.g., --only_strategies "Dense Grid (12x12)" "VLM Think-and-Zoom")')

    args = parser.parse_args()
    benchmark(args.data_dir, args.num_samples, args.split, args.output_csv, args.only_strategies, args.update_csv)
