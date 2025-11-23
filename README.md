# VLM + SAM3 Counting

Object counting using Qwen3-VL for spatial reasoning and SAM3 for segmentation.

## Setup

```bash
pip install -r requirements.txt

# Install SAM3
git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e .
cd ..

# Login to HuggingFace (need access to facebook/sam3)
huggingface-cli login
```

## Usage

Single image:
```bash
python count.py image.jpg
```

Benchmark on FSC147:
```bash
python benchmark.py --data_dir /path/to/FSC147 --num_samples 20
```

## Results

FSC147 validation (20 images, results vary due to VLM stochasticity):

**Best observed:**
- SAM3 Think-and-Zoom: MAE 17.70
- SAM3 Full Image: MAE 20.70
- Dense Grid + Overlap: MAE 23.70

**Typical range:**
- SAM3 approaches: MAE 18-25
- Dense Grid methods: MAE 24-29
- Direct VLM: MAE 48-52

Note: Results vary between runs due to temperature settings and VLM object type identification.
