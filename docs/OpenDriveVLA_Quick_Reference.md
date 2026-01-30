# OpenDriveVLA: Quick Reference Guide

**Version:** 1.0
**Date:** January 30, 2026

---

## Quick Start

### Running Inference

```bash
# 1. Activate environment
cd /workspace/opendrivevla/OpenDriveVLA
source ../opendvla/bin/activate

# 2. Run inference on filtered validation set
PYTHONPATH="$(pwd)":$PYTHONPATH \
LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH \
torchrun --nproc_per_node=1 drivevla/inference_drivevla.py \
  --num-workers 1 \
  --bf16 \
  --attn-implementation flash_attention_2 \
  --model-path checkpoints/OpenDriveVLA-0.5B \
  --visualize \
  --output output/OpenDriveVLA-0.5B/filtered_test/plan_conv.json

# 3. Generate visualizations (if not already done)
python create_visualizations_fixed.py
```

**Expected Output:**
- **Inference results:** `output/OpenDriveVLA-0.5B/filtered_test/plan_conv.json`
- **Visualizations:** `output/OpenDriveVLA-0.5B/filtered_test/visualizations_plotted/`
- **Runtime:** ~13 minutes for 162 samples

---

## Architecture at a Glance

### System Overview

```
Input → Perception → Vision-Language Bridge → Language Model → Output
  │         │                  │                      │            │
  │         │                  │                      │            └─ Natural Language
  │         │                  │                      │               + Trajectory
  │         │                  │                      │
  │         │                  │                      └─ Qwen2.5 (3.5B)
  │         │                  │
  │         │                  └─ 3 Projectors (Track, Scene, Map)
  │         │
  │         └─ UniAD (Detection, Tracking, Segmentation, Planning)
  │
  └─ 6 Multi-View Cameras
```

### Key Components

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Vision Encoder** | ResNet-101 + DCNv2 | Extract 2D features from images |
| **BEV Transformer** | BEVFormer | Convert 2D features to 3D Bird's-Eye-View |
| **Perception Tasks** | UniAD Multi-Task Heads | Detection, Tracking, Segmentation, Planning |
| **Projectors** | 3× MLP (256→3584) | Map vision features to language space |
| **Language Model** | Qwen2.5 (3.5B) | Generate natural language responses |
| **Attention** | Flash-Attention 2 | Efficient memory and speed |

---

## Model Architecture Details

### Input Processing

**Multi-View Setup:**
- 6 cameras: FRONT, FRONT_LEFT, FRONT_RIGHT, BACK, BACK_LEFT, BACK_RIGHT
- Resolution: 900×1600 pixels per camera
- Temporal: 5 frames (queue_length=5)
- Coverage: 360° surrounding view

**Preprocessing:**
```python
Normalization: mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0]
Padding: size_divisor=32
Color space: BGR (OpenCV format)
```

### Perception Pipeline

**Stage 1: Feature Extraction**

```
Input Images [6, 3, 900, 1600]
    ↓
ResNet-101 Backbone (with DCNv2 in stages 3-4)
    ↓
Multi-Scale Features:
  - P3: [6, 256, 225, 400]
  - P4: [6, 256, 113, 200]
  - P5: [6, 256, 57, 100]
  - P6: [6, 256, 29, 50]
```

**Stage 2: BEV Transformation**

```
Multi-Scale Features
    ↓
BEVFormer Encoder (6 layers):
  For each layer:
    1. Temporal Self-Attention (with previous BEV)
    2. Spatial Cross-Attention (deformable, multi-scale)
    3. Feed-Forward Network
    ↓
BEV Features [200, 200, 256]
  - Spatial Coverage: 102.4m × 102.4m
  - Resolution: 0.512m per grid cell
  - Center: Ego vehicle position
```

**Stage 3: Multi-Task Prediction**

```
BEV Features [200, 200, 256]
    ↓
┌──────────┬──────────┬──────────┬──────────┐
│  Track   │   Seg    │  Motion  │   Plan   │
│  Head    │   Head   │   Head   │   Head   │
└────┬─────┴────┬─────┴────┬─────┴────┬─────┘
     │          │          │          │
     ▼          ▼          ▼          ▼
  [N,256]   [H,W,K]   [N,12,2]   [6,2]
  Tracks    Seg Map   Traj      Ego Plan
```

### Vision-Language Interface

**Feature Extraction for VLM:**

```python
# From UniAD output, extract three types of features:

1. Track Features [N_objects, 256]
   - Object query embeddings from final decoder layer
   - Represents detected/tracked objects
   - N_objects ≈ 900 queries (filtered to ~10-50 actual objects)

2. Scene Features [256]
   - Global BEV representation
   - Aggregated via spatial pooling
   - Captures overall scene understanding

3. Map Features [N_map, 256]
   - Segmentation-derived map context
   - Road topology and lane structure
   - N_map ≈ 200-500 map elements
```

**Projection to Language Space:**

```python
# Three separate MLP projectors:

Track Projector:  [256] → [3584] → [3584]
Scene Projector:  [256] → [3584] → [3584]
Map Projector:    [256] → [3584] → [3584]

# Architecture per projector:
nn.Sequential(
    nn.Linear(256, 3584),
    nn.GELU(),
    nn.Linear(3584, 3584)
)
```

### Language Model

**Qwen2.5 Architecture:**

```
Input Embeddings [seq_len, 3584]
    ↓
28× Transformer Layers:
  - Multi-Head Self-Attention (28 heads, 128 dim each)
  - Feed-Forward Network (18944 intermediate dim)
  - RMSNorm
  - Residual Connections
    ↓
Output Logits [seq_len, 151936]
    ↓
Token Sampling (Greedy)
    ↓
Natural Language Output
```

**Special Tokens:**

| Token | Index | Purpose |
|-------|-------|---------|
| `<SCENE_TOKEN>` | -201 | Replaced with scene_features |
| `<TRACK_TOKEN>` | -202 | Replaced with track_features |
| `<MAP_TOKEN>` | -203 | Replaced with map_features |
| `<OBJECT_TOKEN>` | -204 | Instance-specific object features |

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│ STEP 1: Data Loading                                            │
│                                                                  │
│  nuScenes Sample                                                │
│  ├─ 6 Camera Images (JPEG)                                     │
│  ├─ Camera Calibration (intrinsics, extrinsics)               │
│  ├─ Ground Truth Annotations (optional)                        │
│  └─ Metadata (scene, timestamp, etc.)                          │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 2: Perception (UniAD)                                      │
│                                                                  │
│  Images → ResNet-101 → FPN → BEVFormer → Task Heads           │
│                                                                  │
│  Output:                                                        │
│  ├─ 3D Bounding Boxes [N, 10]                                  │
│  ├─ Tracking IDs [N]                                           │
│  ├─ Future Trajectories [N, 12, 2]                             │
│  ├─ Segmentation Map [200, 200, K]                             │
│  ├─ Ego Plan [6, 2]                                            │
│  └─ Features for VLM:                                          │
│      ├─ track_features [N, 256]                                │
│      ├─ scene_features [256]                                   │
│      └─ map_features [M, 256]                                  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 3: Conversation Building                                   │
│                                                                  │
│  Question: "What should the ego vehicle do?"                   │
│  Template: "Analyze <SCENE_TOKEN> <TRACK_TOKEN> <MAP_TOKEN>"  │
│                                                                  │
│  Tokenization:                                                  │
│  [1234, 5678, ..., -201, -202, -203, ...]                     │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 4: Feature Projection                                      │
│                                                                  │
│  track_features [N, 256]  → MLP → [N, 3584]                    │
│  scene_features [256]     → MLP → [3584]                       │
│  map_features [M, 256]    → MLP → [M, 3584]                    │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 5: Token Replacement                                       │
│                                                                  │
│  Input Tokens: [1234, 5678, ..., -201, -202, -203, ...]       │
│                                     ↓     ↓     ↓              │
│  Embeddings:   [e₁,  e₂, ..., scene, track, map, ...]         │
│                                [3584] [N,3584] [M,3584]        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 6: Language Model Generation                               │
│                                                                  │
│  Qwen2.5 Transformer (28 layers)                               │
│  ├─ Process combined text + vision embeddings                  │
│  ├─ Generate next tokens autoregressively                      │
│  └─ Stop at EOS token or max_length (512)                      │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 7: Output                                                  │
│                                                                  │
│  Generated Text:                                                │
│  "The ego vehicle should maintain speed and continue forward.  │
│   There is a car ahead at approximately 25 meters. Monitor    │
│   the vehicle and maintain safe following distance."           │
│                                                                  │
│  Trajectory (from UniAD planning):                             │
│  [(2.1, 0.1), (4.3, 0.2), ..., (13.6, 0.4)] meters            │
└─────────────────────────────────────────────────────────────────┘
```

---

## Configuration Quick Reference

### Model Configuration

```python
# From: projects/configs/stage1_track_map/base_track_map.py

# Spatial Settings
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]  # meters
bev_h, bev_w = 200, 200                                     # BEV grid size
canvas_size = (102.4, 102.4)                                # meters

# Temporal Settings
queue_length = 5                                             # frames

# Prediction Settings
predict_steps = 12                                           # 6 seconds
predict_modes = 6                                            # multimodal
fut_steps = 4                                                # future context
past_steps = 4                                               # past context

# Planning Settings
planning_steps = 6                                           # 3 seconds
use_col_optim = True                                         # collision opt

# Model Architecture
model.type = "UniAD"
model.img_backbone = "ResNet-101 + DCNv2"
model.img_neck = "FPN"
model.pts_bbox_head = "BEVFormerTrackHead"
```

### Dataset Configuration

```python
# Dataset
dataset_type = 'NuScenesE2EDataset'
data_root = 'data/nuscenes/'

# Annotations
ann_file_train = 'data/infos/nuscenes_infos_temporal_train.pkl'
ann_file_val = 'data/infos/nuscenes_infos_temporal_val_filtered.pkl'  # 162 samples

# Data Loading
samples_per_gpu = 1
workers_per_gpu = 1  # Reduced for memory optimization
```

### Inference Configuration

```python
# DeepSpeed Config
ds_config = {
    "bf16": {"enabled": True},
    "zero_optimization": {"stage": 0},
    "inference_mode": True
}

# Generation Config
generation_config = {
    "do_sample": False,        # Greedy decoding
    "temperature": 0,          # Deterministic
    "max_new_tokens": 512,     # Max response length
    "num_beams": 1             # No beam search
}

# Memory Optimization
attn_implementation = "flash_attention_2"
num_workers = 1
batch_size = 1
```

---

## Output Format

### Inference Output (JSON)

```json
{
  "id": "ca9a282c9e77460f8360f564131a8af5",
  "question": "What should the ego vehicle do next?",
  "answer": [
    "The ego vehicle should maintain its current speed and continue forward in the lane. There is a vehicle ahead at approximately 25 meters traveling in the same direction. Monitor the vehicle and maintain safe following distance."
  ]
}
```

### Visualization Output

**File:** `vis_{sample_id}.jpg`

**Visual Elements:**
- Background: Front camera image
- Green lines: Trajectory path
- Red dots with yellow outlines: Waypoints
- Text overlay: Trajectory status

---

## Performance Metrics

### Inference Benchmark

| Metric | Value |
|--------|-------|
| **Latency per Sample** | ~5 seconds |
| - Model Inference | ~2 seconds |
| - Data Loading | ~3 seconds |
| **Throughput** | ~12 samples/minute |
| **GPU Memory** | ~22 GB |
| **CPU Memory** | ~34 GB |
| **Success Rate** | 100% (162/162) |

### Trajectory Statistics

| Category | Percentage |
|----------|------------|
| Visible in Camera | 71.6% |
| Stopped (zero trajectory) | 25.3% |
| Too Short | 2.5% |
| Projection Issues | 0.6% |

### Memory Profile

| Component | Memory (GB) | Optimization |
|-----------|-------------|--------------|
| Model Weights | 7.2 | BF16 precision |
| Activations | 12.5 | Flash-Attention 2 |
| Data Batch | 3.8 | Single worker |
| System | 10.5 | - |
| **Total** | **34.0** | **< 45 GB limit** |

---

## Troubleshooting

### Common Issues

**1. Out of Memory (OOM)**

```bash
# Solution: Reduce batch size and workers
torchrun --nproc_per_node=1 drivevla/inference_drivevla.py \
  --num-workers 1 \
  --batch-size 1 \
  --bf16
```

**2. CUDA Error: Unsupported Architecture**

```bash
# Solution: Compile mmcv with correct architecture
export TORCH_CUDA_ARCH_LIST="8.0"  # For RTX 2000 Ada
cd third_party/mmcv_1_7_2/
MMCV_WITH_OPS=1 FORCE_CUDA=1 MAX_JOBS=8 uv pip install -e .
```

**3. Missing Image Files**

```bash
# Solution: Use filtered dataset
# Config: projects/configs/stage1_track_map/base_track_map.py
ann_file_test = 'data/infos/nuscenes_infos_temporal_val_filtered.pkl'
```

**4. Slow Inference**

```bash
# Solution: Enable flash-attention and BF16
--attn-implementation flash_attention_2 \
--bf16
```

---

## File Locations Quick Reference

### Essential Files

| File | Purpose | Path |
|------|---------|------|
| **Model Checkpoint** | Pretrained weights | `checkpoints/OpenDriveVLA-0.5B/` |
| **Inference Script** | Main entry point | `drivevla/inference_drivevla.py` |
| **Configuration** | Model config | `projects/configs/stage1_track_map/base_track_map.py` |
| **Filtered Dataset** | Available samples | `data/infos/nuscenes_infos_temporal_val_filtered.pkl` |
| **Results** | Inference output | `output/OpenDriveVLA-0.5B/filtered_test/plan_conv.json` |
| **Visualizations** | Trajectory plots | `output/OpenDriveVLA-0.5B/filtered_test/visualizations_plotted/` |

### Code Components

| Component | Path |
|-----------|------|
| **LLaVA Architecture** | `llava/model/llava_arch.py` |
| **UniAD Vision Tower** | `llava/model/multimodal_encoder/uniad_track_map.py` |
| **Projectors** | `llava/model/multimodal_projector/builder.py` |
| **Qwen Model** | `llava/model/language_model/llava_qwen.py` |
| **UniAD Detector** | `projects/mmdet3d_plugin/uniad/detectors/uniad_track.py` |
| **BEVFormer Encoder** | `projects/mmdet3d_plugin/uniad/modules/encoder.py` |
| **Dataset** | `drivevla/data_utils/nuscenes_llava_dataset.py` |

---

## Additional Resources

### Documentation

- **Installation Guide:** `docs/1_INSTALL.md`
- **Data Preparation:** `docs/2_DATA_PREP.md`
- **Evaluation Guide:** `docs/3_EVAL.md`
- **Status File:** `/workspace/status_updated.md`

### External Links

- **Checkpoint:** https://huggingface.co/OpenDriveVLA/OpenDriveVLA-0.5B
- **Flash-Attention:** https://github.com/Dao-AILab/flash-attention
- **nuScenes Dataset:** https://www.nuscenes.org/

### Support

For issues and questions:
- Check the status file: `/workspace/status_updated.md`
- Review logs in: `output/OpenDriveVLA-0.5B/`
- Check this technical report: `/workspace/docs/OpenDriveVLA_Technical_Report.md`

---

**Last Updated:** January 30, 2026
