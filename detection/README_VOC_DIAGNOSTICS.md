# VOC2007 Detection Diagnostics - Complete Guide

## 📋 Overview

This directory contains comprehensive diagnostic tools for debugging the VOC2007 extremely low mAP issue (0.47% vs expected 70%+) when evaluating Faster R-CNN and other detection models.

## 🎯 Problem Description

**Issue**: VOC2007 dataset evaluation shows mAP of only 0.47% instead of expected 70%+
**Affects**: Both original VOC format and COCO format conversion
**Status**: Weights load successfully, but evaluation fails

## 🛠️ Diagnostic Tools

### 1. **Full Diagnostic Analysis** (`diagnose_voc2007.py`)
**Purpose**: Comprehensive 4-phase diagnostic analysis

**Features**:
- ✅ Phase 1: Direct inference inspection on sampled images
- ✅ Phase 2: Prediction vs ground truth comparison
- ✅ Phase 3: End-to-end pipeline flow inspection
- ✅ Phase 4: Evaluation code verification

**Usage**:
```bash
python detection/diagnose_voc2007.py \
    --cfg detection/configs/global/voc2007_faster_rcnn.py \
    --num_samples 10
```

**Outputs**:
- `diagnose_output/diagnostic_report.txt` - Detailed text report
- `diagnose_output/visualizations/` - Visual comparisons (GT vs Predictions)

**When to use**: After initial setup, when you need comprehensive analysis

---

### 2. **Quick Evaluation Test** (`quick_eval_test.py`)
**Purpose**: Fast sanity checks before full diagnostics

**Features**:
- ✅ Model config and weight validation
- ✅ Annotation file structure check
- ✅ Quick inference test on sample images
- ✅ Basic statistics summary

**Usage**:
```bash
python detection/quick_eval_test.py \
    --cfg detection/configs/detector/faster_rcnn_r50_fpn_1x_voc0712.py \
    --weight work_dirs/faster_rcnn_r50_fpn_1x_voc0712/*.pth \
    --num-images 5
```

**When to use**: First step before running full diagnostics, for quick validation

---

### 3. **VOC Conversion Checker** (`check_voc_conversion.py`)
**Purpose**: Validate VOC to COCO format conversion

**Features**:
- ✅ COCO JSON structure validation
- ✅ Category ID mapping check (0-based vs 1-based)
- ✅ Cross-validation with original VOC XML annotations
- ✅ Bbox format verification
- ✅ Conversion tips and best practices

**Usage**:
```bash
# Basic check
python detection/check_voc_conversion.py \
    --json VOCdevkit/VOC2007/annotations/instances_test.json

# With XML cross-validation
python detection/check_voc_conversion.py \
    --json VOCdevkit/VOC2007/annotations/instances_test.json \
    --voc-root VOCdevkit/VOC2007 \
    --check-samples 10 \
    --show-tips
```

**When to use**: When suspect annotation format issues or class ID mapping problems

---

### 4. **Diagnostic Utilities** (`diagnose_utils.py`)
**Purpose**: Shared utility functions for diagnostics

**Contains**:
- `compute_iou_matrix()` - IoU computation between boxes
- `match_predictions_to_gt()` - Match predictions to ground truth
- `analyze_class_distribution()` - Class distribution statistics
- `check_bbox_validity()` - Bounding box validation
- `analyze_score_distribution()` - Confidence score analysis
- `check_class_id_mapping()` - Class ID mapping verification

**Usage**: Import in custom diagnostic scripts
```python
from diagnose_utils import compute_iou_matrix, match_predictions_to_gt
```

---

## 📚 Documentation

### Reference Guides
1. **DIAGNOSTICS_README.md** - Detailed user guide and troubleshooting
2. **DIAGNOSTIC_PLAN.md** - Comprehensive diagnostic plan and solutions
3. **README_VOC_DIAGNOSTICS.md** (this file) - Overview and quick start

---

## 🚀 Quick Start Guide

### Step 1: Setup
Ensure you have the required directory structure:
```
project/
├── VOCdevkit/
│   └── VOC2007/
│       ├── JPEGImages/
│       ├── Annotations/  (VOC XML format)
│       └── annotations/  (COCO JSON format)
│           ├── instances_trainval.json
│           └── instances_test.json
├── work_dirs/
│   └── faster_rcnn_r50_fpn_1x_voc0712/
│       ├── faster_rcnn_r50_fpn_1x_voc0712.py
│       └── faster_rcnn_r50_fpn_1x_voc0712_*.pth
└── detection/
    ├── configs/
    ├── diagnose_voc2007.py
    └── ...
```

### Step 2: Quick Validation
```bash
# Check annotation file
python detection/check_voc_conversion.py \
    --json VOCdevkit/VOC2007/annotations/instances_test.json \
    --show-tips

# Quick inference test
python detection/quick_eval_test.py \
    --cfg detection/configs/detector/faster_rcnn_r50_fpn_1x_voc0712.py \
    --weight work_dirs/faster_rcnn_r50_fpn_1x_voc0712/*.pth
```

### Step 3: Full Diagnostics
```bash
python detection/diagnose_voc2007.py \
    --cfg detection/configs/global/voc2007_faster_rcnn.py \
    --num_samples 20
```

### Step 4: Review Results
1. Check console output for ⚠️ warnings and ❌ errors
2. Open `diagnose_output/diagnostic_report.txt`
3. View visualizations in `diagnose_output/visualizations/`
4. Read "ROOT CAUSE ANALYSIS" section

### Step 5: Apply Fix
Based on diagnostic results, see **Common Issues** section below.

---

## 🔍 Common Issues and Solutions

### Issue 1: Class ID Mapping Mismatch ⭐ MOST COMMON
**Symptoms**: 
- mAP ~0% despite predictions existing
- Predictions don't match GT classes
- High IoU but wrong class labels

**Diagnostic Output**:
```
GT objects matched (IoU>0.5): 2/180 (1.1%)
⚠️ Predictions exist but don't match GT
```

**Root Cause**: 
- Annotations use 1-based indexing (1-20) but model expects 0-based (0-19), or vice versa
- Results in all class predictions being offset by 1

**Solution**:
Check and fix category ID mapping in COCO JSON:
```python
# Correct for VOC (1-based):
{"id": 1, "name": "aeroplane"}
{"id": 2, "name": "bicycle"}
...
{"id": 20, "name": "tvmonitor"}

# NOT (0-based with background):
{"id": 0, "name": "background"}  # ❌ Wrong!
{"id": 1, "name": "aeroplane"}
```

**See**: `DIAGNOSTIC_PLAN.md` → Issue 1 for detailed fix

---

### Issue 2: Wrong Data Preprocessing
**Symptoms**:
- Very low confidence scores (<0.1)
- Few or no predictions
- Poor detection quality

**Diagnostic Output**:
```
Average predictions per image: 0.3
Average confidence score: 0.052
⚠️ Very low predictions and confidence
```

**Root Cause**: 
Mismatch between training and evaluation preprocessing (mean/std/RGB/BGR)

**Solution**:
Verify preprocessor settings match training config:
```python
model = dict(
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[103.530, 116.280, 123.675],  # BGR, ImageNet
        std=[1.0, 1.0, 1.0],
        bgr_to_rgb=False,  # Important!
        pad_size_divisor=32
    )
)
```

**See**: `DIAGNOSTIC_PLAN.md` → Issue 2

---

### Issue 3: Annotation Format Problems
**Symptoms**:
- Evaluation crashes or returns NaN
- "Category ID not found" errors
- COCO API errors

**Diagnostic Output**:
```
❌ ERROR: Annotations reference undefined category IDs: {21, 22}
```

**Solution**:
Regenerate COCO annotations with correct format using `check_voc_conversion.py --show-tips`

**See**: `DIAGNOSTIC_PLAN.md` → Issue 3

---

### Issue 4: Test Config Too Strict
**Symptoms**:
- Predictions made but all filtered out
- Empty evaluation results

**Diagnostic Output**:
```
Number of predictions before NMS: 500
Number of predictions after NMS: 0
```

**Solution**:
Lower thresholds in test_cfg:
```python
test_cfg = dict(
    rcnn=dict(
        score_thr=0.05,  # Lower from 0.7
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100
    )
)
```

**See**: `DIAGNOSTIC_PLAN.md` → Issue 4

---

## 📊 Interpreting Diagnostic Output

### Healthy System (Expected)
```
✅ Average predictions per image: 4.5
✅ Average confidence score: 0.672
✅ Images with no predictions: 0/20
✅ GT objects matched (IoU>0.5): 156/180 (86.7%)
✅ mAP: 73.2%
```

### Broken System - Class Mapping Issue
```
⚠️ Average predictions per image: 4.2
⚠️ Average confidence score: 0.615
✅ Images with no predictions: 0/20
❌ GT objects matched (IoU>0.5): 2/180 (1.1%)  ← PROBLEM
❌ mAP: 0.47%
→ Check class ID mapping!
```

### Broken System - Preprocessing Issue
```
❌ Average predictions per image: 0.3  ← PROBLEM
❌ Average confidence score: 0.052   ← PROBLEM
❌ Images with no predictions: 14/20 ← PROBLEM
❌ GT objects matched (IoU>0.5): 0/180 (0.0%)
❌ mAP: 0.0%
→ Check data preprocessing!
```

---

## 🎓 Usage Examples

### Example 1: Full Diagnostic Workflow
```bash
# Step 1: Check annotation file
python detection/check_voc_conversion.py \
    --json VOCdevkit/VOC2007/annotations/instances_test.json \
    --voc-root VOCdevkit/VOC2007 \
    --check-samples 10

# Step 2: Quick test
python detection/quick_eval_test.py \
    --cfg detection/configs/detector/faster_rcnn_r50_fpn_1x_voc0712.py \
    --weight work_dirs/faster_rcnn_r50_fpn_1x_voc0712/*.pth

# Step 3: Full diagnostics
python detection/diagnose_voc2007.py \
    --cfg detection/configs/global/voc2007_faster_rcnn.py \
    --num_samples 20

# Step 4: Review and fix
cat diagnose_output/diagnostic_report.txt
# Apply fixes based on findings

# Step 5: Verify
python detection/run.py \
    --cfg detection/configs/global/voc2007_faster_rcnn.py \
    --eval_only
```

### Example 2: Annotation-Only Check
```bash
# Just validate the annotation file
python detection/check_voc_conversion.py \
    --json VOCdevkit/VOC2007/annotations/instances_test.json \
    --voc-root VOCdevkit/VOC2007 \
    --check-samples 20 \
    --show-tips
```

### Example 3: Quick Inference Only
```bash
# Test model can load and run inference
python detection/quick_eval_test.py \
    --cfg detection/configs/detector/faster_rcnn_r50_fpn_1x_voc0712.py \
    --weight work_dirs/faster_rcnn_r50_fpn_1x_voc0712/*.pth \
    --num-images 10
```

---

## 📝 Configuration Files

### Sample Configs Provided

1. **detection/configs/detector/faster_rcnn_r50_fpn_1x_voc0712.py**
   - Detector configuration for VOC2007
   - Uses CocoDataset with VOC metainfo
   - Configured for COCO-format annotations

2. **detection/configs/global/voc2007_faster_rcnn.py**
   - Attack/evaluation configuration
   - Points to detector config and weights
   - Evaluation-only mode supported

### Customization

To use with your own setup:

1. **Update detector config**:
```python
# In faster_rcnn_r50_fpn_1x_voc0712.py
data_root = 'path/to/your/VOCdevkit/'  # Update path
metainfo = {'classes': (...)}  # Verify class names
```

2. **Update attack config**:
```python
# In voc2007_faster_rcnn.py
detector = dict(
    cfg_file='path/to/detector/config.py',
    weight_file='path/to/weights.pth'
)
```

---

## 🐛 Troubleshooting

### Problem: "Annotation file not found"
**Solution**: Check paths in config file, ensure annotations directory exists

### Problem: "No module named 'mmdet'"
**Solution**: Install mmdetection: `pip install mmdet`

### Problem: "CUDA out of memory"
**Solution**: Reduce batch size or num_samples in diagnostic scripts

### Problem: Scripts hang/freeze
**Solution**: Check if model weights are downloading, may need to wait

### Problem: Visualizations not generated
**Solution**: Install opencv: `pip install opencv-python`

---

## 📋 Requirements

- Python 3.7+
- PyTorch 1.8+
- mmdet 3.x
- mmcv 2.x
- mmengine
- opencv-python (for visualizations)
- numpy
- terminaltables

Install with:
```bash
pip install -r requirements.txt
```

---

## 🔗 Related Resources

- **MMDetection VOC Guide**: https://mmdetection.readthedocs.io/en/latest/user_guides/train.html#train-predefined-models-on-standard-datasets
- **COCO Format Spec**: https://cocodataset.org/#format-data
- **VOC Dataset**: http://host.robots.ox.ac.uk/pascal/VOC/

---

## ✅ Expected Results After Fix

After applying the appropriate fix, you should see:

```
Evaluation Results:
+----------------+------+------+--------+-------+-------+
| class          | gts  | dets | recall | ap    | ap50  |
+----------------+------+------+--------+-------+-------+
| aeroplane      | 285  | 347  | 0.886  | 0.767 | 0.834 |
| bicycle        | 337  | 412  | 0.891  | 0.701 | 0.802 |
| ...            | ...  | ...  | ...    | ...   | ...   |
+----------------+------+------+--------+-------+-------+
| mAP            |      |      |        | 0.697 | 0.732 |
+----------------+------+------+--------+-------+-------+

bbox_mAP: 0.697
bbox_mAP_50: 0.732  ← Target achieved!
```

---

## 🤝 Support

If diagnostics don't identify the issue:

1. Check versions: `python -c "import mmdet; print(mmdet.__version__)"`
2. Review all three documentation files
3. Enable debug logging: `log_level = 'DEBUG'` in config
4. Compare with official mmdet VOC evaluation
5. Open an issue with diagnostic report attached

---

## 📄 License

This diagnostic toolkit is part of the ARES project and follows the same license.

---

**Last Updated**: 2024
**Maintainer**: ARES Development Team
