# VOC2007 Low mAP Diagnostic Plan & Solution Guide

## Problem Summary
**Issue**: VOC2007 Faster R-CNN evaluation shows 0.47% mAP (expected 70%+)
**Status**: Weights load successfully, occurs with both VOC and COCO format
**Impact**: Critical - prevents proper model evaluation

---

## Diagnostic Tools Created

### 1. Main Diagnostic Script
**File**: `detection/diagnose_voc2007.py`
**Purpose**: Comprehensive 4-phase diagnostic analysis
**Usage**:
```bash
python detection/diagnose_voc2007.py \
    --cfg detection/configs/global/voc2007_faster_rcnn.py \
    --num_samples 10
```

**Output**:
- `diagnose_output/diagnostic_report.txt` - Full diagnostic report
- `diagnose_output/visualizations/` - Comparison images (GT vs predictions)

**What it checks**:
- ✅ Direct inference results on sample images
- ✅ Prediction vs ground truth comparison
- ✅ Complete pipeline flow (data loading → evaluation)
- ✅ ARES evaluation code correctness

### 2. Quick Evaluation Test
**File**: `detection/quick_eval_test.py`
**Purpose**: Fast sanity checks before full diagnostics
**Usage**:
```bash
python detection/quick_eval_test.py \
    --cfg detection/configs/detector/faster_rcnn_r50_fpn_1x_voc0712.py \
    --weight work_dirs/faster_rcnn_r50_fpn_1x_voc0712/faster_rcnn_r50_fpn_1x_voc0712_20220320_192712-54bef0f3.pth \
    --num-images 5
```

**What it checks**:
- ✅ Model config and weight file existence
- ✅ Annotation file format and structure
- ✅ Quick inference on sample images
- ✅ Basic prediction statistics

### 3. VOC to COCO Conversion Checker
**File**: `detection/check_voc_conversion.py`
**Purpose**: Verify annotation format conversion correctness
**Usage**:
```bash
python detection/check_voc_conversion.py \
    --json VOCdevkit/VOC2007/annotations/instances_test.json \
    --voc-root VOCdevkit/VOC2007 \
    --check-samples 10 \
    --show-tips
```

**What it checks**:
- ✅ COCO JSON structure and completeness
- ✅ Category ID mapping (0-based vs 1-based)
- ✅ Class name consistency with VOC
- ✅ Cross-validation with original VOC XML annotations
- ✅ Bounding box format correctness

### 4. Diagnostic Utilities
**File**: `detection/diagnose_utils.py`
**Purpose**: Common functions for diagnostics
**Contains**:
- IoU computation and matching
- Class distribution analysis
- Bbox validity checks
- Score distribution analysis
- Class ID mapping checks

---

## Common Root Causes & Solutions

### ⚠️ Issue 1: Class ID Mapping Mismatch (Most Likely)

**Symptoms**:
- mAP near zero despite reasonable detection outputs
- Predictions exist but don't match GT
- Class IDs seem shifted

**Root Cause**:
```python
# VOC has 20 classes, but indexing can be:
# Option A: 0-based (0-19) - COCO style
# Option B: 1-based (1-20) - Traditional VOC style

# If annotation uses 1-based but model expects 0-based:
# GT label 1 (aeroplane) → Model thinks it's "bicycle"
# Result: Zero matches, 0% mAP
```

**How to Diagnose**:
```bash
python detection/check_voc_conversion.py --json <ann_file> --show-tips
```

Look for:
- Category IDs in annotations: `[0] aeroplane` vs `[1] aeroplane`
- Model num_classes vs actual categories
- Label range warnings

**Solution A**: Fix annotation file
```python
# If annotations are 0-based but should be 1-based:
# Regenerate annotations with correct mapping
# Or programmatically adjust category IDs in JSON
```

**Solution B**: Fix model config
```python
# In detector config, ensure cat2label mapping is correct
# Check: ares/attack/detection/custom/coco_dataset.py line 76
self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}

# If using ARES CocoDataset, it auto-generates mapping
# Ensure metainfo classes match annotation category names exactly
```

### ⚠️ Issue 2: Wrong Data Preprocessing

**Symptoms**:
- Very low confidence scores (<0.1)
- Model outputs predictions but poor quality
- Random-looking predictions

**Root Cause**:
```python
# Model trained with ImageNet preprocessing:
mean = [103.530, 116.280, 123.675]  # BGR
std = [1.0, 1.0, 1.0]

# But evaluation uses different normalization:
mean = [123.675, 116.280, 103.530]  # RGB (wrong order!)
# Or wrong scale: [0.485, 0.456, 0.406]  # 0-1 scale instead of 0-255
```

**How to Diagnose**:
```bash
python detection/diagnose_voc2007.py --cfg <config>
```

Check diagnostic report for:
- "Preprocessor mean: ..." and "Preprocessor std: ..."
- Compare with model training config

**Solution**:
```python
# In detector config file:
model = dict(
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[103.530, 116.280, 123.675],  # Must match training
        std=[1.0, 1.0, 1.0],               # Must match training
        bgr_to_rgb=False,  # Important: check RGB vs BGR
        pad_size_divisor=32
    ),
    ...
)
```

### ⚠️ Issue 3: Annotation File Issues

**Symptoms**:
- COCO API errors during evaluation
- "Annotation ids not unique" warnings
- Evaluation returns NaN or 0

**Root Cause**:
- Malformed COCO JSON
- Missing required fields
- Category ID mismatches
- Duplicate annotation IDs

**How to Diagnose**:
```bash
python detection/check_voc_conversion.py \
    --json VOCdevkit/VOC2007/annotations/instances_test.json \
    --voc-root VOCdevkit/VOC2007 \
    --check-samples 10
```

**Solution**:
1. **Regenerate annotations from VOC XML**:
```bash
# Using mmdetection tools (if available):
python tools/dataset_converters/pascal_voc.py \
    VOCdevkit/ \
    --out-format coco

# Or use custom conversion script ensuring:
# - Contiguous category IDs (1-20 or 0-19)
# - Unique annotation IDs
# - Correct bbox format [x, y, w, h]
# - All category_ids in annotations exist in categories
```

2. **Verify generated JSON**:
```python
import json
with open('annotations/instances_test.json', 'r') as f:
    data = json.load(f)

# Check structure
assert 'images' in data
assert 'annotations' in data
assert 'categories' in data

# Check IDs
cat_ids = set(cat['id'] for cat in data['categories'])
ann_cat_ids = set(ann['category_id'] for ann in data['annotations'])
assert ann_cat_ids.issubset(cat_ids), "Undefined category IDs!"

# Check uniqueness
ann_ids = [ann['id'] for ann in data['annotations']]
assert len(ann_ids) == len(set(ann_ids)), "Duplicate annotation IDs!"
```

### ⚠️ Issue 4: Test Configuration Too Strict

**Symptoms**:
- Model makes predictions internally but output is empty
- Diagnostic shows predictions but evaluator sees none
- All detections filtered out

**Root Cause**:
```python
# Test config with too high thresholds:
test_cfg = dict(
    rcnn=dict(
        score_thr=0.7,  # Too high! Filters out most detections
        nms=dict(type='nms', iou_threshold=0.3),  # Too strict NMS
        max_per_img=100
    )
)
```

**How to Diagnose**:
```bash
python detection/diagnose_voc2007.py --cfg <config>
```

Look for:
- "Number of predictions after NMS: 0" or very low
- Phase 3 output showing test_cfg settings

**Solution**:
```python
# In detector config, relax test_cfg:
model = dict(
    ...
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0
        ),
        rcnn=dict(
            score_thr=0.05,  # Lower threshold for VOC
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100
        )
    )
)
```

### ⚠️ Issue 5: Dataset Metainfo Mismatch

**Symptoms**:
- "Class name not found" errors
- Index out of range errors
- Predicted label IDs outside valid range

**Root Cause**:
```python
# Dataset config has wrong or incomplete classes:
metainfo = {
    'classes': ('person', 'car', ...)  # Only 2 classes!?
}

# Or wrong order:
metainfo = {
    'classes': ('bicycle', 'aeroplane', ...)  # Wrong order!
}
```

**How to Diagnose**:
Check config files for metainfo definition.

**Solution**:
```python
# Use exact VOC2007 class order:
metainfo = {
    'classes': (
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 
        'bus', 'car', 'cat', 'chair', 'cow', 
        'diningtable', 'dog', 'horse', 'motorbike', 'person', 
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    )
}

# This MUST match:
# 1. Annotation category order
# 2. Model training class order
# 3. Cat2label mapping
```

---

## Step-by-Step Diagnostic Workflow

### Phase 1: Pre-Flight Checks (2 minutes)
```bash
# 1. Check files exist
ls -lh VOCdevkit/VOC2007/annotations/instances_test.json
ls -lh work_dirs/faster_rcnn_r50_fpn_1x_voc0712/*.pth

# 2. Quick annotation check
python detection/check_voc_conversion.py \
    --json VOCdevkit/VOC2007/annotations/instances_test.json \
    --show-tips

# 3. Quick inference test
python detection/quick_eval_test.py \
    --cfg detection/configs/detector/faster_rcnn_r50_fpn_1x_voc0712.py \
    --weight work_dirs/faster_rcnn_r50_fpn_1x_voc0712/*.pth
```

**Expected Output**:
- ✅ All files exist
- ✅ 20 categories, IDs 1-20 or 0-19 (consistent)
- ✅ Average 3-5 predictions per image
- ✅ Average confidence > 0.3

**If any check fails**: Fix before proceeding.

### Phase 2: Full Diagnostics (5-10 minutes)
```bash
python detection/diagnose_voc2007.py \
    --cfg detection/configs/global/voc2007_faster_rcnn.py \
    --num_samples 20
```

**Review outputs**:
1. **Console output**: Look for ⚠️ warnings and ❌ errors
2. **diagnostic_report.txt**: Read "ROOT CAUSE ANALYSIS" section
3. **visualizations/**: Visually confirm predictions vs GT

### Phase 3: Targeted Fix
Based on diagnostic results, apply appropriate solution from above.

### Phase 4: Verification
```bash
# Run full evaluation
python detection/run.py \
    --cfg detection/configs/global/voc2007_faster_rcnn.py \
    --eval_only
```

**Expected Result**: mAP ~70% for Faster R-CNN on VOC2007

---

## Quick Reference: Diagnostic Output Interpretation

### Good System (70%+ mAP)
```
Average predictions per image: 4.5
Average confidence score: 0.672
Images with no predictions: 0/20
GT objects matched (IoU>0.5): 156/180 (86.7%)
```

### Broken System - Class ID Issue (0.47% mAP)
```
Average predictions per image: 4.2
Average confidence score: 0.615
Images with no predictions: 0/20
GT objects matched (IoU>0.5): 2/180 (1.1%)  ← PROBLEM!

⚠️ Predictions exist but don't match GT
→ Likely class ID mapping issue
```

### Broken System - Preprocessing Issue (0.47% mAP)
```
Average predictions per image: 0.3  ← PROBLEM!
Average confidence score: 0.052   ← PROBLEM!
Images with no predictions: 14/20 ← PROBLEM!
GT objects matched (IoU>0.5): 0/180 (0.0%)

⚠️ Very low predictions and confidence
→ Likely preprocessing/normalization issue
```

### Broken System - Annotation Issue (0% mAP)
```
ERROR: Annotation file not found
OR
ERROR: Annotations reference undefined category IDs: {21, 22}
OR
ERROR: Duplicate annotation IDs

→ Regenerate annotation file
```

---

## Files Created for This Task

1. **detection/diagnose_voc2007.py** - Main diagnostic script (4-phase analysis)
2. **detection/quick_eval_test.py** - Fast pre-flight checks
3. **detection/check_voc_conversion.py** - Annotation format validator
4. **detection/diagnose_utils.py** - Shared diagnostic utilities
5. **detection/configs/detector/faster_rcnn_r50_fpn_1x_voc0712.py** - Sample detector config
6. **detection/configs/global/voc2007_faster_rcnn.py** - Sample attack config
7. **detection/DIAGNOSTICS_README.md** - User guide
8. **detection/DIAGNOSTIC_PLAN.md** - This comprehensive plan

---

## Next Steps

1. **Run diagnostics** with your actual config and data
2. **Identify root cause** from diagnostic output
3. **Apply appropriate fix** from solutions above
4. **Verify** with full evaluation
5. **Document** what was fixed for future reference

---

## Support & Debugging

If diagnostics don't identify the issue:

1. **Check versions**:
   ```bash
   python -c "import mmdet; print(mmdet.__version__)"
   python -c "import mmcv; print(mmcv.__version__)"
   ```

2. **Enable debug mode** in config:
   ```python
   log_level = 'DEBUG'
   ```

3. **Compare with official MMDetection**:
   ```bash
   # Test same weights with official mmdet config
   python tools/test.py \
       configs/pascal_voc/faster_rcnn_r50_fpn_1x_voc0712.py \
       checkpoints/faster_rcnn_r50_fpn_1x_voc0712.pth
   ```

4. **Save intermediate outputs**:
   - Modify diagnostic script to save more details
   - Add debug prints in ARES evaluation code
   - Compare with known-good evaluation results

---

**END OF DIAGNOSTIC PLAN**
