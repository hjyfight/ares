# VOC2007 Low mAP Diagnostics Guide

## Problem Statement
VOC2007 dataset using Faster R-CNN shows extremely low mAP of 0.47% (expected: 70%+). This occurs with both original VOC format and COCO format conversion, despite weights loading successfully.

## Diagnostic Script Usage

### Basic Usage
```bash
# Run diagnostics on VOC2007 evaluation
python detection/diagnose_voc2007.py --cfg detection/configs/global/voc2007_faster_rcnn.py --num_samples 10
```

### Prerequisites
1. **Dataset Setup**: Ensure VOC2007 dataset is properly formatted:
   ```
   VOCdevkit/
   └── VOC2007/
       ├── JPEGImages/          # Image files
       ├── Annotations/         # VOC XML annotations (for original format)
       └── annotations/         # COCO JSON annotations (for COCO format)
           ├── instances_trainval.json
           └── instances_test.json
   ```

2. **Model Weights**: Download Faster R-CNN weights trained on VOC:
   ```bash
   # Create work_dirs if it doesn't exist
   mkdir -p work_dirs/faster_rcnn_r50_fpn_1x_voc0712
   
   # Download weights (example URL, replace with actual)
   wget -P work_dirs/faster_rcnn_r50_fpn_1x_voc0712/ \
       https://download.openmmlab.com/mmdetection/v2.0/pascal_voc/faster_rcnn_r50_fpn_1x_voc0712/faster_rcnn_r50_fpn_1x_voc0712_20220320_192712-54bef0f3.pth
   ```

3. **Config Files**: 
   - Detector config: `detection/configs/detector/faster_rcnn_r50_fpn_1x_voc0712.py`
   - Attack config: `detection/configs/global/voc2007_faster_rcnn.py`

## What the Diagnostic Script Does

### Phase 1: Direct Inference Inspection
- Randomly samples N images from validation set
- Runs model inference on each image
- Records and analyzes:
  - Number of predictions per image
  - Class label distributions
  - Confidence score ranges
  - Bounding box validity
- **Anomaly Detection**:
  - All predictions have same class
  - All confidence scores below threshold
  - Invalid bounding boxes
  - No predictions at all

### Phase 2: Compare with Ground Truth
- Loads ground truth annotations for sampled images
- Compares predictions vs GT:
  - Label overlap analysis
  - IoU-based matching (IoU > 0.5 and IoU > 0.1)
  - Missing/extra classes detection
- **Class ID Mapping Check**:
  - Verifies label ranges match class count
  - Checks for 0-based vs 1-based indexing issues
- Generates visualization images showing GT (green) vs Predictions (red)

### Phase 3: Step-by-Step Flow Inspection
Traces the entire pipeline:
1. **Data Loading**: Checks DataLoader output format
2. **Preprocessing**: Verifies resize, normalization, format conversion
3. **Model Input**: Inspects tensor shapes, value ranges, device
4. **Data Preprocessor**: Checks mean/std normalization settings
5. **Model Output**: Analyzes raw predictions before/after NMS
6. **Post-processing**: Examines NMS thresholds, confidence filtering
7. **Evaluation Pipeline**: Reviews evaluator configuration

### Phase 4: Evaluation Code Check
- Examines ARES evaluation implementation
- Verifies:
  - Evaluator type (CocoMetric, VOCMetric, etc.)
  - COCO API availability and correctness
  - Annotation file format and content
  - Category ID mapping (0-based vs 1-based)
  - Class name consistency
- Compares against standard implementations

## Output Files

After running diagnostics, you'll find:

1. **diagnostic_report.txt**: Comprehensive text report with:
   - Summary statistics
   - Detected anomalies
   - Prediction vs GT comparison metrics
   - Pipeline configuration details
   - Root cause analysis

2. **visualizations/**: Directory containing comparison images:
   - Format: `comparison_{image_id}.jpg`
   - Left side: Ground truth (green boxes)
   - Right side: Predictions (red boxes with confidence scores)

## Common Issues and Solutions

### Issue 1: Class ID Mapping Mismatch
**Symptoms**: 
- Predictions have wrong class labels
- mAP near zero despite reasonable IoU

**Diagnosis**:
- Check if GT uses 0-based indexing but model expects 1-based (or vice versa)
- VOC classes are typically 1-indexed in some formats, 0-indexed in COCO format

**Solution**:
```python
# In dataset conversion, ensure proper mapping
# COCO format uses 0-based indexing for background
# VOC uses 1-based (no background class in 20 classes)
```

### Issue 2: Wrong Normalization
**Symptoms**:
- Model outputs very low confidence scores
- Poor detection quality

**Diagnosis**:
- Check data_preprocessor mean/std values
- Verify RGB vs BGR format

**Solution**:
```python
# Correct for VOC/COCO with ImageNet pretrained weights:
mean = [103.530, 116.280, 123.675]  # BGR format
std = [1.0, 1.0, 1.0]
```

### Issue 3: Annotation Format Issues
**Symptoms**:
- Evaluation fails or returns 0 mAP
- COCO API errors

**Diagnosis**:
- Check COCO JSON annotation file structure
- Verify category IDs in annotations match model expectations

**Solution**:
- Regenerate COCO format annotations from VOC XML
- Use proper conversion tools that maintain correct category mapping

### Issue 4: Test Configuration Problems
**Symptoms**:
- Too few or no detections
- All predictions filtered out

**Diagnosis**:
- Check test_cfg settings: score_thr, nms threshold, max_per_img

**Solution**:
```python
test_cfg = dict(
    rpn=dict(
        nms_pre=1000,
        max_per_img=1000,
        nms=dict(type='nms', iou_threshold=0.7),
        min_bbox_size=0
    ),
    rcnn=dict(
        score_thr=0.05,  # Lower threshold for debugging
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100
    )
)
```

## Debugging Workflow

1. **Run diagnostics first**:
   ```bash
   python detection/diagnose_voc2007.py --cfg <your_config> --num_samples 20
   ```

2. **Examine the report**: Look for critical issues in `diagnostic_report.txt`

3. **Check visualizations**: Verify model is actually detecting objects

4. **Test specific hypotheses**:
   - If class IDs seem wrong: Add debug prints to metric computation
   - If no detections: Lower score_thr in test_cfg
   - If normalization suspect: Check preprocessor settings

5. **Run full evaluation** with fixes:
   ```bash
   python detection/run.py --cfg <your_config> --eval_only
   ```

## Additional Diagnostic Commands

### Check annotation file structure:
```bash
python -c "import json; data = json.load(open('VOCdevkit/VOC2007/annotations/instances_test.json')); \
    print('Images:', len(data['images'])); \
    print('Annotations:', len(data['annotations'])); \
    print('Categories:', [(c['id'], c['name']) for c in data['categories']])"
```

### Test single image inference:
```python
from mmdet.apis import init_detector, inference_detector
import mmcv

config_file = 'detection/configs/detector/faster_rcnn_r50_fpn_1x_voc0712.py'
checkpoint_file = 'work_dirs/faster_rcnn_r50_fpn_1x_voc0712/faster_rcnn_r50_fpn_1x_voc0712_20220320_192712-54bef0f3.pth'

model = init_detector(config_file, checkpoint_file, device='cuda:0')
img = 'VOCdevkit/VOC2007/JPEGImages/000001.jpg'
result = inference_detector(model, img)

# Print results
print(f"Detected {len(result.pred_instances.bboxes)} objects")
print(f"Labels: {result.pred_instances.labels}")
print(f"Scores: {result.pred_instances.scores}")
```

## Expected Diagnostic Output

### Healthy System (70%+ mAP):
- Average 3-5 predictions per image with reasonable confidence (>0.5)
- 60-80% of GT objects matched with IoU > 0.5
- Diverse class predictions matching GT distribution
- No systematic anomalies

### Problematic System (0.47% mAP):
- Could show: No predictions, all same class, very low confidence
- <10% of GT objects matched
- Class ID mapping errors
- Systematic issues in preprocessing or evaluation

## Support

If diagnostics don't identify the root cause:
1. Save full diagnostic output
2. Check ARES and MMDetection versions
3. Verify dataset integrity
4. Compare with official MMDetection VOC eval results
