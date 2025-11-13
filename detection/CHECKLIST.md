# VOC2007 Low mAP Diagnostic Checklist

Use this checklist to systematically diagnose and fix the VOC2007 low mAP issue.

## 📋 Pre-Diagnostic Checklist

### ✅ Environment Setup
- [ ] Python 3.7+ installed
- [ ] PyTorch installed and CUDA available (optional)
- [ ] mmdet, mmcv, mmengine installed
- [ ] Required packages: `pip install opencv-python terminaltables numpy`

### ✅ Data Setup
- [ ] VOC2007 dataset downloaded
- [ ] Directory structure correct:
  ```
  VOCdevkit/
  └── VOC2007/
      ├── JPEGImages/       (4,952 test images)
      ├── Annotations/      (VOC XML files)
      └── annotations/      (COCO JSON - if converted)
          └── instances_test.json
  ```
- [ ] Annotation file exists and is readable
- [ ] Sample images can be loaded

### ✅ Model Setup
- [ ] Model weights downloaded
- [ ] Detector config file exists
- [ ] Config paths point to correct locations
- [ ] Weights can be loaded without errors

---

## 🔍 Diagnostic Steps

### Step 1: Annotation File Check (2 min)
```bash
python detection/check_voc_conversion.py \
    --json VOCdevkit/VOC2007/annotations/instances_test.json \
    --show-tips
```

**Check Results**:
- [ ] File exists and is valid JSON
- [ ] Contains 4,952 images (VOC2007 test set)
- [ ] Contains 20 categories (VOC classes)
- [ ] Category IDs are either 0-19 or 1-20 (consistent)
- [ ] No gaps in category IDs
- [ ] All annotation category_ids exist in categories
- [ ] No duplicate annotation IDs

**If fails**: Fix annotation file before proceeding
- Regenerate from VOC XML if needed
- Ensure correct category ID mapping
- See `DIAGNOSTIC_PLAN.md` → Issue 3

---

### Step 2: Quick Model Test (3 min)
```bash
python detection/quick_eval_test.py \
    --cfg detection/configs/detector/faster_rcnn_r50_fpn_1x_voc0712.py \
    --weight work_dirs/faster_rcnn_r50_fpn_1x_voc0712/*.pth \
    --num-images 5
```

**Check Results**:
- [ ] Model config loads successfully
- [ ] Weights load without errors
- [ ] Model has 20 output classes
- [ ] Data preprocessor settings displayed
- [ ] Inference runs without errors
- [ ] Average predictions per image > 1
- [ ] Average confidence score > 0.3
- [ ] Less than 50% images have no predictions

**If fails**: Check model/config issues
- Verify weight file path
- Check num_classes in config
- Verify preprocessor settings
- See `DIAGNOSTIC_PLAN.md` → Issue 2, 4

---

### Step 3: Full Diagnostics (10 min)
```bash
python detection/diagnose_voc2007.py \
    --cfg detection/configs/global/voc2007_faster_rcnn.py \
    --num_samples 20
```

**Check Outputs**:

#### Phase 1: Inference Inspection
- [ ] Model makes predictions on most images
- [ ] Predicted labels span multiple classes (not all same class)
- [ ] Confidence scores reasonable (0.3-0.9 range)
- [ ] Bounding boxes are valid (not zero/negative)
- [ ] No critical anomalies detected

**If anomalies found**: Note specific issues, check later phases

#### Phase 2: Annotation Comparison
- [ ] GT objects exist in sampled images
- [ ] GT and predicted label ranges similar
- [ ] Some label overlap between GT and predictions
- [ ] **IoU matching rate > 50%** ← CRITICAL METRIC
- [ ] Visualizations show reasonable spatial alignment

**Key Metric**: `GT objects matched (IoU>0.5): X/Y (Z%)`
- If Z% > 70%: Likely preprocessing or config issue
- If Z% < 10%: **Likely class ID mapping issue** ⚠️
- If Z% = 0%: Critical issue, check all phases

#### Phase 3: Pipeline Inspection
- [ ] Dataset type matches expected (CocoDataset/VOCDataset)
- [ ] Evaluator type matches dataset (CocoMetric/VOCMetric)
- [ ] Number of classes = 20
- [ ] Preprocessor mean/std values look correct
- [ ] Test config score_thr not too high (< 0.3)
- [ ] NMS threshold reasonable (0.3-0.7)

#### Phase 4: Evaluation Check
- [ ] Evaluator type matches annotation format
- [ ] COCO API available (if using CocoMetric)
- [ ] Category IDs in annotations match expectations
- [ ] No 0-based vs 1-based mismatch warnings
- [ ] Class names match between config and annotations

**Review Report**:
- [ ] Read `diagnose_output/diagnostic_report.txt`
- [ ] Check "ROOT CAUSE ANALYSIS" section
- [ ] Review visualization images in `diagnose_output/visualizations/`

---

## 🔧 Fix Application

Based on diagnostic results, identify the root cause and apply fix:

### Fix 1: Class ID Mapping Mismatch
**Apply if**:
- IoU matching rate < 10% despite predictions existing
- "Class ID mapping" warnings in diagnostics
- Visualizations show spatially correct but wrong classes

**Steps**:
1. [ ] Verify category IDs in COCO JSON (check_voc_conversion.py output)
2. [ ] Check if 0-based or 1-based
3. [ ] Ensure model cat2label mapping is consistent
4. [ ] Regenerate annotations if needed with correct mapping
5. [ ] Update dataset metainfo in config to match exactly

**Details**: See `DIAGNOSTIC_PLAN.md` → Issue 1

---

### Fix 2: Wrong Preprocessing
**Apply if**:
- Very low confidence scores (< 0.1)
- Few predictions per image
- "Preprocessing" warnings in diagnostics

**Steps**:
1. [ ] Check model training config preprocessing settings
2. [ ] Update detector config data_preprocessor:
   ```python
   mean=[103.530, 116.280, 123.675],  # BGR ImageNet
   std=[1.0, 1.0, 1.0],
   bgr_to_rgb=False
   ```
3. [ ] Ensure RGB vs BGR correct
4. [ ] Verify scale (0-1 vs 0-255)

**Details**: See `DIAGNOSTIC_PLAN.md` → Issue 2

---

### Fix 3: Annotation Format Issues
**Apply if**:
- Evaluation crashes with COCO API errors
- "Undefined category IDs" error
- Annotation check failed in Step 1

**Steps**:
1. [ ] Run VOC to COCO conversion from scratch
2. [ ] Use check_voc_conversion.py to validate
3. [ ] Cross-check with VOC XML files
4. [ ] Ensure all required fields present
5. [ ] Fix any ID gaps or duplicates

**Details**: See `DIAGNOSTIC_PLAN.md` → Issue 3

---

### Fix 4: Test Config Too Strict
**Apply if**:
- Predictions made but filtered out
- NMS threshold or score threshold very high
- Empty evaluation results

**Steps**:
1. [ ] Lower score_thr to 0.05 in test_cfg
2. [ ] Adjust NMS threshold to 0.5
3. [ ] Increase max_per_img if needed
4. [ ] Re-run evaluation

**Details**: See `DIAGNOSTIC_PLAN.md` → Issue 4

---

### Fix 5: Dataset Metainfo Mismatch
**Apply if**:
- "Class name not found" errors
- Index out of range
- Wrong number of classes

**Steps**:
1. [ ] Verify metainfo classes in config:
   ```python
   metainfo = {
       'classes': ('aeroplane', 'bicycle', 'bird', ...  # All 20 classes
   }
   ```
2. [ ] Ensure exact order matches training
3. [ ] Check against annotation categories
4. [ ] Update both train and test dataloader configs

**Details**: See `DIAGNOSTIC_PLAN.md` → Issue 5

---

## ✅ Verification

After applying fix:

### Step 4: Re-run Quick Test
```bash
python detection/quick_eval_test.py \
    --cfg detection/configs/detector/faster_rcnn_r50_fpn_1x_voc0712.py \
    --weight work_dirs/faster_rcnn_r50_fpn_1x_voc0712/*.pth \
    --num-images 10
```

**Expected**:
- [ ] Average predictions: 3-6 per image
- [ ] Average confidence: 0.5-0.8
- [ ] Most images have predictions

### Step 5: Full Evaluation
```bash
python detection/run.py \
    --cfg detection/configs/global/voc2007_faster_rcnn.py \
    --eval_only
```

**Expected Results** (Faster R-CNN VOC2007):
- [ ] **mAP ~ 70-75%** ← Primary goal
- [ ] mAP@0.5 ~ 73%+
- [ ] All 20 classes have reasonable AP (> 50%)
- [ ] No NaN or 0.0 values

**If still failing**:
1. [ ] Re-run full diagnostics (Step 3)
2. [ ] Check if different issue from before
3. [ ] Review all documentation files
4. [ ] Compare with official mmdet VOC evaluation

---

## 📊 Success Criteria

### ✅ Fixed Successfully If:
- [ ] mAP >= 70% (Faster R-CNN baseline)
- [ ] mAP@0.5 >= 73%
- [ ] All 20 classes detected
- [ ] AP per class > 50% for most classes
- [ ] No errors or warnings during evaluation
- [ ] Results consistent across multiple runs

### ❌ Still Has Issues If:
- [ ] mAP < 20%
- [ ] Many classes have 0% AP
- [ ] Evaluation crashes or hangs
- [ ] Warnings about undefined classes
- [ ] Inconsistent results across runs

---

## 📝 Documentation

After successful fix:

1. [ ] Document what was fixed
2. [ ] Note the root cause
3. [ ] Update config files if needed
4. [ ] Save diagnostic report for reference
5. [ ] Test on different dataset split if available

---

## 🆘 Escalation

If checklist doesn't resolve issue:

1. [ ] Collect all diagnostic outputs
2. [ ] Save diagnostic_report.txt
3. [ ] Note exact error messages
4. [ ] Check mmdet/mmcv versions:
   ```bash
   python -c "import mmdet; print(mmdet.__version__)"
   python -c "import mmcv; print(mmcv.__version__)"
   ```
5. [ ] Try with official mmdet config (not ARES):
   ```bash
   python tools/test.py \
       configs/pascal_voc/faster_rcnn_r50_fpn_1x_voc0712.py \
       checkpoints/faster_rcnn_r50_fpn_1x_voc0712.pth
   ```
6. [ ] Compare results between ARES and official mmdet

---

## 📚 Quick Reference

**Key Files**:
- Diagnostic scripts: `detection/diagnose_*.py`
- Documentation: `detection/*README*.md`, `detection/DIAGNOSTIC_PLAN.md`
- Configs: `detection/configs/`
- Outputs: `diagnose_output/`

**Key Commands**:
```bash
# Quick check
python detection/check_voc_conversion.py --json <ann_file>

# Quick test
python detection/quick_eval_test.py --cfg <cfg> --weight <weight>

# Full diagnostics
python detection/diagnose_voc2007.py --cfg <cfg> --num_samples 20

# Evaluation
python detection/run.py --cfg <cfg> --eval_only
```

**Expected Timeline**:
- Setup: 5 minutes
- Annotation check: 2 minutes
- Quick test: 3 minutes
- Full diagnostics: 10 minutes
- Fix application: 5-30 minutes (depends on issue)
- Verification: 5-20 minutes (full eval time)
- **Total**: 30-70 minutes

---

**END OF CHECKLIST**

✅ Complete each section in order
✅ Don't skip steps
✅ Document your findings
✅ Test thoroughly after fixes
