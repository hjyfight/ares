"""
Comprehensive diagnostic script for VOC2007 extremely low mAP issue with Faster R-CNN.

This script performs 4 main diagnostics:
1. Direct inference inspection on sample images
2. Compare predictions with ground truth annotations
3. Step-by-step flow inspection (data loading, preprocessing, model I/O, post-processing)
4. Check ARES evaluation code logic

Usage:
    python detection/diagnose_voc2007.py --cfg path/to/voc_config.py [--num_samples N]
"""

import os
import sys
import argparse
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.distributed as dist
import numpy as np
import cv2
import json
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

from mmengine.config import Config, DictAction
from mmengine.runner import Runner
from mmengine.registry import MODELS, DefaultScope
from mmengine.evaluator.evaluator import Evaluator
from mmdet.datasets.api_wrappers import COCO

# Import ARES components
from ares.attack.detection.utils import HiddenPrints
from ares.attack.detection.custom import CocoDataset, CocoMetric


class VOCDiagnostics:
    """Comprehensive diagnostics for VOC2007 detection evaluation."""
    
    def __init__(self, cfg_path: str, num_samples: int = 10):
        """Initialize diagnostics.
        
        Args:
            cfg_path: Path to detection config file
            num_samples: Number of sample images to inspect
        """
        self.cfg_path = cfg_path
        self.num_samples = num_samples
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load configs
        print("=" * 80)
        print("Loading configurations...")
        self.attack_cfg = Config.fromfile(cfg_path)
        self.detector_cfg = Config.fromfile(self.attack_cfg.detector.cfg_file)
        
        # Modify configs for evaluation
        self.detector_cfg.test_dataloader.batch_size = 1  # Single image for diagnosis
        self.detector_cfg.test_dataloader.dataset.filter_cfg = dict(filter_empty_gt=True)
        
        if not self.detector_cfg.model.data_preprocessor.get('mean', False):
            self.detector_cfg.model.data_preprocessor.mean = [0.0] * 3
        if not self.detector_cfg.model.data_preprocessor.get('std', False):
            self.detector_cfg.model.data_preprocessor.std = [1.0] * 3
        
        # Initialize model, dataloader, and evaluator
        print("\nInitializing model, dataloader, and evaluator...")
        with HiddenPrints():
            DefaultScope.get_instance('diagnose', scope_name='mmdet')
            self.detector = MODELS.build(self.detector_cfg.model)
            self.detector.eval()
            self.detector.to(self.device)
            
            self.test_dataloader = Runner.build_dataloader(
                self.detector_cfg.test_dataloader, seed=0, diff_rank_seed=False
            )
            self.evaluator = Evaluator(self.detector_cfg.test_evaluator)
            self.evaluator.dataset_meta = self.test_dataloader.dataset.metainfo
        
        print(f"✓ Model: {self.detector_cfg.model.type}")
        print(f"✓ Weight file: {self.attack_cfg.detector.weight_file}")
        print(f"✓ Dataset: {len(self.test_dataloader.dataset)} images")
        print(f"✓ Classes: {self.test_dataloader.dataset.metainfo['classes']}")
        print(f"✓ Number of classes: {len(self.test_dataloader.dataset.metainfo['classes'])}")
        
        # Create output directory
        self.output_dir = Path("diagnose_output")
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "visualizations").mkdir(exist_ok=True)
        
    def run_full_diagnostics(self):
        """Run all 4 diagnostic phases."""
        print("\n" + "=" * 80)
        print("STARTING COMPREHENSIVE DIAGNOSTICS FOR VOC2007 LOW mAP ISSUE")
        print("=" * 80)
        
        # Phase 1: Direct inference inspection
        print("\n" + "=" * 80)
        print("PHASE 1: DIRECT INFERENCE INSPECTION")
        print("=" * 80)
        inference_results = self.phase1_inference_inspection()
        
        # Phase 2: Compare with annotations
        print("\n" + "=" * 80)
        print("PHASE 2: COMPARE PREDICTIONS WITH GROUND TRUTH")
        print("=" * 80)
        annotation_comparison = self.phase2_annotation_comparison(inference_results)
        
        # Phase 3: Step-by-step flow inspection
        print("\n" + "=" * 80)
        print("PHASE 3: STEP-BY-STEP FLOW INSPECTION")
        print("=" * 80)
        flow_results = self.phase3_flow_inspection()
        
        # Phase 4: Check evaluation code
        print("\n" + "=" * 80)
        print("PHASE 4: CHECK ARES EVALUATION CODE")
        print("=" * 80)
        eval_check = self.phase4_evaluation_check()
        
        # Generate final report
        self.generate_report(inference_results, annotation_comparison, flow_results, eval_check)
        
    def phase1_inference_inspection(self) -> List[Dict]:
        """Phase 1: Randomly sample images and inspect model outputs."""
        print(f"\nSampling {self.num_samples} random images from validation set...")
        
        # Random sample indices
        total_images = len(self.test_dataloader.dataset)
        sample_indices = np.random.choice(total_images, min(self.num_samples, total_images), replace=False)
        
        results = []
        for idx in sample_indices:
            print(f"\n--- Processing image {idx} ---")
            
            # Get data sample
            data_sample = self.test_dataloader.dataset[idx]
            img_path = data_sample['data_samples'].img_path
            img_id = data_sample['data_samples'].img_id
            
            print(f"Image path: {img_path}")
            print(f"Image ID: {img_id}")
            
            # Prepare input
            inputs = data_sample['inputs'].unsqueeze(0).to(self.device)
            data_samples = [data_sample['data_samples']]
            
            print(f"Input tensor shape: {inputs.shape}")
            print(f"Input tensor range: [{inputs.min().item():.3f}, {inputs.max().item():.3f}]")
            print(f"Input tensor mean: {inputs.mean().item():.3f}, std: {inputs.std().item():.3f}")
            
            # Run inference
            with torch.no_grad():
                predictions = self.detector.test_step([{'inputs': inputs, 'data_samples': data_samples}])
            
            pred_instances = predictions[0].pred_instances
            num_predictions = len(pred_instances.bboxes)
            
            print(f"Number of predictions: {num_predictions}")
            
            if num_predictions > 0:
                bboxes = pred_instances.bboxes.cpu().numpy()
                scores = pred_instances.scores.cpu().numpy()
                labels = pred_instances.labels.cpu().numpy()
                
                print(f"Predicted labels: {labels}")
                print(f"Predicted scores range: [{scores.min():.3f}, {scores.max():.3f}]")
                print(f"Predicted scores mean: {scores.mean():.3f}")
                print(f"Number of unique labels: {len(np.unique(labels))}")
                print(f"Unique labels: {np.unique(labels)}")
                
                # Check for anomalies
                anomalies = []
                if len(np.unique(labels)) == 1:
                    anomalies.append("All predictions have same class")
                if scores.max() < 0.1:
                    anomalies.append("All scores below 0.1")
                if scores.max() > 0.99 and num_predictions > 10:
                    anomalies.append("Suspiciously high confidence")
                
                # Check bbox validity
                invalid_boxes = 0
                for bbox in bboxes:
                    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                        invalid_boxes += 1
                if invalid_boxes > 0:
                    anomalies.append(f"{invalid_boxes} invalid bounding boxes")
                
                if anomalies:
                    print("⚠️  ANOMALIES DETECTED:")
                    for anomaly in anomalies:
                        print(f"   - {anomaly}")
            else:
                print("⚠️  No predictions made!")
                anomalies = ["No predictions"]
            
            # Store results
            result = {
                'idx': idx,
                'img_path': img_path,
                'img_id': img_id,
                'num_predictions': num_predictions,
                'predictions': predictions[0],
                'data_sample': data_sample,
                'anomalies': anomalies if num_predictions > 0 else ["No predictions"]
            }
            results.append(result)
        
        return results
    
    def phase2_annotation_comparison(self, inference_results: List[Dict]):
        """Phase 2: Compare predictions with ground truth annotations."""
        print("\nComparing predictions with ground truth annotations...")
        
        comparison_results = []
        
        for result in inference_results:
            idx = result['idx']
            print(f"\n--- Comparing image {idx} ---")
            
            # Get ground truth
            gt_instances = result['data_sample']['data_samples'].gt_instances
            gt_bboxes = gt_instances.bboxes.numpy()
            gt_labels = gt_instances.labels.numpy()
            
            print(f"GT: {len(gt_bboxes)} objects")
            print(f"GT labels: {gt_labels}")
            print(f"GT unique labels: {np.unique(gt_labels)}")
            
            # Get predictions
            pred_instances = result['predictions'].pred_instances
            if len(pred_instances.bboxes) > 0:
                pred_bboxes = pred_instances.bboxes.cpu().numpy()
                pred_labels = pred_instances.labels.cpu().numpy()
                pred_scores = pred_instances.scores.cpu().numpy()
                
                print(f"Pred: {len(pred_bboxes)} objects")
                print(f"Pred labels: {pred_labels}")
                print(f"Pred unique labels: {np.unique(pred_labels)}")
                
                # Label overlap analysis
                gt_label_set = set(gt_labels.tolist())
                pred_label_set = set(pred_labels.tolist())
                overlap = gt_label_set.intersection(pred_label_set)
                
                print(f"Label overlap: {len(overlap)} / {len(gt_label_set)} GT classes detected")
                print(f"Missing GT classes: {gt_label_set - pred_label_set}")
                print(f"Extra predicted classes: {pred_label_set - gt_label_set}")
                
                # IoU analysis for matching classes
                matched_count = 0
                high_iou_count = 0
                for gt_bbox, gt_label in zip(gt_bboxes, gt_labels):
                    # Find predictions with same label
                    same_label_mask = pred_labels == gt_label
                    if same_label_mask.any():
                        same_label_bboxes = pred_bboxes[same_label_mask]
                        # Compute IoU with all same-label predictions
                        ious = self.compute_iou(gt_bbox, same_label_bboxes)
                        max_iou = ious.max()
                        if max_iou > 0.5:
                            high_iou_count += 1
                        if max_iou > 0.1:
                            matched_count += 1
                
                print(f"GT objects with IoU>0.5: {high_iou_count} / {len(gt_bboxes)}")
                print(f"GT objects with IoU>0.1: {matched_count} / {len(gt_bboxes)}")
                
                # Class ID mapping check
                print("\nClass ID mapping check:")
                class_names = self.test_dataloader.dataset.metainfo['classes']
                print(f"Dataset has {len(class_names)} classes")
                print(f"GT label range: [{gt_labels.min()}, {gt_labels.max()}]")
                print(f"Pred label range: [{pred_labels.min()}, {pred_labels.max()}]")
                
                # Sample some predictions
                print("\nSample predictions (top 5 by confidence):")
                top_k = min(5, len(pred_scores))
                top_indices = np.argsort(pred_scores)[-top_k:][::-1]
                for i in top_indices:
                    label_id = pred_labels[i]
                    class_name = class_names[label_id] if label_id < len(class_names) else "OUT_OF_RANGE"
                    print(f"  Label {label_id} ({class_name}): score={pred_scores[i]:.3f}, "
                          f"bbox={pred_bboxes[i]}")
                
                comparison = {
                    'idx': idx,
                    'gt_count': len(gt_bboxes),
                    'pred_count': len(pred_bboxes),
                    'label_overlap': len(overlap),
                    'matched_iou_05': high_iou_count,
                    'matched_iou_01': matched_count,
                    'gt_labels': gt_labels,
                    'pred_labels': pred_labels
                }
            else:
                print("⚠️  No predictions to compare!")
                comparison = {
                    'idx': idx,
                    'gt_count': len(gt_bboxes),
                    'pred_count': 0,
                    'label_overlap': 0,
                    'matched_iou_05': 0,
                    'matched_iou_01': 0,
                    'gt_labels': gt_labels,
                    'pred_labels': []
                }
            
            comparison_results.append(comparison)
            
            # Visualize
            self.visualize_comparison(result, comparison)
        
        return comparison_results
    
    def compute_iou(self, bbox1, bboxes2):
        """Compute IoU between one box and multiple boxes."""
        # bbox format: [x1, y1, x2, y2]
        x1_max = np.maximum(bbox1[0], bboxes2[:, 0])
        y1_max = np.maximum(bbox1[1], bboxes2[:, 1])
        x2_min = np.minimum(bbox1[2], bboxes2[:, 2])
        y2_min = np.minimum(bbox1[3], bboxes2[:, 3])
        
        intersection = np.maximum(0, x2_min - x1_max) * np.maximum(0, y2_min - y1_max)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])
        union = area1 + area2 - intersection
        
        return intersection / (union + 1e-6)
    
    def visualize_comparison(self, result, comparison):
        """Visualize predictions vs ground truth."""
        # Read original image
        img_path = result['img_path']
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read image {img_path}")
            return
        
        img_gt = img.copy()
        img_pred = img.copy()
        
        # Draw GT
        gt_instances = result['data_sample']['data_samples'].gt_instances
        gt_bboxes = gt_instances.bboxes.numpy()
        gt_labels = gt_instances.labels.numpy()
        class_names = self.test_dataloader.dataset.metainfo['classes']
        
        for bbox, label in zip(gt_bboxes, gt_labels):
            x1, y1, x2, y2 = bbox.astype(int)
            cv2.rectangle(img_gt, (x1, y1), (x2, y2), (0, 255, 0), 2)
            class_name = class_names[label] if label < len(class_names) else f"cls{label}"
            cv2.putText(img_gt, class_name, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw predictions
        pred_instances = result['predictions'].pred_instances
        if len(pred_instances.bboxes) > 0:
            pred_bboxes = pred_instances.bboxes.cpu().numpy()
            pred_labels = pred_instances.labels.cpu().numpy()
            pred_scores = pred_instances.scores.cpu().numpy()
            
            for bbox, label, score in zip(pred_bboxes, pred_labels, pred_scores):
                if score < 0.3:  # Only show confident predictions
                    continue
                x1, y1, x2, y2 = bbox.astype(int)
                cv2.rectangle(img_pred, (x1, y1), (x2, y2), (0, 0, 255), 2)
                class_name = class_names[label] if label < len(class_names) else f"cls{label}"
                text = f"{class_name}:{score:.2f}"
                cv2.putText(img_pred, text, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Combine images
        combined = np.hstack([img_gt, img_pred])
        
        # Add text labels
        h, w = combined.shape[:2]
        label_img = np.zeros((50, w, 3), dtype=np.uint8)
        cv2.putText(label_img, "Ground Truth", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(label_img, "Predictions", (w//2 + 10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        final = np.vstack([label_img, combined])
        
        # Save
        output_path = self.output_dir / "visualizations" / f"comparison_{result['idx']}.jpg"
        cv2.imwrite(str(output_path), final)
        print(f"Saved visualization to {output_path}")
    
    def phase3_flow_inspection(self):
        """Phase 3: Step-by-step inspection of the entire pipeline."""
        print("\nInspecting data loading and preprocessing pipeline...")
        
        # Get one sample
        data_sample = self.test_dataloader.dataset[0]
        
        print("\n1. DATA LOADING:")
        print(f"   Data sample keys: {data_sample.keys()}")
        print(f"   Inputs type: {type(data_sample['inputs'])}")
        print(f"   Inputs shape: {data_sample['inputs'].shape}")
        print(f"   Data samples type: {type(data_sample['data_samples'])}")
        
        print("\n2. PREPROCESSING:")
        img_meta = data_sample['data_samples']
        print(f"   Original image shape: {img_meta.ori_shape}")
        print(f"   Input image shape: {img_meta.img_shape}")
        print(f"   Scale factor: {img_meta.scale_factor}")
        
        print("\n3. MODEL INPUT:")
        inputs = data_sample['inputs'].unsqueeze(0).to(self.device)
        print(f"   Tensor shape: {inputs.shape}")
        print(f"   Tensor dtype: {inputs.dtype}")
        print(f"   Tensor device: {inputs.device}")
        print(f"   Value range: [{inputs.min():.3f}, {inputs.max():.3f}]")
        print(f"   Mean: {inputs.mean():.3f}, Std: {inputs.std():.3f}")
        
        # Check data preprocessor settings
        print("\n4. DATA PREPROCESSOR SETTINGS:")
        preprocessor = self.detector.data_preprocessor
        print(f"   Mean: {preprocessor.mean}")
        print(f"   Std: {preprocessor.std}")
        print(f"   RGB to BGR: {getattr(preprocessor, 'rgb_to_bgr', 'N/A')}")
        print(f"   Pad size divisor: {getattr(preprocessor, 'pad_size_divisor', 'N/A')}")
        
        print("\n5. MODEL OUTPUT (raw):")
        with torch.no_grad():
            # Get raw model outputs before NMS
            data_samples = [data_sample['data_samples']]
            # Run test_step to get predictions
            predictions = self.detector.test_step([{'inputs': inputs, 'data_samples': data_samples}])
        
        pred_instances = predictions[0].pred_instances
        print(f"   Number of predictions after NMS: {len(pred_instances.bboxes)}")
        if len(pred_instances.bboxes) > 0:
            print(f"   Score range: [{pred_instances.scores.min():.3f}, {pred_instances.scores.max():.3f}]")
            print(f"   Label range: [{pred_instances.labels.min()}, {pred_instances.labels.max()}]")
        
        print("\n6. POST-PROCESSING:")
        # Check NMS settings
        test_cfg = self.detector_cfg.model.get('test_cfg', {})
        print(f"   Test config: {test_cfg}")
        
        print("\n7. EVALUATION PIPELINE:")
        evaluator_cfg = self.detector_cfg.test_evaluator
        print(f"   Evaluator type: {evaluator_cfg.type}")
        print(f"   Metric: {evaluator_cfg.get('metric', 'N/A')}")
        print(f"   Ann file: {evaluator_cfg.get('ann_file', 'N/A')}")
        print(f"   Format only: {evaluator_cfg.get('format_only', False)}")
        
        # Check if dataset is VOC or COCO format
        dataset_type = self.detector_cfg.test_dataloader.dataset.type
        print(f"   Dataset type: {dataset_type}")
        
        return {
            'dataset_type': dataset_type,
            'evaluator_type': evaluator_cfg.type,
            'num_classes': len(self.test_dataloader.dataset.metainfo['classes']),
            'preprocessor_mean': preprocessor.mean,
            'preprocessor_std': preprocessor.std
        }
    
    def phase4_evaluation_check(self):
        """Phase 4: Check ARES evaluation code for issues."""
        print("\nChecking ARES evaluation code implementation...")
        
        # Check evaluator configuration
        evaluator_cfg = self.detector_cfg.test_evaluator
        print(f"\n1. EVALUATOR CONFIGURATION:")
        print(f"   Type: {evaluator_cfg.type}")
        print(f"   Config: {evaluator_cfg}")
        
        # Check dataset metainfo
        print(f"\n2. DATASET METAINFO:")
        metainfo = self.test_dataloader.dataset.metainfo
        print(f"   Classes: {metainfo['classes']}")
        print(f"   Number of classes: {len(metainfo['classes'])}")
        
        # Check if there's a COCO API instance
        print(f"\n3. COCO API CHECK:")
        dataset = self.test_dataloader.dataset
        if hasattr(dataset, 'coco'):
            print(f"   COCO API available: Yes")
            print(f"   Number of images: {len(dataset.coco.imgs)}")
            print(f"   Number of annotations: {len(dataset.coco.anns)}")
            print(f"   Category IDs: {dataset.coco.getCatIds()}")
            
            # Check category mapping
            cats = dataset.coco.loadCats(dataset.coco.getCatIds())
            print(f"   Categories: {[cat['name'] for cat in cats]}")
            
            # Check for class ID mismatch
            dataset_classes = set(metainfo['classes'])
            coco_classes = set([cat['name'] for cat in cats])
            if dataset_classes != coco_classes:
                print(f"   ⚠️  Class mismatch detected!")
                print(f"   Dataset classes not in COCO: {dataset_classes - coco_classes}")
                print(f"   COCO classes not in dataset: {coco_classes - dataset_classes}")
        else:
            print(f"   COCO API available: No")
        
        # Check annotation file
        ann_file = evaluator_cfg.get('ann_file', None)
        if ann_file:
            print(f"\n4. ANNOTATION FILE CHECK:")
            print(f"   Ann file: {ann_file}")
            if os.path.exists(ann_file):
                print(f"   File exists: Yes")
                # Load and check
                with open(ann_file, 'r') as f:
                    ann_data = json.load(f)
                print(f"   Number of images: {len(ann_data.get('images', []))}")
                print(f"   Number of annotations: {len(ann_data.get('annotations', []))}")
                print(f"   Number of categories: {len(ann_data.get('categories', []))}")
                
                # Check category IDs
                categories = ann_data.get('categories', [])
                print(f"   Category IDs: {[cat['id'] for cat in categories]}")
                print(f"   Category names: {[cat['name'] for cat in categories]}")
                
                # Check if 0-based or 1-based
                cat_ids = [cat['id'] for cat in categories]
                if 0 in cat_ids:
                    print(f"   ⚠️  WARNING: 0-based category IDs detected (contains 0)")
                if min(cat_ids) == 1:
                    print(f"   ✓ 1-based category IDs (starts from 1)")
            else:
                print(f"   ⚠️  File does not exist!")
        
        return {
            'evaluator_type': evaluator_cfg.type,
            'ann_file': ann_file,
            'has_coco_api': hasattr(dataset, 'coco')
        }
    
    def generate_report(self, inference_results, annotation_comparison, flow_results, eval_check):
        """Generate comprehensive diagnostic report."""
        report_path = self.output_dir / "diagnostic_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("VOC2007 LOW mAP DIAGNOSTIC REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # Summary statistics
            f.write("SUMMARY STATISTICS:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Number of samples inspected: {len(inference_results)}\n")
            
            total_predictions = sum(r['num_predictions'] for r in inference_results)
            avg_predictions = total_predictions / len(inference_results)
            f.write(f"Average predictions per image: {avg_predictions:.2f}\n")
            
            images_with_no_pred = sum(1 for r in inference_results if r['num_predictions'] == 0)
            f.write(f"Images with no predictions: {images_with_no_pred}\n")
            
            images_with_anomalies = sum(1 for r in inference_results if r['anomalies'])
            f.write(f"Images with anomalies: {images_with_anomalies}\n\n")
            
            # Anomaly details
            if images_with_anomalies > 0:
                f.write("DETECTED ANOMALIES:\n")
                f.write("-" * 80 + "\n")
                all_anomalies = defaultdict(int)
                for r in inference_results:
                    for anomaly in r['anomalies']:
                        all_anomalies[anomaly] += 1
                for anomaly, count in sorted(all_anomalies.items(), key=lambda x: x[1], reverse=True):
                    f.write(f"  - {anomaly}: {count} images\n")
                f.write("\n")
            
            # Annotation comparison
            f.write("PREDICTION vs GROUND TRUTH COMPARISON:\n")
            f.write("-" * 80 + "\n")
            total_gt = sum(c['gt_count'] for c in annotation_comparison)
            total_matched_05 = sum(c['matched_iou_05'] for c in annotation_comparison)
            total_matched_01 = sum(c['matched_iou_01'] for c in annotation_comparison)
            
            f.write(f"Total GT objects: {total_gt}\n")
            f.write(f"Matched with IoU > 0.5: {total_matched_05} ({100*total_matched_05/max(total_gt,1):.1f}%)\n")
            f.write(f"Matched with IoU > 0.1: {total_matched_01} ({100*total_matched_01/max(total_gt,1):.1f}%)\n\n")
            
            # Flow inspection results
            f.write("PIPELINE CONFIGURATION:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Dataset type: {flow_results['dataset_type']}\n")
            f.write(f"Evaluator type: {flow_results['evaluator_type']}\n")
            f.write(f"Number of classes: {flow_results['num_classes']}\n")
            f.write(f"Preprocessor mean: {flow_results['preprocessor_mean']}\n")
            f.write(f"Preprocessor std: {flow_results['preprocessor_std']}\n\n")
            
            # Evaluation check
            f.write("EVALUATION CODE CHECK:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Evaluator type: {eval_check['evaluator_type']}\n")
            f.write(f"Annotation file: {eval_check['ann_file']}\n")
            f.write(f"Has COCO API: {eval_check['has_coco_api']}\n\n")
            
            # Root cause analysis
            f.write("ROOT CAUSE ANALYSIS:\n")
            f.write("-" * 80 + "\n")
            
            # Check for common issues
            issues = []
            
            if images_with_no_pred > len(inference_results) * 0.5:
                issues.append("CRITICAL: More than 50% of images have no predictions")
            
            if total_matched_05 < total_gt * 0.1:
                issues.append("CRITICAL: Less than 10% of GT objects matched (IoU>0.5)")
            
            if "All predictions have same class" in all_anomalies:
                issues.append("CRITICAL: Model predicting same class for all objects")
            
            if flow_results['evaluator_type'] == 'CocoMetric':
                issues.append("INFO: Using COCO metric for VOC dataset - check category ID mapping")
            
            if issues:
                f.write("Detected issues:\n")
                for i, issue in enumerate(issues, 1):
                    f.write(f"{i}. {issue}\n")
            else:
                f.write("No critical issues detected in preliminary analysis.\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")
        
        print(f"\n\nDiagnostic report saved to: {report_path}")
        print(f"Visualizations saved to: {self.output_dir / 'visualizations'}")
        
        # Print summary to console
        print("\n" + "=" * 80)
        print("DIAGNOSTIC SUMMARY")
        print("=" * 80)
        print(f"Samples inspected: {len(inference_results)}")
        print(f"Average predictions per image: {avg_predictions:.2f}")
        print(f"Images with no predictions: {images_with_no_pred}")
        print(f"GT objects matched (IoU>0.5): {total_matched_05}/{total_gt} ({100*total_matched_05/max(total_gt,1):.1f}%)")
        
        if issues:
            print("\n⚠️  DETECTED ISSUES:")
            for issue in issues:
                print(f"   {issue}")
        
        print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description='VOC2007 Detection Diagnostics')
    parser.add_argument('--cfg', required=True, help='Path to detection config file')
    parser.add_argument('--num_samples', type=int, default=10, 
                       help='Number of sample images to inspect (default: 10)')
    args = parser.parse_args()
    
    # Run diagnostics
    diagnostics = VOCDiagnostics(args.cfg, args.num_samples)
    diagnostics.run_full_diagnostics()
    
    print("\n✓ Diagnostics completed successfully!")
    print(f"Check the output directory: {diagnostics.output_dir}")


if __name__ == '__main__':
    main()
