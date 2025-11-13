"""
Quick evaluation test script for VOC2007 detection.
This script provides quick sanity checks and evaluations.
"""

import os
import sys
import argparse
import json
from pathlib import Path

import torch
import numpy as np
from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.registry import MODELS, DefaultScope
from mmengine.evaluator.evaluator import Evaluator

from ares.attack.detection.utils import HiddenPrints


def check_dataset_annotation(ann_file):
    """Check COCO annotation file structure and content."""
    print("=" * 80)
    print("CHECKING ANNOTATION FILE")
    print("=" * 80)
    
    if not os.path.exists(ann_file):
        print(f"❌ ERROR: Annotation file not found: {ann_file}")
        return False
    
    print(f"✓ Annotation file exists: {ann_file}")
    
    try:
        with open(ann_file, 'r') as f:
            data = json.load(f)
        
        print(f"\nAnnotation file structure:")
        print(f"  - Images: {len(data.get('images', []))}")
        print(f"  - Annotations: {len(data.get('annotations', []))}")
        print(f"  - Categories: {len(data.get('categories', []))}")
        
        # Check categories
        categories = data.get('categories', [])
        if not categories:
            print("❌ ERROR: No categories found in annotation file")
            return False
        
        print(f"\nCategories:")
        cat_ids = []
        for cat in categories:
            cat_ids.append(cat['id'])
            print(f"  [{cat['id']}] {cat['name']}")
        
        # Check category ID range
        min_id = min(cat_ids)
        max_id = max(cat_ids)
        print(f"\nCategory ID range: [{min_id}, {max_id}]")
        
        if 0 in cat_ids:
            print("⚠️  WARNING: Category ID 0 detected (usually background/reserved)")
        if min_id == 1:
            print("✓ Categories start from 1 (1-based indexing)")
        if min_id == 0 and max_id == len(categories) - 1:
            print("✓ Categories use 0-based indexing")
        
        # Check for gaps in category IDs
        expected_ids = set(range(min_id, max_id + 1))
        actual_ids = set(cat_ids)
        if expected_ids != actual_ids:
            missing = expected_ids - actual_ids
            print(f"⚠️  WARNING: Gaps in category IDs: {missing}")
        
        # Sample some annotations
        annotations = data.get('annotations', [])
        if annotations:
            print(f"\nSample annotation (first one):")
            ann = annotations[0]
            print(f"  Image ID: {ann.get('image_id')}")
            print(f"  Category ID: {ann.get('category_id')}")
            print(f"  BBox: {ann.get('bbox')}")
            print(f"  Area: {ann.get('area')}")
            
            # Check annotation category IDs
            ann_cat_ids = set(a['category_id'] for a in annotations)
            if not ann_cat_ids.issubset(actual_ids):
                extra = ann_cat_ids - actual_ids
                print(f"❌ ERROR: Annotations reference undefined category IDs: {extra}")
                return False
        
        return True
        
    except json.JSONDecodeError as e:
        print(f"❌ ERROR: Invalid JSON format: {e}")
        return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False


def check_model_config(cfg_file, weight_file):
    """Check model configuration and weights."""
    print("\n" + "=" * 80)
    print("CHECKING MODEL CONFIGURATION")
    print("=" * 80)
    
    if not os.path.exists(cfg_file):
        print(f"❌ ERROR: Config file not found: {cfg_file}")
        return False
    
    print(f"✓ Config file exists: {cfg_file}")
    
    if not os.path.exists(weight_file):
        print(f"❌ ERROR: Weight file not found: {weight_file}")
        return False
    
    print(f"✓ Weight file exists: {weight_file}")
    
    try:
        cfg = Config.fromfile(cfg_file)
        
        # Check model configuration
        model_cfg = cfg.model
        print(f"\nModel configuration:")
        print(f"  - Type: {model_cfg.get('type', 'N/A')}")
        print(f"  - Backbone: {model_cfg.get('backbone', {}).get('type', 'N/A')}")
        
        # Check data preprocessor
        preprocessor = model_cfg.get('data_preprocessor', {})
        print(f"\nData preprocessor:")
        print(f"  - Mean: {preprocessor.get('mean', 'N/A')}")
        print(f"  - Std: {preprocessor.get('std', 'N/A')}")
        print(f"  - BGR to RGB: {preprocessor.get('bgr_to_rgb', 'N/A')}")
        
        # Check number of classes
        roi_head = model_cfg.get('roi_head', {})
        bbox_head = roi_head.get('bbox_head', {})
        num_classes = bbox_head.get('num_classes', 'N/A')
        print(f"\n  - Number of classes in model: {num_classes}")
        
        # Check test config
        test_cfg = model_cfg.get('test_cfg', {})
        if test_cfg:
            rcnn_cfg = test_cfg.get('rcnn', {})
            print(f"\nTest configuration:")
            print(f"  - Score threshold: {rcnn_cfg.get('score_thr', 'N/A')}")
            print(f"  - NMS threshold: {rcnn_cfg.get('nms', {}).get('iou_threshold', 'N/A')}")
            print(f"  - Max per image: {rcnn_cfg.get('max_per_img', 'N/A')}")
        
        # Try to load weights
        print(f"\nAttempting to load weights...")
        checkpoint = torch.load(weight_file, map_location='cpu')
        
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print(f"✓ Checkpoint contains state_dict with {len(state_dict)} keys")
            
            # Check for classifier weights
            for key in state_dict.keys():
                if 'fc_cls' in key or 'cls' in key:
                    shape = state_dict[key].shape
                    print(f"  Classifier layer '{key}': shape {shape}")
                    if len(shape) == 2:
                        print(f"    → Output classes: {shape[0]}")
                    break
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def quick_inference_test(cfg_file, weight_file, num_images=5):
    """Run quick inference test on a few images."""
    print("\n" + "=" * 80)
    print("RUNNING QUICK INFERENCE TEST")
    print("=" * 80)
    
    try:
        # Load config
        cfg = Config.fromfile(cfg_file)
        
        # Initialize model
        print("\nInitializing model...")
        with HiddenPrints():
            DefaultScope.get_instance('test', scope_name='mmdet')
            model = MODELS.build(cfg.model)
            model.eval()
        
        # Load weights
        print(f"Loading weights from {weight_file}...")
        checkpoint = torch.load(weight_file, map_location='cpu')
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        print(f"✓ Model loaded on {device}")
        
        # Build dataloader
        print("\nBuilding dataloader...")
        cfg.test_dataloader.batch_size = 1
        with HiddenPrints():
            dataloader = Runner.build_dataloader(cfg.test_dataloader, seed=0, diff_rank_seed=False)
        
        print(f"✓ Dataset has {len(dataloader.dataset)} images")
        print(f"✓ Classes: {dataloader.dataset.metainfo['classes']}")
        
        # Test on a few images
        print(f"\nTesting inference on {num_images} images...")
        results = []
        
        for i, data_batch in enumerate(dataloader):
            if i >= num_images:
                break
            
            inputs = data_batch['inputs'][0].to(device)
            data_samples = data_batch['data_samples']
            
            with torch.no_grad():
                predictions = model.test_step([{'inputs': inputs.unsqueeze(0), 'data_samples': data_samples}])
            
            pred_instances = predictions[0].pred_instances
            gt_instances = data_samples[0].gt_instances
            
            img_path = data_samples[0].img_path
            num_pred = len(pred_instances.bboxes)
            num_gt = len(gt_instances.bboxes)
            
            result = {
                'image': os.path.basename(img_path),
                'num_predictions': num_pred,
                'num_gt': num_gt
            }
            
            if num_pred > 0:
                scores = pred_instances.scores.cpu().numpy()
                labels = pred_instances.labels.cpu().numpy()
                result['avg_score'] = float(scores.mean())
                result['max_score'] = float(scores.max())
                result['unique_labels'] = len(np.unique(labels))
            else:
                result['avg_score'] = 0.0
                result['max_score'] = 0.0
                result['unique_labels'] = 0
            
            results.append(result)
            
            print(f"\n  Image {i+1}: {result['image']}")
            print(f"    GT objects: {num_gt}")
            print(f"    Predictions: {num_pred}")
            if num_pred > 0:
                print(f"    Avg score: {result['avg_score']:.3f}")
                print(f"    Max score: {result['max_score']:.3f}")
                print(f"    Unique labels: {result['unique_labels']}")
        
        # Summary
        print(f"\n" + "-" * 80)
        print("SUMMARY:")
        avg_pred = np.mean([r['num_predictions'] for r in results])
        avg_score = np.mean([r['avg_score'] for r in results if r['avg_score'] > 0])
        images_no_pred = sum(1 for r in results if r['num_predictions'] == 0)
        
        print(f"  Average predictions per image: {avg_pred:.2f}")
        print(f"  Average confidence score: {avg_score:.3f}")
        print(f"  Images with no predictions: {images_no_pred}/{len(results)}")
        
        if avg_pred < 1:
            print("\n⚠️  WARNING: Very few predictions on average!")
        if images_no_pred > len(results) * 0.5:
            print("⚠️  WARNING: More than half of images have no predictions!")
        if avg_score < 0.3:
            print("⚠️  WARNING: Average confidence score is very low!")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR during inference test: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='Quick VOC2007 Evaluation Test')
    parser.add_argument('--cfg', required=True, help='Path to detector config file')
    parser.add_argument('--weight', required=True, help='Path to model weights')
    parser.add_argument('--ann-file', help='Path to annotation file (optional, will try to infer from config)')
    parser.add_argument('--num-images', type=int, default=5, help='Number of images for quick test')
    args = parser.parse_args()
    
    print("VOC2007 QUICK EVALUATION TEST")
    print("=" * 80)
    
    # Step 1: Check model config
    success = check_model_config(args.cfg, args.weight)
    if not success:
        print("\n❌ Model configuration check failed!")
        return
    
    # Step 2: Check annotation file
    if args.ann_file:
        ann_file = args.ann_file
    else:
        # Try to infer from config
        try:
            cfg = Config.fromfile(args.cfg)
            ann_file = cfg.test_dataloader.dataset.get('ann_file')
            if ann_file and not os.path.isabs(ann_file):
                data_root = cfg.test_dataloader.dataset.get('data_root', '')
                ann_file = os.path.join(data_root, ann_file)
        except:
            ann_file = None
    
    if ann_file:
        success = check_dataset_annotation(ann_file)
        if not success:
            print("\n⚠️  Annotation file check failed, but continuing...")
    else:
        print("\n⚠️  No annotation file specified, skipping check")
    
    # Step 3: Run quick inference test
    success = quick_inference_test(args.cfg, args.weight, args.num_images)
    
    if success:
        print("\n" + "=" * 80)
        print("✓ QUICK TEST COMPLETED")
        print("=" * 80)
        print("\nIf tests passed but mAP is still low, run full diagnostics:")
        print(f"  python detection/diagnose_voc2007.py --cfg <attack_config> --num_samples 20")
    else:
        print("\n❌ QUICK TEST FAILED - Check errors above")


if __name__ == '__main__':
    main()
