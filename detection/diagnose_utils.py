"""
Diagnostic utilities for VOC2007 low mAP debugging.
Common helper functions used across diagnostic scripts.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple


def compute_iou_matrix(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    Compute IoU matrix between two sets of boxes.
    
    Args:
        boxes1: (N, 4) array in [x1, y1, x2, y2] format
        boxes2: (M, 4) array in [x1, y1, x2, y2] format
    
    Returns:
        (N, M) IoU matrix
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    # Expand dimensions for broadcasting
    boxes1_exp = boxes1[:, None, :]  # (N, 1, 4)
    boxes2_exp = boxes2[None, :, :]  # (1, M, 4)
    
    # Compute intersection
    x1_max = np.maximum(boxes1_exp[:, :, 0], boxes2_exp[:, :, 0])
    y1_max = np.maximum(boxes1_exp[:, :, 1], boxes2_exp[:, :, 1])
    x2_min = np.minimum(boxes1_exp[:, :, 2], boxes2_exp[:, :, 2])
    y2_min = np.minimum(boxes1_exp[:, :, 3], boxes2_exp[:, :, 3])
    
    intersection = np.maximum(0, x2_min - x1_max) * np.maximum(0, y2_min - y1_max)
    
    # Compute union
    union = area1[:, None] + area2[None, :] - intersection
    
    # Compute IoU
    iou = intersection / (union + 1e-10)
    
    return iou


def match_predictions_to_gt(
    pred_boxes: np.ndarray,
    pred_labels: np.ndarray,
    pred_scores: np.ndarray,
    gt_boxes: np.ndarray,
    gt_labels: np.ndarray,
    iou_threshold: float = 0.5
) -> Dict:
    """
    Match predictions to ground truth boxes.
    
    Returns:
        Dictionary with matching statistics
    """
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return {
            'num_matched': 0,
            'num_pred': len(pred_boxes),
            'num_gt': len(gt_boxes),
            'precision': 0.0,
            'recall': 0.0
        }
    
    # Compute IoU matrix
    iou_matrix = compute_iou_matrix(pred_boxes, gt_boxes)
    
    # Match predictions to GT
    num_matched = 0
    matched_gt = set()
    
    # Sort predictions by confidence
    sorted_indices = np.argsort(pred_scores)[::-1]
    
    for pred_idx in sorted_indices:
        pred_label = pred_labels[pred_idx]
        
        # Find best matching GT box with same label
        best_iou = 0
        best_gt_idx = -1
        
        for gt_idx in range(len(gt_boxes)):
            if gt_idx in matched_gt:
                continue
            
            if gt_labels[gt_idx] != pred_label:
                continue
            
            iou = iou_matrix[pred_idx, gt_idx]
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            num_matched += 1
            matched_gt.add(best_gt_idx)
    
    precision = num_matched / len(pred_boxes) if len(pred_boxes) > 0 else 0.0
    recall = num_matched / len(gt_boxes) if len(gt_boxes) > 0 else 0.0
    
    return {
        'num_matched': num_matched,
        'num_pred': len(pred_boxes),
        'num_gt': len(gt_boxes),
        'precision': precision,
        'recall': recall,
        'iou_matrix': iou_matrix
    }


def analyze_class_distribution(labels: np.ndarray, class_names: List[str]) -> Dict:
    """
    Analyze class distribution in labels.
    
    Returns:
        Dictionary with class statistics
    """
    unique, counts = np.unique(labels, return_counts=True)
    
    distribution = {}
    for label_id, count in zip(unique, counts):
        if label_id < len(class_names):
            class_name = class_names[label_id]
        else:
            class_name = f"INVALID_CLASS_{label_id}"
        
        distribution[class_name] = {
            'id': int(label_id),
            'count': int(count),
            'percentage': float(count) / len(labels) * 100
        }
    
    return distribution


def check_bbox_validity(boxes: np.ndarray, img_width: int, img_height: int) -> Dict:
    """
    Check bounding box validity.
    
    Returns:
        Dictionary with validity statistics
    """
    if len(boxes) == 0:
        return {
            'num_total': 0,
            'num_valid': 0,
            'num_invalid_format': 0,
            'num_out_of_bounds': 0,
            'num_zero_area': 0
        }
    
    num_total = len(boxes)
    num_invalid_format = 0
    num_out_of_bounds = 0
    num_zero_area = 0
    
    for box in boxes:
        x1, y1, x2, y2 = box
        
        # Check format (x2 > x1, y2 > y1)
        if x2 <= x1 or y2 <= y1:
            num_invalid_format += 1
        
        # Check bounds
        if x1 < 0 or y1 < 0 or x2 > img_width or y2 > img_height:
            num_out_of_bounds += 1
        
        # Check area
        if (x2 - x1) * (y2 - y1) <= 0:
            num_zero_area += 1
    
    num_valid = num_total - num_invalid_format
    
    return {
        'num_total': num_total,
        'num_valid': num_valid,
        'num_invalid_format': num_invalid_format,
        'num_out_of_bounds': num_out_of_bounds,
        'num_zero_area': num_zero_area,
        'validity_rate': num_valid / num_total if num_total > 0 else 0.0
    }


def analyze_score_distribution(scores: np.ndarray) -> Dict:
    """
    Analyze prediction score distribution.
    
    Returns:
        Dictionary with score statistics
    """
    if len(scores) == 0:
        return {
            'count': 0,
            'min': 0.0,
            'max': 0.0,
            'mean': 0.0,
            'median': 0.0,
            'std': 0.0,
            'percentiles': {}
        }
    
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    percentile_values = np.percentile(scores, percentiles)
    
    return {
        'count': len(scores),
        'min': float(scores.min()),
        'max': float(scores.max()),
        'mean': float(scores.mean()),
        'median': float(np.median(scores)),
        'std': float(scores.std()),
        'percentiles': {p: float(v) for p, v in zip(percentiles, percentile_values)},
        'num_below_0.1': int((scores < 0.1).sum()),
        'num_below_0.3': int((scores < 0.3).sum()),
        'num_below_0.5': int((scores < 0.5).sum()),
        'num_above_0.5': int((scores >= 0.5).sum()),
        'num_above_0.7': int((scores >= 0.7).sum()),
        'num_above_0.9': int((scores >= 0.9).sum())
    }


def check_class_id_mapping(
    pred_labels: np.ndarray,
    gt_labels: np.ndarray,
    num_classes: int
) -> Dict:
    """
    Check for class ID mapping issues.
    
    Returns:
        Dictionary with mapping analysis
    """
    issues = []
    
    # Check if labels are in valid range
    if len(pred_labels) > 0:
        pred_min, pred_max = pred_labels.min(), pred_labels.max()
        if pred_min < 0:
            issues.append(f"Negative prediction labels detected: min={pred_min}")
        if pred_max >= num_classes:
            issues.append(f"Prediction labels exceed num_classes: max={pred_max}, num_classes={num_classes}")
    else:
        pred_min, pred_max = None, None
    
    if len(gt_labels) > 0:
        gt_min, gt_max = gt_labels.min(), gt_labels.max()
        if gt_min < 0:
            issues.append(f"Negative GT labels detected: min={gt_min}")
        if gt_max >= num_classes:
            issues.append(f"GT labels exceed num_classes: max={gt_max}, num_classes={num_classes}")
    else:
        gt_min, gt_max = None, None
    
    # Check for 0-based vs 1-based confusion
    if gt_min is not None and pred_min is not None:
        if gt_min == 1 and pred_min == 0:
            issues.append("Possible 0-based/1-based mismatch: GT starts from 1, predictions from 0")
        elif gt_min == 0 and pred_min == 1:
            issues.append("Possible 0-based/1-based mismatch: GT starts from 0, predictions from 1")
    
    return {
        'pred_label_range': (pred_min, pred_max) if pred_min is not None else None,
        'gt_label_range': (gt_min, gt_max) if gt_min is not None else None,
        'num_classes': num_classes,
        'issues': issues
    }


def print_comparison_table(pred_stats: Dict, gt_stats: Dict, class_names: List[str]):
    """Print a comparison table of prediction vs GT statistics."""
    print("\nPrediction vs Ground Truth Comparison:")
    print("-" * 80)
    print(f"{'Class':<20} {'GT Count':>10} {'Pred Count':>10} {'Ratio':>10}")
    print("-" * 80)
    
    all_classes = set(pred_stats.keys()) | set(gt_stats.keys())
    
    for class_name in sorted(all_classes):
        gt_count = gt_stats.get(class_name, {'count': 0})['count']
        pred_count = pred_stats.get(class_name, {'count': 0})['count']
        ratio = pred_count / gt_count if gt_count > 0 else 0.0
        
        print(f"{class_name:<20} {gt_count:>10} {pred_count:>10} {ratio:>10.2f}")
    
    print("-" * 80)


def summarize_diagnostics(results: List[Dict], class_names: List[str]):
    """
    Summarize diagnostic results across multiple images.
    
    Args:
        results: List of per-image diagnostic results
        class_names: List of class names
    """
    print("\n" + "=" * 80)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 80)
    
    total_images = len(results)
    total_gt = sum(r['num_gt'] for r in results)
    total_pred = sum(r['num_pred'] for r in results)
    total_matched = sum(r['num_matched'] for r in results)
    
    images_no_pred = sum(1 for r in results if r['num_pred'] == 0)
    images_no_gt = sum(1 for r in results if r['num_gt'] == 0)
    
    print(f"\nOverall Statistics:")
    print(f"  Total images: {total_images}")
    print(f"  Total GT objects: {total_gt}")
    print(f"  Total predictions: {total_pred}")
    print(f"  Total matched (IoU>0.5): {total_matched}")
    print(f"  Images with no predictions: {images_no_pred}")
    print(f"  Images with no GT: {images_no_gt}")
    
    if total_pred > 0:
        overall_precision = total_matched / total_pred
        print(f"  Overall precision: {overall_precision:.3f}")
    
    if total_gt > 0:
        overall_recall = total_matched / total_gt
        print(f"  Overall recall: {overall_recall:.3f}")
    
    if total_pred > 0 and total_gt > 0:
        f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall + 1e-10)
        print(f"  Overall F1: {f1:.3f}")
    
    # Identify critical issues
    print(f"\nCritical Issue Detection:")
    
    if images_no_pred > total_images * 0.5:
        print(f"  ❌ CRITICAL: {images_no_pred}/{total_images} images have no predictions!")
    
    if total_matched < total_gt * 0.1:
        print(f"  ❌ CRITICAL: Only {total_matched}/{total_gt} GT objects matched (<10%)")
    
    if total_pred == 0:
        print(f"  ❌ CRITICAL: No predictions at all!")
    
    # Check if all predictions are same class
    if 'pred_labels_all' in results[0]:
        all_pred_labels = []
        for r in results:
            if 'pred_labels_all' in r:
                all_pred_labels.extend(r['pred_labels_all'])
        
        if len(all_pred_labels) > 0:
            unique_labels = len(np.unique(all_pred_labels))
            if unique_labels == 1:
                print(f"  ❌ CRITICAL: All predictions are class {all_pred_labels[0]}!")
            elif unique_labels < len(class_names) * 0.5:
                print(f"  ⚠️  WARNING: Only {unique_labels}/{len(class_names)} classes predicted")


if __name__ == '__main__':
    # Test functions
    print("Diagnostic utilities loaded successfully.")
    print("Available functions:")
    print("  - compute_iou_matrix")
    print("  - match_predictions_to_gt")
    print("  - analyze_class_distribution")
    print("  - check_bbox_validity")
    print("  - analyze_score_distribution")
    print("  - check_class_id_mapping")
    print("  - print_comparison_table")
    print("  - summarize_diagnostics")
