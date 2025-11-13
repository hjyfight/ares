"""
Script to check and verify VOC to COCO format conversion.
Helps diagnose class ID mapping issues which are common causes of low mAP.
"""

import os
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict
import argparse


VOC_CLASSES = (
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
    'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
)


def parse_voc_xml(xml_file):
    """Parse VOC XML annotation file."""
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    objects = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        difficult = int(obj.find('difficult').text) if obj.find('difficult') is not None else 0
        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)
        
        objects.append({
            'name': name,
            'difficult': difficult,
            'bbox': [xmin, ymin, xmax, ymax]
        })
    
    return objects


def check_coco_json(json_file, voc_root=None, check_samples=5):
    """Check COCO JSON annotation file."""
    print("=" * 80)
    print(f"CHECKING COCO JSON: {json_file}")
    print("=" * 80)
    
    if not os.path.exists(json_file):
        print(f"❌ ERROR: File not found: {json_file}")
        return False
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    images = data.get('images', [])
    annotations = data.get('annotations', [])
    categories = data.get('categories', [])
    
    print(f"\nBasic statistics:")
    print(f"  - Images: {len(images)}")
    print(f"  - Annotations: {len(annotations)}")
    print(f"  - Categories: {len(categories)}")
    
    # Check categories
    print(f"\nCategories in COCO JSON:")
    category_map = {}
    for cat in categories:
        category_map[cat['id']] = cat['name']
        print(f"  [{cat['id']}] {cat['name']}")
    
    # Check if categories match VOC
    coco_classes = set(category_map.values())
    voc_classes = set(VOC_CLASSES)
    
    if coco_classes == voc_classes:
        print(f"\n✓ All VOC classes present in COCO JSON")
    else:
        missing = voc_classes - coco_classes
        extra = coco_classes - voc_classes
        if missing:
            print(f"\n⚠️  Missing VOC classes: {missing}")
        if extra:
            print(f"\n⚠️  Extra classes not in VOC: {extra}")
    
    # Check category ID mapping
    print(f"\nCategory ID analysis:")
    cat_ids = sorted(category_map.keys())
    print(f"  - ID range: [{min(cat_ids)}, {max(cat_ids)}]")
    
    if 0 in cat_ids:
        print(f"  - ⚠️  WARNING: Category ID 0 present (unusual for VOC)")
        print(f"      Class at ID 0: {category_map[0]}")
    
    if min(cat_ids) == 1 and max(cat_ids) == 20:
        print(f"  - ✓ 1-based indexing (1-20) - Standard VOC format")
    elif min(cat_ids) == 0 and max(cat_ids) == 19:
        print(f"  - ✓ 0-based indexing (0-19) - COCO-style format")
    else:
        print(f"  - ⚠️  WARNING: Unusual category ID range!")
    
    # Check if IDs are contiguous
    expected_ids = set(range(min(cat_ids), max(cat_ids) + 1))
    actual_ids = set(cat_ids)
    if expected_ids != actual_ids:
        missing_ids = expected_ids - actual_ids
        print(f"  - ⚠️  WARNING: Non-contiguous category IDs, missing: {missing_ids}")
    
    # Check annotations
    print(f"\nAnnotation analysis:")
    ann_cat_ids = set(ann['category_id'] for ann in annotations)
    print(f"  - Unique category IDs in annotations: {len(ann_cat_ids)}")
    print(f"  - Category IDs used: {sorted(ann_cat_ids)}")
    
    # Check for undefined categories
    undefined = ann_cat_ids - actual_ids
    if undefined:
        print(f"  - ❌ ERROR: Annotations reference undefined category IDs: {undefined}")
        return False
    
    # Category distribution
    cat_count = defaultdict(int)
    for ann in annotations:
        cat_count[ann['category_id']] += 1
    
    print(f"\nAnnotation distribution:")
    for cat_id in sorted(cat_count.keys()):
        cat_name = category_map[cat_id]
        count = cat_count[cat_id]
        print(f"  [{cat_id}] {cat_name:15s}: {count:5d} instances")
    
    # Sample comparison with VOC XML if available
    if voc_root and check_samples > 0:
        print(f"\n" + "=" * 80)
        print(f"CROSS-CHECKING WITH VOC XML (sampling {check_samples} images)")
        print("=" * 80)
        
        # Create image ID to filename mapping
        img_map = {img['id']: img['file_name'] for img in images}
        
        # Group annotations by image
        img_anns = defaultdict(list)
        for ann in annotations:
            img_anns[ann['image_id']].append(ann)
        
        # Sample some images
        sample_img_ids = sorted(img_map.keys())[:check_samples]
        
        issues_found = 0
        for img_id in sample_img_ids:
            filename = img_map[img_id]
            base_name = os.path.splitext(filename)[0]
            
            # Find corresponding XML file
            xml_file = os.path.join(voc_root, 'Annotations', base_name + '.xml')
            if not os.path.exists(xml_file):
                print(f"\n⚠️  XML not found for {filename}: {xml_file}")
                continue
            
            print(f"\nComparing image: {filename}")
            
            # Parse XML
            voc_objects = parse_voc_xml(xml_file)
            coco_anns = img_anns[img_id]
            
            print(f"  VOC XML: {len(voc_objects)} objects")
            print(f"  COCO JSON: {len(coco_anns)} annotations")
            
            if len(voc_objects) != len(coco_anns):
                print(f"  ⚠️  WARNING: Object count mismatch!")
                issues_found += 1
            
            # Check class mapping
            voc_classes_in_img = set(obj['name'] for obj in voc_objects)
            coco_classes_in_img = set(category_map[ann['category_id']] for ann in coco_anns)
            
            print(f"  VOC classes: {sorted(voc_classes_in_img)}")
            print(f"  COCO classes: {sorted(coco_classes_in_img)}")
            
            if voc_classes_in_img != coco_classes_in_img:
                print(f"  ⚠️  WARNING: Class mismatch!")
                print(f"      In VOC but not COCO: {voc_classes_in_img - coco_classes_in_img}")
                print(f"      In COCO but not VOC: {coco_classes_in_img - voc_classes_in_img}")
                issues_found += 1
            else:
                print(f"  ✓ Classes match")
            
            # Check a bbox sample
            if voc_objects and coco_anns:
                voc_bbox = voc_objects[0]['bbox']  # [xmin, ymin, xmax, ymax]
                coco_bbox = coco_anns[0]['bbox']  # [x, y, w, h]
                
                # Convert COCO to VOC format
                coco_as_voc = [
                    coco_bbox[0],
                    coco_bbox[1],
                    coco_bbox[0] + coco_bbox[2],
                    coco_bbox[1] + coco_bbox[3]
                ]
                
                bbox_match = all(abs(a - b) < 1.0 for a, b in zip(voc_bbox, coco_as_voc))
                
                print(f"  Sample bbox check:")
                print(f"    VOC:  {[f'{x:.1f}' for x in voc_bbox]}")
                print(f"    COCO: {[f'{x:.1f}' for x in coco_as_voc]}")
                
                if bbox_match:
                    print(f"    ✓ Bboxes match")
                else:
                    print(f"    ⚠️  WARNING: Bbox mismatch!")
                    issues_found += 1
        
        if issues_found == 0:
            print(f"\n✓ All sampled images verified successfully!")
        else:
            print(f"\n⚠️  Found {issues_found} potential issues in sampled images")
    
    return True


def generate_conversion_tips():
    """Generate tips for proper VOC to COCO conversion."""
    print("\n" + "=" * 80)
    print("VOC TO COCO CONVERSION TIPS")
    print("=" * 80)
    
    tips = """
1. CATEGORY ID MAPPING:
   - VOC has 20 classes (no background class in annotations)
   - COCO format can use either 0-based (0-19) or 1-based (1-20) indexing
   - MUST be consistent between annotation JSON and model configuration
   - Recommended: Use 1-based indexing to match mmdetection defaults

2. PROPER CATEGORY FORMAT:
   {
     "id": 1,              # 1-20 for VOC (or 0-19 if 0-based)
     "name": "aeroplane",  # Exact VOC class name
     "supercategory": "none"
   }

3. BBOX FORMAT:
   - VOC XML: [xmin, ymin, xmax, ymax]
   - COCO JSON: [x, y, width, height]
   - Conversion: [xmin, ymin, xmax-xmin, ymax-ymin]

4. COMMON ISSUES:
   a) Category ID starts from 0 but model expects 1-based
      → Solution: Ensure model num_classes matches, check cat2label mapping
   
   b) Background class (ID=0) included in categories
      → Solution: VOC doesn't have background in 20 classes, remove it
   
   c) Class names don't match exactly
      → Solution: Use exact VOC class names (e.g., 'aeroplane' not 'airplane')
   
   d) Missing 'difficult' flag handling
      → Solution: Decide whether to include difficult objects in eval

5. VERIFICATION STEPS:
   - Check category IDs are contiguous
   - Verify all annotation category_ids exist in categories
   - Compare random samples with original XML
   - Ensure bbox coordinates are valid (x,y,w,h all > 0)

6. RECOMMENDED CONVERSION COMMAND (using mmdetection):
   python tools/dataset_converters/pascal_voc.py \\
       VOCdevkit/ \\
       --out-format coco \\
       --classes 'aeroplane' 'bicycle' ... 'tvmonitor'
"""
    print(tips)


def main():
    parser = argparse.ArgumentParser(description='Check VOC to COCO conversion')
    parser.add_argument('--json', required=True, help='Path to COCO JSON annotation file')
    parser.add_argument('--voc-root', help='Path to VOC root directory (e.g., VOCdevkit/VOC2007)')
    parser.add_argument('--check-samples', type=int, default=5,
                       help='Number of samples to cross-check with XML (default: 5)')
    parser.add_argument('--show-tips', action='store_true',
                       help='Show conversion tips and best practices')
    args = parser.parse_args()
    
    # Check COCO JSON
    success = check_coco_json(args.json, args.voc_root, args.check_samples)
    
    # Show tips if requested
    if args.show_tips:
        generate_conversion_tips()
    
    if success:
        print("\n" + "=" * 80)
        print("✓ COCO JSON CHECK COMPLETED")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("❌ COCO JSON CHECK FAILED - See errors above")
        print("=" * 80)
        print("\nRun with --show-tips for conversion guidance")


if __name__ == '__main__':
    main()
