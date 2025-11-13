# VOC2007 Low mAP Diagnostic Tools - Quick Start Guide

## 🚀 5分钟快速开始

### 前提条件
```bash
# 确保已安装ARES 2.0和依赖
pip install -r requirements.txt
mim install mmengine==0.8.4 mmcv==2.0.0 mmdet==3.1.0
```

### 场景：VOC2007评估mAP只有0.47%，应该是70%+

---

## 方案A: 快速诊断（15分钟）

### 步骤1: 检查标注文件 (2分钟)
```bash
python detection/check_voc_conversion.py \
    --json VOCdevkit/VOC2007/annotations/instances_test.json \
    --show-tips
```

**期望输出**:
```
✓ Basic statistics: 4952 images, ~12000 annotations, 20 categories
✓ Category IDs: [1-20] or [0-19] (check this!)
```

**如果看到**:
- `⚠️ WARNING: Category ID 0 present` → 可能有背景类问题
- `✓ 1-based indexing (1-20)` → 正常VOC格式
- `❌ ERROR: Annotations reference undefined category IDs` → 标注文件有问题

---

### 步骤2: 快速推理测试 (3分钟)
```bash
python detection/quick_eval_test.py \
    --cfg detection/configs/detector/faster_rcnn_r50_fpn_1x_voc0712.py \
    --weight work_dirs/faster_rcnn_r50_fpn_1x_voc0712/*.pth \
    --num-images 5
```

**期望输出**:
```
✓ Model loaded on cuda
✓ Dataset has 4952 images
Average predictions per image: 4.2  ← 应该 > 3
Average confidence: 0.615           ← 应该 > 0.3
Images with no predictions: 0/5     ← 应该是 0
```

**如果看到**:
- `Average predictions: 0.3` + `confidence: 0.05` → **预处理问题**
- `Average predictions: 4.2` + `confidence: 0.6` 但 mAP 低 → **类别ID映射问题**

---

### 步骤3: 全面诊断 (10分钟)
```bash
python detection/diagnose_voc2007.py \
    --cfg detection/configs/global/voc2007_faster_rcnn.py \
    --num_samples 20
```

**关键指标**:
```
GT objects matched (IoU>0.5): ??/180
```

- **如果 < 10%**: 类别ID映射错误 ⭐ 最常见
- **如果 > 70%**: 预处理或配置问题
- **如果 = 0%**: 严重问题，检查所有阶段

---

## 方案B: 按问题类型诊断

### 问题1: 类别ID映射错误 (最常见)

**症状**:
- mAP ~0.47%
- 推理正常，置信度正常
- IoU匹配率 < 10%

**快速检查**:
```bash
python -c "
import json
with open('VOCdevkit/VOC2007/annotations/instances_test.json') as f:
    data = json.load(f)
    cat_ids = [cat['id'] for cat in data['categories']]
    print(f'Category IDs: {min(cat_ids)} to {max(cat_ids)}')
    if 0 in cat_ids:
        print('⚠️  WARNING: Contains ID 0 (background?)')
    else:
        print('✓ No ID 0, looks good')
"
```

**修复**: 见 `DIAGNOSTIC_PLAN.md` → Issue 1

---

### 问题2: 预处理错误

**症状**:
- mAP ~0%
- 很少预测或置信度极低
- IoU匹配率低

**快速检查**:
```python
from mmengine.config import Config
cfg = Config.fromfile('detection/configs/detector/faster_rcnn_r50_fpn_1x_voc0712.py')
print(cfg.model.data_preprocessor.mean)  # 应该是 [103.530, 116.280, 123.675]
print(cfg.model.data_preprocessor.std)   # 应该是 [1.0, 1.0, 1.0]
```

**修复**: 见 `DIAGNOSTIC_PLAN.md` → Issue 2

---

### 问题3: 标注文件损坏

**症状**:
- 评估崩溃
- COCO API错误
- "Category not found"

**快速检查**:
```bash
python detection/check_voc_conversion.py \
    --json VOCdevkit/VOC2007/annotations/instances_test.json \
    --voc-root VOCdevkit/VOC2007 \
    --check-samples 10
```

**修复**: 重新生成COCO标注

---

## 方案C: 完整诊断流程（30-60分钟）

按照 `CHECKLIST.md` 中的步骤：

1. ✅ **环境设置** (5分钟)
2. ✅ **数据设置** (5分钟)
3. ✅ **模型设置** (5分钟)
4. 🔍 **标注文件检查** (2分钟) - `check_voc_conversion.py`
5. 🔍 **快速模型测试** (3分钟) - `quick_eval_test.py`
6. 🔍 **全面诊断** (10分钟) - `diagnose_voc2007.py`
7. 🔧 **应用修复** (5-30分钟)
8. ✅ **验证** (5-20分钟)

---

## 常见修复速查表

| 症状 | 可能原因 | 快速修复 | 文档 |
|------|---------|---------|------|
| mAP 0.47%, 预测正常 | 类别ID映射 | 重新生成标注或修改cat2label | Plan→Issue1 |
| mAP 0%, 预测很少 | 预处理错误 | 修改data_preprocessor配置 | Plan→Issue2 |
| 评估崩溃 | 标注损坏 | 重新生成COCO JSON | Plan→Issue3 |
| 预测全被过滤 | test_cfg太严 | 降低score_thr | Plan→Issue4 |
| 类名错误 | metainfo不匹配 | 更新classes定义 | Plan→Issue5 |

---

## 输出文件位置

诊断完成后，查看：
```
diagnose_output/
├── diagnostic_report.txt       ← 阅读这个！
└── visualizations/
    ├── comparison_0.jpg        ← 查看这些图片
    ├── comparison_1.jpg
    └── ...
```

---

## 一键命令（有数据和模型时）

```bash
# 完整诊断流程
cd /home/engine/project

# 1. 检查标注
python detection/check_voc_conversion.py \
    --json VOCdevkit/VOC2007/annotations/instances_test.json

# 2. 快速测试 (如果通过跳到步骤3)
python detection/quick_eval_test.py \
    --cfg detection/configs/detector/faster_rcnn_r50_fpn_1x_voc0712.py \
    --weight work_dirs/faster_rcnn_r50_fpn_1x_voc0712/*.pth

# 3. 全面诊断
python detection/diagnose_voc2007.py \
    --cfg detection/configs/global/voc2007_faster_rcnn.py \
    --num_samples 20

# 4. 查看报告
cat diagnose_output/diagnostic_report.txt
```

---

## 预期时间线

- **最快**: 15分钟（如果问题明显）
- **正常**: 30-45分钟（完整诊断+修复）
- **复杂**: 60-90分钟（需要深入调试）

---

## 获取帮助

1. 查看诊断报告的 "ROOT CAUSE ANALYSIS" 部分
2. 阅读 `DIAGNOSTIC_PLAN.md` 对应的Issue章节
3. 参考 `CHECKLIST.md` 逐步排查
4. 查看 `README_VOC_DIAGNOSTICS.md` 完整文档

---

## 成功标志

修复后运行：
```bash
python detection/run.py \
    --cfg detection/configs/global/voc2007_faster_rcnn.py \
    --eval_only
```

**期望结果**:
```
bbox_mAP: 0.697
bbox_mAP_50: 0.732  ← 目标达成！
```

如果 mAP > 70%，恭喜你成功了！🎉

---

## 快速测试（无需数据）

验证工具本身是否正常：
```bash
# 测试导入
python -c "from mmengine.config import Config; print('✓ MMEngine OK')"

# 测试工具函数
python detection/diagnose_utils.py

# 测试配置加载
python -c "from mmengine.config import Config; \
    cfg = Config.fromfile('detection/configs/detector/faster_rcnn_r50_fpn_1x_voc0712.py'); \
    print('✓ Config OK')"
```

全部通过说明工具安装正确！

---

**准备好了吗？开始诊断吧！** 🚀
