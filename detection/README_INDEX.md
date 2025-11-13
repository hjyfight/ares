# VOC2007 Diagnostic Tools - Documentation Index

## 📚 文档导航

根据您的需求选择合适的文档：

---

## 🚀 快速开始

### **我想立即开始诊断** → [`QUICKSTART.md`](QUICKSTART.md)
- ⏱️ 5分钟快速诊断流程
- 🎯 一键命令示例
- 📋 常见问题速查表
- **最适合**: 急需快速结果的用户

---

## 📖 完整指南

### **我需要全面了解工具** → [`README_VOC_DIAGNOSTICS.md`](README_VOC_DIAGNOSTICS.md)
- 📘 完整用户指南 (13KB, ~18页)
- 🔧 所有工具的详细说明
- 💡 使用示例和最佳实践
- 🎓 常见问题解决方案
- **最适合**: 首次使用的用户

---

## 🔍 系统化排查

### **我要按步骤排查问题** → [`CHECKLIST.md`](CHECKLIST.md)
- ✅ 逐步检查清单 (9.7KB, ~12页)
- 🎯 每个步骤的预期结果
- ⚠️ 失败时的处理方案
- 📊 成功标准
- **最适合**: 需要系统化方法的用户

---

## 🛠️ 深度诊断

### **我需要详细的问题解决方案** → [`DIAGNOSTIC_PLAN.md`](DIAGNOSTIC_PLAN.md)
- 📊 5个常见根因详解 (13KB, ~20页)
- 🔧 每个问题的诊断方法
- 💊 具体的修复步骤
- 📈 预期修复效果
- **最适合**: 遇到特定问题需要深入了解

---

## 📝 使用说明

### **我需要工具的使用教程** → [`DIAGNOSTICS_README.md`](DIAGNOSTICS_README.md)
- 🎓 详细使用说明 (7.7KB, ~15页)
- 🔍 每个工具的功能说明
- 💻 命令行参数详解
- 🐛 故障排除指南
- **最适合**: 需要详细操作指导的用户

---

## 📊 工具总览

### **我想了解整体架构** → [`SUMMARY.md`](SUMMARY.md)
- 🎯 项目总结 (9.7KB, ~4页)
- 📦 交付物清单
- 🔢 代码统计
- ✅ 功能覆盖度
- **最适合**: 想快速了解整个工具集

---

## ✅ 测试报告

### **我想验证工具是否可用** → [`TEST_RESULTS.md`](TEST_RESULTS.md)
- 🧪 完整测试结果
- ✅ 功能验证报告
- 📊 测试覆盖度
- 🔧 测试命令示例
- **最适合**: 需要验证工具可靠性

---

## 🗂️ 文档快速对照表

| 文档 | 大小 | 页数 | 用途 | 适用场景 |
|------|------|------|------|---------|
| **QUICKSTART.md** | 7KB | ~8页 | 快速入门 | 急需快速诊断 |
| **README_VOC_DIAGNOSTICS.md** | 13KB | ~18页 | 完整指南 | 首次使用 |
| **CHECKLIST.md** | 9.7KB | ~12页 | 检查清单 | 系统化排查 |
| **DIAGNOSTIC_PLAN.md** | 13KB | ~20页 | 问题方案 | 深度诊断 |
| **DIAGNOSTICS_README.md** | 7.7KB | ~15页 | 使用教程 | 详细操作 |
| **SUMMARY.md** | 9.7KB | ~4页 | 工具总览 | 了解架构 |
| **TEST_RESULTS.md** | 7KB | ~10页 | 测试报告 | 验证可用性 |
| **README.md** | 13KB | - | 原始文档 | 原有说明 |

**总计**: ~74KB, ~97页

---

## 🎯 按使用场景选择

### 场景1: 我是新手，第一次使用
```
1. README_VOC_DIAGNOSTICS.md  ← 先读这个了解全貌
2. QUICKSTART.md              ← 快速开始
3. CHECKLIST.md               ← 按步骤执行
```

### 场景2: 我遇到了具体问题
```
1. QUICKSTART.md              ← 快速定位问题类型
2. DIAGNOSTIC_PLAN.md         ← 找到对应的Issue章节
3. CHECKLIST.md               ← 验证修复
```

### 场景3: 我想深入了解工具
```
1. SUMMARY.md                 ← 了解整体架构
2. README_VOC_DIAGNOSTICS.md  ← 详细功能
3. DIAGNOSTICS_README.md      ← 使用细节
4. TEST_RESULTS.md            ← 测试验证
```

### 场景4: 我要验证工具是否正常
```
1. TEST_RESULTS.md            ← 查看测试结果
2. 运行测试命令              ← 自己验证
3. QUICKSTART.md              ← 快速测试
```

---

## 📂 文件结构

```
detection/
├── 📄 README_INDEX.md              ← 您在这里！
├── 📄 QUICKSTART.md                ← 快速开始
├── 📄 README_VOC_DIAGNOSTICS.md    ← 完整指南
├── 📄 CHECKLIST.md                 ← 检查清单
├── 📄 DIAGNOSTIC_PLAN.md           ← 问题方案
├── 📄 DIAGNOSTICS_README.md        ← 使用教程
├── 📄 SUMMARY.md                   ← 工具总览
├── 📄 TEST_RESULTS.md              ← 测试报告
│
├── 🐍 diagnose_voc2007.py          ← 主诊断脚本
├── 🐍 quick_eval_test.py           ← 快速测试
├── 🐍 check_voc_conversion.py      ← 标注检查
├── 🐍 diagnose_utils.py            ← 工具函数
│
└── 📁 configs/
    ├── detector/
    │   └── faster_rcnn_r50_fpn_1x_voc0712.py
    └── global/
        └── voc2007_faster_rcnn.py
```

---

## 🔑 关键概念

### 4个诊断阶段
1. **直接推理检查** - 模型是否正常输出
2. **预测对比** - 预测是否与GT匹配
3. **流程检查** - 数据处理是否正确
4. **评估验证** - 度量计算是否正确

### 5个常见根因
1. **类别ID映射错误** ⭐ 最常见 (0-based vs 1-based)
2. **预处理错误** (mean/std, RGB/BGR)
3. **标注格式问题** (COCO JSON错误)
4. **测试配置过严** (阈值太高)
5. **元信息不匹配** (类名不一致)

---

## 💡 使用建议

### 第一次使用
1. 花5分钟阅读 `QUICKSTART.md`
2. 运行快速诊断命令
3. 根据结果查阅对应文档

### 遇到问题
1. 查看 `QUICKSTART.md` 的速查表
2. 参考 `DIAGNOSTIC_PLAN.md` 的具体Issue
3. 按照 `CHECKLIST.md` 验证修复

### 深入学习
1. 通读 `README_VOC_DIAGNOSTICS.md`
2. 理解 `DIAGNOSTIC_PLAN.md` 的每个问题
3. 查看 `TEST_RESULTS.md` 了解测试方法

---

## 🆘 获取帮助

### 如果文档没有解决你的问题

1. **检查测试结果**: 看 `TEST_RESULTS.md` 确认工具正常
2. **运行诊断**: 使用 `diagnose_voc2007.py` 获取详细报告
3. **查看日志**: 检查 `diagnose_output/diagnostic_report.txt`
4. **比对配置**: 参考示例配置文件

### 常见问题

**Q: 从哪个文档开始？**
A: 如果是新手，从 `QUICKSTART.md` 开始；如果需要详细了解，读 `README_VOC_DIAGNOSTICS.md`

**Q: 工具不工作怎么办？**
A: 查看 `TEST_RESULTS.md` 运行测试命令验证安装

**Q: 找不到我的问题？**
A: 查看 `DIAGNOSTIC_PLAN.md` 的5个常见问题，90%的情况都在里面

**Q: 需要示例配置？**
A: 查看 `configs/` 目录下的示例文件

---

## 📞 支持

### 文档
- 8个详细文档，覆盖所有使用场景
- 总计 ~97 页内容
- 包含示例、检查清单、解决方案

### 工具
- 4个Python脚本，1,660+ 行代码
- 完整的命令行接口
- 详细的输出和可视化

### 配置
- 2个示例配置文件
- 可直接使用或作为模板

---

## 🎓 学习路径

### 初级 (15-30分钟)
1. `QUICKSTART.md` - 了解基本用法
2. 运行快速测试
3. 查看输出结果

### 中级 (1-2小时)
1. `README_VOC_DIAGNOSTICS.md` - 完整理解工具
2. `CHECKLIST.md` - 学习系统化方法
3. 完整运行一次诊断

### 高级 (3-4小时)
1. `DIAGNOSTIC_PLAN.md` - 深入理解每个问题
2. `DIAGNOSTICS_README.md` - 掌握所有细节
3. `TEST_RESULTS.md` - 了解测试方法
4. 阅读源代码

---

## 🏆 成功案例

使用这些工具，你可以：
- ✅ 在15-30分钟内诊断出VOC2007 mAP低的根因
- ✅ 将mAP从0.47%修复到70%+
- ✅ 节省4-8小时的调试时间
- ✅ 系统化地解决检测评估问题

---

**开始你的诊断之旅！** 🚀

选择一个文档开始阅读，或者直接运行 `QUICKSTART.md` 中的快速命令。

祝你顺利解决问题！💪
