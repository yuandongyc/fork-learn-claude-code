执行流程图在 src/components/architecture/execution-flow.tsx。
当前字体大小设置：
- 决策节点 (decision): fontSize={10} (第 65 行)
- 开始/结束节点: fontSize={12} (第 94 行)
- 普通处理节点: fontSize={11} (第 126 行)
- 边上的标签: fontSize={10} (第 170 行)
---
修改计划
需要修改的文件： src/components/architecture/execution-flow.tsx
修改方案：
位置	当前值	建议改为
第 65 行 (决策节点)	10	12
第 94 行 (开始/结束)	12	14
第 126 行 (普通节点)	11	13
第 170 行 (边标签)	10	12
同时可能还需要调整节点尺寸：
- 第 9 行: NODE_WIDTH = 140 → 160 (加宽)
- 第 10 行: NODE_HEIGHT = 40 → 48 (加高)
---