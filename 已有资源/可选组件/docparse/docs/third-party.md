# 第三方代码

## Vendored（已并入本仓库）

为让 `docparse` 运行时自包含，以下文件从 [SeerRay-Lab/Xiaomi-OCR-0](https://github.com/SeerRay-Lab/Xiaomi-OCR-0) 复制到 `docparse/vendor/`：

| 本仓库路径 | 上游路径 | 用途 |
|---|---|---|
| `docparse/vendor/otsl.py` | `postprocess/otsl.py` | OTSL 表格 → HTML |
| `docparse/vendor/repetition_guard.py` | `postprocess/repetition_guard.py` | 重复/循环检测 |
| `docparse/vendor/markdown_merge.py` | `postprocess/markdown_merge.py` | 区域结果按阅读顺序合并 |
| `docparse/vendor/document.py` | `postprocess/document.py` | 整页后处理（公式定界符、OTSL 转换、截断重复） |
| `docparse/vendor/cropping.py` | `evaluation_pipeline/cropping.py` | 归一化 bbox 裁块 |

修改内容：**仅把 `from postprocess.otsl import` 改为 `from .otsl import`**，其余逻辑未改动。每个文件头部记录了来源。

许可：上游为 **Apache-2.0**，完整文本见 `docparse/vendor/LICENSE-Xiaomi-OCR-0.txt`。

## 未 vendored

以下上游内容不参与运行时，也不需要克隆：

- 模型权重与 GGUF 转换脚本
- `evaluation_pipeline/` 的评测服务与 vLLM 启动逻辑
- `demo/`、`skills/`、示例图片与基准数据
- `postprocess/region.py`、`postprocess/assemble_markdown.py`（未被调用）

如需复现上游基准或查看示例，再按根 README 的方式克隆上游仓库即可。
