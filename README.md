# ComfyUI H3 NEAT ImageFusion

**多帧共识融合 / Multi-frame consensus fusion for MiniMax H3 still-image workflows**

H3 NEAT ImageFusion 是一个面向 ComfyUI 的后处理节点：把 H3 生成的短帧序列作为一次「计算摄影连拍」，以锚定帧为基准，对后续帧进行曝光匹配、光流对齐和时域共识融合，抑制单帧伪纹、彩色噪纹与部分不稳定细节。

This is a custom ComfyUI post-processing node that blends a short H3 frame burst into a single still image. It uses exposure matching, optical-flow alignment, consensus-based artifact repair, chroma cleanup and conservative detail enhancement.

> **Versions**: release **v1.0.0**, ImageFusion core **v0.3.1**. This repository contains the node source and the example ComfyUI workflow, **not** the MiniMax H3 model weights or NEAT Prime weights.

## 功能 / Features

- 以第 1 帧为结构锚点，后续帧提供冗余信息 (anchor-first fusion).
- 曝光匹配 + 稠密光流对齐 (exposure matching and dense optical flow).
- Temporal Consensus Artifact Repair: 以跨帧证据压制孤立伪影。
- 色度清理、鬼影抑制、细节保护和纹理整合。
- 输出融合图、锚点图、融合置信度。实际输出以节点连接为准。

## 安装 / Install

1. 将 [`custom_nodes/ComfyUI_H3_NEAT_ImageFusion`](custom_nodes/ComfyUI_H3_NEAT_ImageFusion) **整个文件夹**复制到 `ComfyUI/custom_nodes/`。
2. 在 ComfyUI 使用的 Python 环境中确保可导入 `torch`、`numpy` 和 `cv2`（OpenCV）。不要随意覆盖已经安装的 ComfyUI CUDA/PyTorch 组合。
3. 重启 ComfyUI，在节点列表中寻找 **H3 NEAT ImageFusion · 时域共识**（节点标识为 `H3BurstDenoise5`）。
4. 导入 [`workflows/H3_NEAT_IMAGE.json`](workflows/H3_NEAT_IMAGE.json)。

> **注意 / Important**: 提供的**完整示例工作流**还依赖 MiniMax H3 / H3 NEAT Prime 的模型与自定义节点，以及 **DMAD** sampler 节点。仅安装 ImageFusion 文件夹，只能提供融合节点，不能自动让整个 H3 生成工作流运行。模型名称和安装路径需要根据你的环境调整。The workflow depends on external H3 and DMAD nodes and model files; these are not bundled here.

## 建议参数 / Reference settings

| 参数 | 参考值 |
| --- | --- |
| 画面目标 | 16:9 / 3.5 MP |
| Image Mode 推理 | 10 steps |
| `anchor_frame` | `1` |
| `denoise_strength` | `0.72` |
| `exposure_match` | `true` |
| `align` | `true` |
| `ghost_guard` | `0.82` |
| `detail_protect` | `0.88` |
| `chroma_clean` | `0.82` |
| `texture_solid` | `0.52` |
| `artifact_repair` | `0.85` |

这组参数是原始发布包的预设，非所有场景的通用最优值。实际效果受输入帧相似度、光流质量、显存和模型输出影响。

## 输出 / Outputs

- `H3_NEAT_FINAL_...png`: 最终融合图。
- `H3_NEAT_SOURCE_...png`: 原始源图（由示例工作流内部保存）。

## 文件结构 / Repository layout

```text
custom_nodes/ComfyUI_H3_NEAT_ImageFusion/
  __init__.py
  nodes.py
  burst_fusion.py
  requirements.txt
  README.md
  example_workflows/H3_NEAT_IMAGE.json
workflows/H3_NEAT_IMAGE.json
CHANGELOG.md
```

## 状态 / Status

v1.0.0 是历史发布包的源码整理版。已检查文件结构和 Python 语法；**尚未在所有 ComfyUI/H3 运行环境上重新进行端到端验证**。遇到节点缺失，请先确认 H3、DMAD 和模型依赖齐备。

本仓库未附带独立开源许可证。公开可读不等于自动授权第三方复制、改作或再分发；许可证将由维护者另行明确。模型及第三方项目的许可证分别遵循其原始条款。

---

*H3 NEAT ImageFusion is a community workflow and node project, not an official MiniMax release.*