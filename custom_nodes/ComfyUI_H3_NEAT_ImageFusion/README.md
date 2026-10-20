# ComfyUI H3 NEAT ImageFusion

H3 NEAT Image 的多帧时域共识计算摄影后处理节点。

发布包版本：**v1.0.0**  
ImageFusion 核心：**v0.3.1**

## 功能

- 以第 1 帧作为结构与曝光锚点。
- 后续帧做曝光匹配与稠密光流对齐。
- Temporal Consensus Artifact Repair：删除只有锚点存在、其余帧不支持的明暗/色彩伪纹。
- 色度鲁棒融合：压制鼻梁、面颊等位置的不自然彩色细纹。
- 共识细节补强：让头发、织物、木纹更实，同时抑制单帧幻觉细节。
- 输出最终融合图、锚定原图与融合置信度。

## 发布工作流默认参数

- 目标分辨率：3.5 MP
- Steps：10
- artifact_repair：0.85
- denoise_strength：0.72
- ghost_guard：0.82
- detail_protect：0.88
- chroma_clean：0.82
- texture_solid：0.52

## 图片命名

- `H3_NEAT_FINAL_...png`：最终融合成图
- `H3_NEAT_SOURCE_...png`：内部保存的原始连拍源图

## 安装

把 `ComfyUI_H3_NEAT_ImageFusion` 整个目录复制到：

`ComfyUI/custom_nodes/`

然后重启 ComfyUI。

发布包中的工作流：

`workflows/H3_NEAT_IMAGE.json`

依赖：ComfyUI 当前 Python 环境中的 `torch`、`numpy`、`cv2`。如果环境缺少 OpenCV，请安装与该 ComfyUI Python 环境兼容的 OpenCV 包。