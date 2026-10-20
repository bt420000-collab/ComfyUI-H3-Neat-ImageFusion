from __future__ import annotations

try:
    from .burst_fusion import fuse_h3_burst
except ImportError:  # allows direct module tests
    from burst_fusion import fuse_h3_burst


class H3BurstDenoise5:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "anchor_frame": ("INT", {"default": 1, "min": 1, "max": 5, "step": 1}),
                "denoise_strength": ("FLOAT", {"default": 0.72, "min": 0.0, "max": 1.0, "step": 0.01}),
                "exposure_match": ("BOOLEAN", {"default": True}),
                "align": ("BOOLEAN", {"default": True}),
                "ghost_guard": ("FLOAT", {"default": 0.82, "min": 0.0, "max": 1.0, "step": 0.01}),
                "detail_protect": ("FLOAT", {"default": 0.88, "min": 0.0, "max": 1.0, "step": 0.01}),
                "chroma_clean": ("FLOAT", {"default": 0.72, "min": 0.0, "max": 1.0, "step": 0.01}),
                "texture_solid": ("FLOAT", {"default": 0.42, "min": 0.0, "max": 1.0, "step": 0.01}),
                "artifact_repair": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("融合图", "锚定原图", "融合置信度")
    FUNCTION = "fuse"
    CATEGORY = "H3/Image"
    DESCRIPTION = "H3 多帧时域共识计算摄影：首帧锚定；参考帧投票删除无共识明暗/色彩伪纹，并用方向一致的高频细节补强头发与纹理。"

    def fuse(
        self,
        images,
        anchor_frame=1,
        denoise_strength=0.72,
        exposure_match=True,
        align=True,
        ghost_guard=0.82,
        detail_protect=0.88,
        chroma_clean=0.72,
        texture_solid=0.42,
        artifact_repair=0.85,
    ):
        return fuse_h3_burst(
            images,
            anchor_frame=anchor_frame,
            denoise_strength=denoise_strength,
            exposure_match=exposure_match,
            align=align,
            ghost_guard=ghost_guard,
            detail_protect=detail_protect,
            chroma_clean=chroma_clean,
            texture_solid=texture_solid,
            artifact_repair=artifact_repair,
        )


NODE_CLASS_MAPPINGS = {
    "H3BurstDenoise5": H3BurstDenoise5,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3BurstDenoise5": "H3 NEAT ImageFusion · 时域共识",
}