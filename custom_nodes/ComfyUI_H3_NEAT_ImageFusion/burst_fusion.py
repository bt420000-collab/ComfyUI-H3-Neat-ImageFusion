from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np
import torch

_EPS = 1e-6


def _luma(rgb: np.ndarray) -> np.ndarray:
    return (
        rgb[..., 0] * 0.2126
        + rgb[..., 1] * 0.7152
        + rgb[..., 2] * 0.0722
    ).astype(np.float32)


def match_exposure_to_anchor(
    anchor: np.ndarray,
    frame: np.ndarray,
) -> Tuple[np.ndarray, float, float]:
    """Match frame luminance distribution to anchor while preserving chroma.

    Uses robust 10/50/90 percentiles instead of pixel correspondence, so it can
    run before optical-flow alignment and tolerates the tiny H3 motion between
    the five frames.
    """
    ya = _luma(anchor)
    yf = _luma(frame)

    qa = np.percentile(ya, [10.0, 50.0, 90.0]).astype(np.float32)
    qf = np.percentile(yf, [10.0, 50.0, 90.0]).astype(np.float32)

    # Robust affine photometric match. Fall back to median gain if contrast is
    # nearly flat (e.g. a close-up wall or a dark patch).
    if float(qf[2] - qf[0]) > 0.025:
        gain, offset = np.polyfit(qf, qa, 1)
    else:
        gain = float(qa[1] / max(float(qf[1]), 0.02))
        offset = 0.0

    gain = float(np.clip(gain, 0.65, 1.60))
    offset = float(np.clip(offset, -0.15, 0.15))

    y_target = np.clip(yf * gain + offset, 0.0, 1.0)
    ratio = y_target / np.maximum(yf, 0.015)
    ratio = np.clip(ratio, 0.55, 1.85)[..., None]
    corrected = np.clip(frame * ratio, 0.0, 1.0).astype(np.float32)
    return corrected, gain, offset


def _rgb_to_ycbcr(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    cb = (b - y) * 0.5389
    cr = (r - y) * 0.6350
    return y.astype(np.float32), cb.astype(np.float32), cr.astype(np.float32)


def _ycbcr_to_rgb(y: np.ndarray, cb: np.ndarray, cr: np.ndarray) -> np.ndarray:
    r = y + cr / 0.6350
    b = y + cb / 0.5389
    g = (y - 0.2126 * r - 0.0722 * b) / 0.7152
    return np.clip(np.stack([r, g, b], axis=-1), 0.0, 1.0).astype(np.float32)



def _apply_temporal_artifact_repair(
    base: np.ndarray,
    anchor: np.ndarray,
    stack: np.ndarray,
    strength: float,
    detail_protect: float,
) -> np.ndarray:
    """Remove anchor-only mid/fine luminance artifacts using temporal consensus.

    H3 5F often keeps the true geometry stable while tiny AI/VAE ripples drift
    between frames.  The references therefore act as witnesses: only detail that
    is an anchor outlier *and* sits in a region where the references agree is
    eligible for repair.  Very low-frequency lighting is never replaced.
    """
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 1e-6 or stack.shape[0] < 3:
        return base

    refs = stack[1:]
    y_base, cb_base, cr_base = _rgb_to_ycbcr(base)
    y_anchor = _luma(anchor)
    y_refs = np.stack([_luma(img) for img in refs], axis=0).astype(np.float32)
    y_ref_med = np.median(y_refs, axis=0).astype(np.float32)
    y_ref_mad = np.median(np.abs(y_refs - y_ref_med[None]), axis=0).astype(np.float32)

    # Anchor-vs-reference disagreement, split into fine and middle scales.
    # The broad residual is treated as lighting/exposure and intentionally kept.
    delta = (y_anchor - y_ref_med).astype(np.float32)
    d_s1 = cv2.GaussianBlur(delta, (0, 0), sigmaX=1.05)
    d_s4 = cv2.GaussianBlur(delta, (0, 0), sigmaX=4.2)
    d_s10 = cv2.GaussianBlur(delta, (0, 0), sigmaX=10.0)
    fine = delta - d_s1
    mid = d_s1 - d_s4
    broad_mid = d_s4 - d_s10
    repair_signal = fine + 0.95 * mid + 0.72 * broad_mid

    # A true repair candidate needs all three: references agree, anchor differs,
    # and the location is not a strong structural edge.
    agreement = np.exp(-np.square(y_ref_mad / 0.010)).astype(np.float32)
    evidence = np.clip((np.abs(fine) + 0.90 * np.abs(mid) + 0.65 * np.abs(broad_mid) - 0.0008) / 0.010, 0.0, 1.0).astype(np.float32)
    # Protect structure that the references also see. Using the anchor edge map
    # here would mistakenly protect the very anchor-only ripple we want to remove.
    ref_rgb_med = np.median(refs, axis=0).astype(np.float32)
    y_struct = _luma(ref_rgb_med)
    gx_s = cv2.Sobel(y_struct, cv2.CV_32F, 1, 0, ksize=3)
    gy_s = cv2.Sobel(y_struct, cv2.CV_32F, 0, 1, ksize=3)
    grad_s = cv2.magnitude(gx_s, gy_s)
    # Absolute rather than percentile-only scaling: a smooth skin gradient must
    # not be promoted into an "edge" just because the whole crop is smooth.
    edge = np.clip((grad_s - 0.004) / 0.032, 0.0, 1.0).astype(np.float32)
    edge = cv2.GaussianBlur(edge, (0, 0), sigmaX=0.75)
    dp = float(np.clip(detail_protect, 0.0, 1.0))
    edge_guard = np.clip(1.0 - (0.62 + 0.34 * dp) * edge, 0.03, 1.0).astype(np.float32)

    artifact = np.clip(agreement * evidence * edge_guard, 0.0, 1.0)
    # Slightly soften the decision map so repaired zones do not look patched.
    artifact = cv2.GaussianBlur(artifact, (0, 0), sigmaX=0.65)
    artifact = np.clip(artifact, 0.0, 1.0).astype(np.float32)

    # Never inject a large reference change.  We only subtract unsupported
    # middle/high-frequency residue from the current base.
    correction = np.clip(repair_signal, -0.050, 0.050)
    y_out = np.clip(y_base - strength * artifact * correction, 0.0, 1.0).astype(np.float32)
    return _ycbcr_to_rgb(y_out, cb_base, cr_base)

def _apply_chroma_consensus(
    base: np.ndarray,
    anchor: np.ndarray,
    stack: np.ndarray,
    confidence: np.ndarray,
    edge: np.ndarray,
    strength: float,
) -> np.ndarray:
    """Clean unstable color micro-texture while preserving base luminance/geometry.

    The H3 burst often carries tiny frame-to-frame chroma ripples on otherwise
    stable skin.  We keep the base Y channel and move only Cb/Cr toward the
    robust temporal median, with strong edge/ghost protection.
    """
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 1e-6:
        return base

    y_base, cb_base, cr_base = _rgb_to_ycbcr(base)
    # Let the four references vote without the anchor biasing their median.
    refs = stack[1:] if stack.shape[0] > 1 else stack
    chroma = []
    for img in refs:
        _, cb, cr = _rgb_to_ycbcr(img)
        chroma.append(np.stack([cb, cr], axis=-1))
    cstack = np.stack(chroma, axis=0).astype(np.float32)
    cmed = np.median(cstack, axis=0).astype(np.float32)
    cmad = np.median(np.abs(cstack - cmed[None]), axis=0).mean(axis=2).astype(np.float32)

    _, cb_anchor, cr_anchor = _rgb_to_ycbcr(anchor)
    anchor_dev = np.sqrt(np.square(cb_anchor - cmed[..., 0]) + np.square(cr_anchor - cmed[..., 1])).astype(np.float32)

    # Stable reference chroma + anchor-specific deviation is the signature of
    # colored AI micro-ripples. Shared color edges stay protected.
    stability = np.exp(-np.square(cmad / 0.020)).astype(np.float32)
    outlier = np.clip((anchor_dev - 0.0015) / 0.022, 0.0, 1.0).astype(np.float32)
    edge_guard = np.clip(1.0 - 0.92 * edge, 0.04, 1.0).astype(np.float32)
    conf_gate = np.clip(0.30 + 1.25 * confidence, 0.0, 1.0).astype(np.float32)
    w = np.clip(strength * stability * (0.30 + 0.70 * outlier) * edge_guard * conf_gate, 0.0, 0.96)

    cb = cb_base * (1.0 - w) + cmed[..., 0] * w
    cr = cr_base * (1.0 - w) + cmed[..., 1] * w
    return _ycbcr_to_rgb(y_base, cb, cr)


def _apply_texture_solid(
    base: np.ndarray,
    anchor: np.ndarray,
    stack: np.ndarray,
    confidence: np.ndarray,
    strength: float,
) -> np.ndarray:
    """Reinforce consensus-supported high-frequency structure.

    Designed for H3 hair/fabric texture: low frequencies stay in the base image;
    only coherent high-frequency residuals from the aligned burst are injected.
    Flat skin is protected automatically by a texture gate.
    """
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 1e-6:
        return base

    y_base, cb, cr = _rgb_to_ycbcr(base)
    y_anchor = _luma(anchor)
    refs = stack[1:] if stack.shape[0] > 1 else stack
    ys = np.stack([_luma(img) for img in refs], axis=0).astype(np.float32)

    def highpass(y):
        return y - cv2.GaussianBlur(y, (0, 0), sigmaX=0.85)

    hf_stack = np.stack([highpass(y) for y in ys], axis=0).astype(np.float32)
    hf_med = np.median(hf_stack, axis=0).astype(np.float32)
    hf_mad = np.median(np.abs(hf_stack - hf_med[None]), axis=0).astype(np.float32)
    hf_base = highpass(y_base).astype(np.float32)

    # Directional consensus, sign-invariant via the doubled-angle structure
    # tensor. A single crisp hallucinated strand cannot win by sharpness alone.
    c2_acc = np.zeros_like(y_anchor, dtype=np.float32)
    s2_acc = np.zeros_like(y_anchor, dtype=np.float32)
    w_acc = np.zeros_like(y_anchor, dtype=np.float32)
    for y in ys:
        gx_r = cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3)
        gy_r = cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3)
        mag2 = gx_r * gx_r + gy_r * gy_r + 1e-8
        mag = np.sqrt(mag2)
        c2_acc += mag * ((gx_r * gx_r - gy_r * gy_r) / mag2)
        s2_acc += mag * ((2.0 * gx_r * gy_r) / mag2)
        w_acc += mag
    orientation_coherence = np.sqrt(c2_acc * c2_acc + s2_acc * s2_acc) / np.maximum(w_acc, 1e-6)
    orientation_coherence = np.clip(orientation_coherence, 0.0, 1.0).astype(np.float32)

    # Robust best-detail estimate: among candidates that agree with the median
    # sign and stay within a MAD envelope, take the strongest supported detail.
    dev = np.abs(hf_stack - hf_med[None])
    valid = (hf_stack * hf_med[None] >= 0.0) & (dev <= (2.5 * hf_mad[None] + 0.006))
    candidate_strength = np.where(valid, np.abs(hf_stack), -1.0)
    best_idx = np.argmax(candidate_strength, axis=0)
    hf_best = np.take_along_axis(hf_stack, best_idx[None], axis=0)[0]
    any_valid = np.max(candidate_strength, axis=0) >= 0.0
    hf_best = np.where(any_valid, hf_best, hf_med).astype(np.float32)
    hf_target = (0.68 * hf_med + 0.32 * hf_best).astype(np.float32)

    # Texture gate: require real local structure and temporal agreement.
    gx = cv2.Sobel(y_anchor, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(y_anchor, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    g50 = max(float(np.percentile(grad, 55.0)), 0.004)
    g90 = max(float(np.percentile(grad, 90.0)), g50 + 0.006)
    texture_gate = np.clip((grad - g50) / (g90 - g50), 0.0, 1.0).astype(np.float32)

    # Dark textured regions are commonly hair, so give them a modest boost while
    # keeping this generic enough to work on fabric/wood as well.
    dark_bias = np.clip((0.62 - y_anchor) / 0.42, 0.0, 1.0).astype(np.float32)
    texture_gate *= (0.35 + 0.65 * dark_bias)

    agreement = np.exp(-np.square(hf_mad / (np.abs(hf_med) + 0.008))).astype(np.float32)
    evidence = np.clip(np.abs(hf_med) / 0.018, 0.0, 1.0).astype(np.float32)
    direction_gate = np.clip((orientation_coherence - 0.30) / 0.62, 0.0, 1.0).astype(np.float32)
    conf_gate = np.clip(0.25 + 1.10 * confidence, 0.0, 1.0).astype(np.float32)
    gate = texture_gate * agreement * evidence * direction_gate * conf_gate

    # Only move toward consensus-gated best detail; cap the injection so it
    # cannot become a generic sharpening halo.
    delta = np.clip(hf_target - hf_base, -0.035, 0.035)
    y_out = np.clip(y_base + strength * gate * delta, 0.0, 1.0).astype(np.float32)
    return _ycbcr_to_rgb(y_out, cb, cr)


def _resize_for_flow(gray: np.ndarray, max_dim: int = 960) -> tuple[np.ndarray, float]:
    h, w = gray.shape
    scale = min(1.0, float(max_dim) / float(max(h, w)))
    if scale >= 0.999:
        return gray, 1.0
    small = cv2.resize(gray, (max(8, int(round(w * scale))), max(8, int(round(h * scale)))), interpolation=cv2.INTER_AREA)
    return small, scale


def _align_to_anchor(anchor: np.ndarray, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Dense DIS optical-flow warp from frame into anchor coordinates."""
    ag = np.clip(_luma(anchor) * 255.0, 0, 255).astype(np.uint8)
    fg = np.clip(_luma(frame) * 255.0, 0, 255).astype(np.uint8)

    ag_s, scale = _resize_for_flow(ag)
    if scale < 0.999:
        fg_s = cv2.resize(fg, (ag_s.shape[1], ag_s.shape[0]), interpolation=cv2.INTER_AREA)
    else:
        fg_s = fg

    # Flow from each anchor pixel to its corresponding source-frame pixel.
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    dis.setUseSpatialPropagation(True)
    flow = dis.calc(ag_s, fg_s, None)

    h, w = ag.shape
    if scale < 0.999:
        flow = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR)
        flow[..., 0] /= scale
        flow[..., 1] /= scale

    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = xx + flow[..., 0].astype(np.float32)
    map_y = yy + flow[..., 1].astype(np.float32)
    warped = cv2.remap(
        frame,
        map_x,
        map_y,
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT101,
    )
    return np.clip(warped, 0.0, 1.0).astype(np.float32), flow.astype(np.float32)


def _edge_map(anchor: np.ndarray) -> np.ndarray:
    y = _luma(anchor)
    gx = cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3)
    g = cv2.magnitude(gx, gy)
    scale = float(np.percentile(g, 90.0))
    if scale < 1e-5:
        return np.zeros_like(g, dtype=np.float32)
    e = np.clip(g / (scale * 1.35 + _EPS), 0.0, 1.0)
    e = cv2.GaussianBlur(e, (0, 0), sigmaX=0.7)
    return np.clip(e, 0.0, 1.0).astype(np.float32)


def _sharpness_score(img: np.ndarray) -> float:
    y = _luma(img)
    lap = cv2.Laplacian(y, cv2.CV_32F, ksize=3)
    return float(np.mean(np.abs(lap)))


def fuse_h3_burst(
    images: torch.Tensor,
    anchor_frame: int = 1,
    denoise_strength: float = 0.72,
    exposure_match: bool = True,
    align: bool = True,
    ghost_guard: float = 0.82,
    detail_protect: float = 0.88,
    chroma_clean: float = 0.72,
    texture_solid: float = 0.42,
    artifact_repair: float = 0.68,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse an H3 5-frame IMAGE batch into one anchor-preserving still.

    Strategy:
      1) Frame 1 (or selected frame) owns geometry, color and composition.
      2) Other frames are photometrically matched to the anchor.
      3) Dense optical flow warps them into anchor coordinates.
      4) Robust consensus rejects moving/changed/ghost regions.
      5) Temporal consensus removes anchor-only multi-scale luminance/chroma
         artifacts without replacing low-frequency lighting.
      6) Direction-consistent high-frequency evidence can reinforce hair/fabric
         detail without accepting a single-frame sharp hallucination.
    """
    if not isinstance(images, torch.Tensor):
        raise TypeError("images must be a torch.Tensor in ComfyUI IMAGE format")
    if images.ndim != 4 or images.shape[-1] not in (3, 4):
        raise ValueError("Expected IMAGE batch shaped [N,H,W,3/4]")
    if images.shape[0] < 2:
        raise ValueError("H3 Burst Fusion needs at least 2 frames")

    n = min(int(images.shape[0]), 5)
    idx = int(np.clip(anchor_frame - 1, 0, n - 1))
    original_device = images.device
    original_dtype = images.dtype

    rgb = images[:n, ..., :3].detach().to(device="cpu", dtype=torch.float32).numpy()
    rgb = np.clip(rgb, 0.0, 1.0).astype(np.float32)
    anchor = rgb[idx].copy()

    aligned = [anchor]
    flows = [np.zeros((*anchor.shape[:2], 2), dtype=np.float32)]

    for i in range(n):
        if i == idx:
            continue
        f = rgb[i]
        if exposure_match:
            f, _, _ = match_exposure_to_anchor(anchor, f)
        if align:
            f, flow = _align_to_anchor(anchor, f)
        else:
            flow = np.zeros((*anchor.shape[:2], 2), dtype=np.float32)
        aligned.append(f)
        flows.append(flow)

    stack = np.stack(aligned, axis=0)
    consensus = np.median(stack, axis=0).astype(np.float32)
    edge = _edge_map(anchor)

    # Higher ghost_guard -> stricter admission of changed pixels.
    gg = float(np.clip(ghost_guard, 0.0, 1.0))
    tau_anchor = 0.078 - 0.042 * gg
    tau_consensus = tau_anchor * 1.45

    dp = float(np.clip(detail_protect, 0.0, 1.0))
    edge_gate = np.clip(1.0 - dp * edge, 0.08, 1.0)

    anchor_sharp = max(_sharpness_score(anchor), 1e-5)
    numerator = anchor.astype(np.float32) * (1.15 + 1.65 * dp * edge)[..., None]
    denominator = (1.15 + 1.65 * dp * edge).astype(np.float32)
    aux_weight_sum = np.zeros(anchor.shape[:2], dtype=np.float32)

    strength = float(np.clip(denoise_strength, 0.0, 1.0))

    # aligned[0] is always the anchor; remaining entries are references.
    for ref_i in range(1, len(aligned)):
        ref = aligned[ref_i]
        d_anchor = np.mean(np.abs(ref - anchor), axis=2)
        d_cons = np.mean(np.abs(ref - consensus), axis=2)

        w_anchor = np.exp(-np.square(d_anchor / max(tau_anchor, 1e-4))).astype(np.float32)
        w_cons = np.exp(-np.square(d_cons / max(tau_consensus, 1e-4))).astype(np.float32)

        sharp_ratio = _sharpness_score(ref) / anchor_sharp
        quality = float(np.clip(sharp_ratio, 0.55, 1.10))

        flow_mag = np.linalg.norm(flows[ref_i], axis=2)
        # H3 5f motion is normally tiny. Very large inferred motion is often a
        # bad correspondence, so fade it rather than letting it smear the face.
        motion_gate = np.exp(-np.square(np.maximum(flow_mag - 14.0, 0.0) / 12.0)).astype(np.float32)

        w = strength * quality * edge_gate * w_anchor * w_cons * motion_gate
        numerator += ref * w[..., None]
        denominator += w
        aux_weight_sum += w

    fused = numerator / np.maximum(denominator[..., None], _EPS)
    fused = np.clip(fused, 0.0, 1.0).astype(np.float32)

    # Confidence means "how much trustworthy multi-frame evidence contributed".
    confidence = np.clip(aux_weight_sum / max(len(aligned) - 1, 1), 0.0, 1.0)

    # Finishing pass: first remove anchor-only luminance ripples by temporal
    # consensus, then clean chroma micro-artifacts, then reinforce only
    # consensus-supported high-frequency texture. No generative redraw.
    fused = _apply_temporal_artifact_repair(
        fused, anchor, stack, artifact_repair, detail_protect
    )
    fused = _apply_chroma_consensus(
        fused, anchor, stack, confidence, edge, chroma_clean
    )
    fused = _apply_texture_solid(
        fused, anchor, stack, confidence, texture_solid
    )
    confidence_rgb = np.repeat(confidence[..., None], 3, axis=2).astype(np.float32)

    fused_t = torch.from_numpy(fused[None]).to(device=original_device, dtype=original_dtype)
    anchor_t = torch.from_numpy(anchor[None]).to(device=original_device, dtype=original_dtype)
    conf_t = torch.from_numpy(confidence_rgb[None]).to(device=original_device, dtype=original_dtype)
    return fused_t, anchor_t, conf_t