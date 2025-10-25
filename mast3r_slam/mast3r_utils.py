import PIL
import numpy as np
import torch
import einops

import mast3r.utils.path_to_dust3r  # noqa
from dust3r.utils.image import ImgNorm
from mast3r.model import AsymmetricMASt3R
from mast3r_slam.retrieval_database import RetrievalDatabase
from mast3r_slam.config import config
import mast3r_slam.matching as matching

# FP8 optimization support
try:
    import transformer_engine.pytorch as te
    from transformer_engine.common import recipe
    FP8_AVAILABLE = True
    fp8_recipe = te.fp8.DelayedScaling(margin=0, fp8_format=recipe.Format.E4M3)
except ImportError:
    FP8_AVAILABLE = False
    print("[Warning] Transformer Engine not available, FP8 acceleration disabled")

# Global flag to control FP8 usage
USE_FP8 = False


def get_use_fp8():
    """Check if FP8 is currently enabled"""
    return USE_FP8


def load_mast3r(path=None, device="cuda", use_fp8=False):
    """Load MASt3R model with optional FP8 optimization.
    
    Args:
        path: Path to model checkpoint
        device: Device to load model on
        use_fp8: If True, convert model to FP16 for FP8 acceleration
        
    Returns:
        model: Loaded MASt3R model (in FP16 if use_fp8=True)
    """
    global USE_FP8
    
    if path is None:
        # Find checkpoint relative to this file's location
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(current_dir)
        weights_path = os.path.join(repo_root, "checkpoints", "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth")
    else:
        weights_path = path
    
    model = AsymmetricMASt3R.from_pretrained(weights_path).to(device)
    
    if use_fp8:
        if not FP8_AVAILABLE:
            print("[Warning] FP8 requested but Transformer Engine not available, using FP32")
        else:
            print("[FP8] Converting model to FP16 for FP8 acceleration")
            model = model.half()
            USE_FP8 = True
    
    return model


def load_retriever(mast3r_model, retriever_path=None, device="cuda"):
    retriever_path = (
        "checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth"
        if retriever_path is None
        else retriever_path
    )
    retriever = RetrievalDatabase(retriever_path, backbone=mast3r_model, device=device)
    return retriever


@torch.inference_mode
def decoder(model, feat1, feat2, pos1, pos2, shape1, shape2):
    if USE_FP8 and FP8_AVAILABLE:
        with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
            dec1, dec2 = model._decoder(feat1, pos1, feat2, pos2)
    else:
        dec1, dec2 = model._decoder(feat1, pos1, feat2, pos2)
    
    # Keep in FP16 if model is in FP16, otherwise convert to float
    with torch.amp.autocast(enabled=False, device_type="cuda"):
        if USE_FP8:
            # Keep decoder outputs in FP16 for FP16 model
            res1 = model._downstream_head(1, dec1, shape1)
            res2 = model._downstream_head(2, dec2, shape2)
        else:
            # Convert to float for FP32 model
            res1 = model._downstream_head(1, [tok.float() for tok in dec1], shape1)
            res2 = model._downstream_head(2, [tok.float() for tok in dec2], shape2)
    return res1, res2


def downsample(X, C, D, Q):
    downsample = config["dataset"]["img_downsample"]
    if downsample > 1:
        # C and Q: (...xHxW)
        # X and D: (...xHxWxF)
        X = X[..., ::downsample, ::downsample, :].contiguous()
        C = C[..., ::downsample, ::downsample].contiguous()
        D = D[..., ::downsample, ::downsample, :].contiguous()
        Q = Q[..., ::downsample, ::downsample].contiguous()
    return X, C, D, Q


@torch.inference_mode
def mast3r_symmetric_inference(model, frame_i, frame_j):
    if frame_i.feat is None:
        if USE_FP8 and FP8_AVAILABLE:
            with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                frame_i.feat, frame_i.pos, _ = model._encode_image(
                    frame_i.img, frame_i.img_true_shape
                )
        else:
            frame_i.feat, frame_i.pos, _ = model._encode_image(
                frame_i.img, frame_i.img_true_shape
            )
    if frame_j.feat is None:
        if USE_FP8 and FP8_AVAILABLE:
            with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                frame_j.feat, frame_j.pos, _ = model._encode_image(
                    frame_j.img, frame_j.img_true_shape
                )
        else:
            frame_j.feat, frame_j.pos, _ = model._encode_image(
                frame_j.img, frame_j.img_true_shape
            )

    feat1, feat2 = frame_i.feat, frame_j.feat
    pos1, pos2 = frame_i.pos, frame_j.pos
    
    # Ensure features match model dtype (FP16 if using FP8)
    if USE_FP8 and feat1.dtype != torch.float16:
        feat1 = feat1.half()
    if USE_FP8 and feat2.dtype != torch.float16:
        feat2 = feat2.half()
    
    shape1, shape2 = frame_i.img_true_shape, frame_j.img_true_shape

    res11, res21 = decoder(model, feat1, feat2, pos1, pos2, shape1, shape2)
    res22, res12 = decoder(model, feat2, feat1, pos2, pos1, shape2, shape1)
    res = [res11, res21, res22, res12]
    X, C, D, Q = zip(
        *[(r["pts3d"][0], r["conf"][0], r["desc"][0], r["desc_conf"][0]) for r in res]
    )
    # 4xhxwxc
    X, C, D, Q = torch.stack(X), torch.stack(C), torch.stack(D), torch.stack(Q)
    
    # Convert back to FP32 for SLAM backend compatibility (async for better performance)
    if USE_FP8:
        X = X.to(dtype=torch.float32, non_blocking=True)
        C = C.to(dtype=torch.float32, non_blocking=True)
        D = D.to(dtype=torch.float32, non_blocking=True)
        Q = Q.to(dtype=torch.float32, non_blocking=True)
    
    X, C, D, Q = downsample(X, C, D, Q)
    return X, C, D, Q


# NOTE: Assumes img shape the same
@torch.inference_mode
def mast3r_decode_symmetric_batch(
    model, feat_i, pos_i, feat_j, pos_j, shape_i, shape_j
):
    B = feat_i.shape[0]
    
    X, C, D, Q = [], [], [], []
    for b in range(B):
        feat1 = feat_i[b][None]
        feat2 = feat_j[b][None]
        
        # Ensure features match model dtype (FP16 if using FP8)
        if USE_FP8 and feat1.dtype != torch.float16:
            feat1 = feat1.half()
        if USE_FP8 and feat2.dtype != torch.float16:
            feat2 = feat2.half()
        
        pos1 = pos_i[b][None]
        pos2 = pos_j[b][None]
        res11, res21 = decoder(model, feat1, feat2, pos1, pos2, shape_i[b], shape_j[b])
        res22, res12 = decoder(model, feat2, feat1, pos2, pos1, shape_j[b], shape_i[b])
        res = [res11, res21, res22, res12]
        Xb, Cb, Db, Qb = zip(
            *[
                (r["pts3d"][0], r["conf"][0], r["desc"][0], r["desc_conf"][0])
                for r in res
            ]
        )
        X.append(torch.stack(Xb, dim=0))
        C.append(torch.stack(Cb, dim=0))
        D.append(torch.stack(Db, dim=0))
        Q.append(torch.stack(Qb, dim=0))

    X, C, D, Q = (
        torch.stack(X, dim=1),
        torch.stack(C, dim=1),
        torch.stack(D, dim=1),
        torch.stack(Q, dim=1),
    )
    
    # Convert back to FP32 for SLAM backend compatibility (async for better performance)
    if USE_FP8:
        X = X.to(dtype=torch.float32, non_blocking=True)
        C = C.to(dtype=torch.float32, non_blocking=True)
        D = D.to(dtype=torch.float32, non_blocking=True)
        Q = Q.to(dtype=torch.float32, non_blocking=True)
    
    X, C, D, Q = downsample(X, C, D, Q)
    return X, C, D, Q


@torch.inference_mode
def mast3r_inference_mono(model, frame):
    if frame.feat is None:
        if USE_FP8 and FP8_AVAILABLE:
            with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                frame.feat, frame.pos, _ = model._encode_image(frame.img, frame.img_true_shape)
        else:
            frame.feat, frame.pos, _ = model._encode_image(frame.img, frame.img_true_shape)

    feat = frame.feat
    # Ensure features match model dtype (FP16 if using FP8)
    if USE_FP8 and feat.dtype != torch.float16:
        feat = feat.half()
    
    pos = frame.pos
    shape = frame.img_true_shape

    res11, res21 = decoder(model, feat, feat, pos, pos, shape, shape)
    res = [res11, res21]
    X, C, D, Q = zip(
        *[(r["pts3d"][0], r["conf"][0], r["desc"][0], r["desc_conf"][0]) for r in res]
    )
    # 4xhxwxc
    X, C, D, Q = torch.stack(X), torch.stack(C), torch.stack(D), torch.stack(Q)
    
    # Convert back to FP32 for SLAM backend compatibility (async for better performance)
    if USE_FP8:
        X = X.to(dtype=torch.float32, non_blocking=True)
        C = C.to(dtype=torch.float32, non_blocking=True)
        D = D.to(dtype=torch.float32, non_blocking=True)
        Q = Q.to(dtype=torch.float32, non_blocking=True)
    
    X, C, D, Q = downsample(X, C, D, Q)

    Xii, Xji = einops.rearrange(X, "b h w c -> b (h w) c")
    Cii, Cji = einops.rearrange(C, "b h w -> b (h w) 1")

    return Xii, Cii


def mast3r_match_symmetric(model, feat_i, pos_i, feat_j, pos_j, shape_i, shape_j):
    X, C, D, Q = mast3r_decode_symmetric_batch(
        model, feat_i, pos_i, feat_j, pos_j, shape_i, shape_j
    )

    # Ordering 4xbxhxwxc
    b = X.shape[1]

    Xii, Xji, Xjj, Xij = X[0], X[1], X[2], X[3]
    Dii, Dji, Djj, Dij = D[0], D[1], D[2], D[3]
    Qii, Qji, Qjj, Qij = Q[0], Q[1], Q[2], Q[3]

    # Always matching both
    X11 = torch.cat((Xii, Xjj), dim=0)
    X21 = torch.cat((Xji, Xij), dim=0)
    D11 = torch.cat((Dii, Djj), dim=0)
    D21 = torch.cat((Dji, Dij), dim=0)

    # tic()
    idx_1_to_2, valid_match_2 = matching.match(X11, X21, D11, D21)
    # toc("Match")

    # TODO: Avoid this
    match_b = X11.shape[0] // 2
    idx_i2j = idx_1_to_2[:match_b]
    idx_j2i = idx_1_to_2[match_b:]
    valid_match_j = valid_match_2[:match_b]
    valid_match_i = valid_match_2[match_b:]

    return (
        idx_i2j,
        idx_j2i,
        valid_match_j,
        valid_match_i,
        Qii.view(b, -1, 1),
        Qjj.view(b, -1, 1),
        Qji.view(b, -1, 1),
        Qij.view(b, -1, 1),
    )


@torch.inference_mode
def mast3r_asymmetric_inference(model, frame_i, frame_j):
    if frame_i.feat is None:
        if USE_FP8 and FP8_AVAILABLE:
            with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                frame_i.feat, frame_i.pos, _ = model._encode_image(
                    frame_i.img, frame_i.img_true_shape
                )
        else:
            frame_i.feat, frame_i.pos, _ = model._encode_image(
                frame_i.img, frame_i.img_true_shape
            )
    if frame_j.feat is None:
        if USE_FP8 and FP8_AVAILABLE:
            with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                frame_j.feat, frame_j.pos, _ = model._encode_image(
                    frame_j.img, frame_j.img_true_shape
                )
        else:
            frame_j.feat, frame_j.pos, _ = model._encode_image(
                frame_j.img, frame_j.img_true_shape
            )
    
    feat1, feat2 = frame_i.feat, frame_j.feat
    
    # Ensure features match model dtype (FP16 if using FP8)
    if USE_FP8 and feat1.dtype != torch.float16:
        feat1 = feat1.half()
    if USE_FP8 and feat2.dtype != torch.float16:
        feat2 = feat2.half()
    
    pos1, pos2 = frame_i.pos, frame_j.pos
    shape1, shape2 = frame_i.img_true_shape, frame_j.img_true_shape

    res11, res21 = decoder(model, feat1, feat2, pos1, pos2, shape1, shape2)
    res = [res11, res21]
    X, C, D, Q = zip(
        *[(r["pts3d"][0], r["conf"][0], r["desc"][0], r["desc_conf"][0]) for r in res]
    )
    # 4xhxwxc
    X, C, D, Q = torch.stack(X), torch.stack(C), torch.stack(D), torch.stack(Q)
    
    # Convert back to FP32 for SLAM backend compatibility (async for better performance)
    if USE_FP8:
        X = X.to(dtype=torch.float32, non_blocking=True)
        C = C.to(dtype=torch.float32, non_blocking=True)
        D = D.to(dtype=torch.float32, non_blocking=True)
        Q = Q.to(dtype=torch.float32, non_blocking=True)
    
    X, C, D, Q = downsample(X, C, D, Q)
    return X, C, D, Q


def mast3r_match_asymmetric(model, frame_i, frame_j, idx_i2j_init=None):
    X, C, D, Q = mast3r_asymmetric_inference(model, frame_i, frame_j)

    b, h, w = X.shape[:-1]
    # 2 outputs per inference
    b = b // 2

    Xii, Xji = X[:b], X[b:]
    Cii, Cji = C[:b], C[b:]
    Dii, Dji = D[:b], D[b:]
    Qii, Qji = Q[:b], Q[b:]

    idx_i2j, valid_match_j = matching.match(
        Xii, Xji, Dii, Dji, idx_1_to_2_init=idx_i2j_init
    )

    # How rest of system expects it
    Xii, Xji = einops.rearrange(X, "b h w c -> b (h w) c")
    Cii, Cji = einops.rearrange(C, "b h w -> b (h w) 1")
    Dii, Dji = einops.rearrange(D, "b h w c -> b (h w) c")
    Qii, Qji = einops.rearrange(Q, "b h w -> b (h w) 1")

    return idx_i2j, valid_match_j, Xii, Cii, Qii, Xji, Cji, Qji


def _resize_pil_image(img, long_edge_size):
    S = max(img.size)
    if S > long_edge_size:
        interp = PIL.Image.LANCZOS
    elif S <= long_edge_size:
        interp = PIL.Image.BICUBIC
    new_size = tuple(int(round(x * long_edge_size / S)) for x in img.size)
    return img.resize(new_size, interp)


def resize_img(img, size, square_ok=False, return_transformation=False):
    assert size == 224 or size == 512
    # numpy to PIL format
    img = PIL.Image.fromarray(np.uint8(img * 255))
    W1, H1 = img.size
    if size == 224:
        # resize short side to 224 (then crop)
        img = _resize_pil_image(img, round(size * max(W1 / H1, H1 / W1)))
    else:
        # resize long side to 512
        img = _resize_pil_image(img, size)
    W, H = img.size
    cx, cy = W // 2, H // 2
    if size == 224:
        half = min(cx, cy)
        img = img.crop((cx - half, cy - half, cx + half, cy + half))
    else:
        halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
        if not (square_ok) and W == H:
            halfh = 3 * halfw / 4
        img = img.crop((cx - halfw, cy - halfh, cx + halfw, cy + halfh))

    res = dict(
        img=ImgNorm(img)[None],
        true_shape=np.int32([img.size[::-1]]),
        unnormalized_img=np.asarray(img),
    )
    if return_transformation:
        scale_w = W1 / W
        scale_h = H1 / H
        half_crop_w = (W - img.size[0]) / 2
        half_crop_h = (H - img.size[1]) / 2
        return res, (scale_w, scale_h, half_crop_w, half_crop_h)

    return res


def load_and_encode_image(model, frame):
    feat, pos, _ = model._encode_image(frame.img, frame.img_true_shape)
    frame.feat = feat
    frame.pos = pos
    return frame
