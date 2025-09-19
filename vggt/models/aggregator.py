# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple, Union, List, Dict, Any

from vggt.layers import PatchEmbed
from vggt.layers.block import Block
from vggt.layers.rope import RotaryPositionEmbedding2D, PositionGetter
from vggt.layers.vision_transformer import vit_small, vit_base, vit_large, vit_giant2

logger = logging.getLogger(__name__)

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


class Aggregator(nn.Module):
    """
    The Aggregator applies alternating-attention over input frames,
    as described in VGGT: Visual Geometry Grounded Transformer.

    Remember to set model.train() to enable gradient checkpointing to reduce memory usage.

    Args:
        img_size (int): Image size in pixels.
        patch_size (int): Size of each patch for PatchEmbed.
        embed_dim (int): Dimension of the token embeddings.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of MLP hidden dim to embedding dim.
        num_register_tokens (int): Number of register tokens.
        block_fn (nn.Module): The block type used for attention (Block by default).
        qkv_bias (bool): Whether to include bias in QKV projections.
        proj_bias (bool): Whether to include bias in the output projection.
        ffn_bias (bool): Whether to include bias in MLP layers.
        patch_embed (str): Type of patch embed. e.g., "conv" or "dinov2_vitl14_reg".
        aa_order (list[str]): The order of alternating attention, e.g. ["frame", "global"].
        aa_block_size (int): How many blocks to group under each attention type before switching. If not necessary, set to 1.
        qk_norm (bool): Whether to apply QK normalization.
        rope_freq (int): Base frequency for rotary embedding. -1 to disable.
        init_values (float): Init scale for layer scale.
    """

    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=4,
        block_fn=Block,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        patch_embed="dinov2_vitl14_reg",
        aa_order=["frame", "global"],
        aa_block_size=1,
        qk_norm=True,
        rope_freq=100,
        init_values=0.01,
        capture_attn: bool = True,
        capture_which: tuple = ('frame','global'),
        capture_queries: str = 'cam+reg',   # 'cam' | 'cam+reg' | 'all' | 'indices'
        capture_query_indices: Optional[List[int]] = None,
        head_reduce: str = 'mean',  # 'mean' | 'none'
    ):
        super().__init__()

        self.__build_patch_embed__(patch_embed, img_size, patch_size, num_register_tokens, embed_dim=embed_dim)

        # Initialize rotary position embedding if frequency > 0
        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        self.position_getter = PositionGetter() if self.rope is not None else None

        self.capture_attn = capture_attn 
        self.capture_which = capture_which
        self.capture_queries = capture_queries
        self.capture_query_indices = capture_query_indices
        self.head_reduce = head_reduce
        self.attn_records = {'frame': [], 'global': []} # 缓存注意力（按层追加）
        self.capture_block_indices = {
            'frame':  None,        # e.g. {0, 3, 7}
            'global': None         # e.g. {0, 5, 11}
        }
                
        
        self.frame_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.global_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.depth = depth
        self.aa_order = aa_order
        self.patch_size = patch_size
        self.aa_block_size = aa_block_size

        # Validate that depth is divisible by aa_block_size
        if self.depth % self.aa_block_size != 0:
            raise ValueError(f"depth ({depth}) must be divisible by aa_block_size ({aa_block_size})")

        self.aa_block_num = self.depth // self.aa_block_size

        # Note: We have two camera tokens, one for the first frame and one for the rest
        # The same applies for register tokens
        self.camera_token = nn.Parameter(torch.randn(1, 2, 1, embed_dim))
        self.register_token = nn.Parameter(torch.randn(1, 2, num_register_tokens, embed_dim))

        # The patch tokens start after the camera and register tokens
        self.patch_start_idx = 1 + num_register_tokens

        # Initialize parameters with small values
        nn.init.normal_(self.camera_token, std=1e-6)
        nn.init.normal_(self.register_token, std=1e-6)

        # Register normalization constants as buffers
        for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
            self.register_buffer(name, torch.FloatTensor(value).view(1, 1, 3, 1, 1), persistent=False)

        self.use_reentrant = False # hardcoded to False

    def __build_patch_embed__(
        self,
        patch_embed,
        img_size,
        patch_size,
        num_register_tokens,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        block_chunks=0,
        init_values=1.0,
        embed_dim=1024,
    ):
        """
        Build the patch embed layer. If 'conv', we use a
        simple PatchEmbed conv layer. Otherwise, we use a vision transformer.
        """

        if "conv" in patch_embed:
            self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim)
        else:
            vit_models = {
                "dinov2_vitl14_reg": vit_large,
                "dinov2_vitb14_reg": vit_base,
                "dinov2_vits14_reg": vit_small,
                "dinov2_vitg2_reg": vit_giant2,
            }

            self.patch_embed = vit_models[patch_embed](
                img_size=img_size,
                patch_size=patch_size,
                num_register_tokens=num_register_tokens,
                interpolate_antialias=interpolate_antialias,
                interpolate_offset=interpolate_offset,
                block_chunks=block_chunks,
                init_values=init_values,
            )

            # Disable gradient updates for mask token
            if hasattr(self.patch_embed, "mask_token"):
                self.patch_embed.mask_token.requires_grad_(False)

    def forward(self, images: torch.Tensor) -> Tuple[List[torch.Tensor], int]:
        """
        Args:
            images (torch.Tensor): Input images with shape [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width

        Returns:
            (list[torch.Tensor], int):
                The list of outputs from the attention blocks,
                and the patch_start_idx indicating where patch tokens begin.
        """
        B, S, C_in, H, W = images.shape

        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")

        # Normalize images and reshape for patch embed
        images = (images - self._resnet_mean) / self._resnet_std

        # Reshape to [B*S, C, H, W] for patch embedding
        images = images.view(B * S, C_in, H, W)
        patch_tokens = self.patch_embed(images)

        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        _, P, C = patch_tokens.shape

        # Expand camera and register tokens to match batch size and sequence length
        camera_token = slice_expand_and_flatten(self.camera_token, B, S)
        register_token = slice_expand_and_flatten(self.register_token, B, S)

        # Concatenate special tokens with patch tokens
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)

        pos = None
        if self.rope is not None:
            pos = self.position_getter(B * S, H // self.patch_size, W // self.patch_size, device=images.device)

        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = torch.zeros(B * S, self.patch_start_idx, 2).to(images.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)

        # update P because we added special tokens
        _, P, C = tokens.shape

        frame_idx = 0
        global_idx = 0
        output_list = []

        if hasattr(self, 'attn_records') and self.capture_attn:
            self.attn_records['frame'].clear()
            self.attn_records['global'].clear()
        for _ in range(self.aa_block_num):
            for attn_type in self.aa_order:
                if attn_type == "frame":
                    tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                        tokens, B, S, P, C, frame_idx, pos=pos
                    ) 
                    # frame_idx is frame attention block index
                elif attn_type == "global":
                    tokens, global_idx, global_intermediates = self._process_global_attention(
                        tokens, B, S, P, C, global_idx, pos=pos
                    )
                    # global_idx is global attention block index
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")

            for i in range(len(frame_intermediates)):
                # concat frame and global intermediates, [B x S x P x 2C]
                concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)
                output_list.append(concat_inter)

        del concat_inter
        del frame_intermediates
        del global_intermediates
        return output_list, self.patch_start_idx

    # def _process_frame_attention(self, tokens, B, S, P, C, frame_idx, pos=None):
    #     """
    #     Process frame attention blocks. We keep tokens in shape (B*S, P, C).
    #     """
    #     # If needed, reshape tokens or positions:
    #     if tokens.shape != (B * S, P, C):
    #         tokens = tokens.view(B, S, P, C).view(B * S, P, C)

    #     if pos is not None and pos.shape != (B * S, P, 2):
    #         pos = pos.view(B, S, P, 2).view(B * S, P, 2)

    #     intermediates = []

    #     # by default, self.aa_block_size=1, which processes one block at a time
    #     for _ in range(self.aa_block_size):
    #         if self.training:
    #             tokens = checkpoint(self.frame_blocks[frame_idx], tokens, pos, use_reentrant=self.use_reentrant)
    #         else:
    #             tokens = self.frame_blocks[frame_idx](tokens, pos=pos)
    #         frame_idx += 1
    #         intermediates.append(tokens.view(B, S, P, C))

    #     return tokens, frame_idx, intermediates
    
    def _should_capture(self, kind: str, idx: int) -> bool:
        # 索引优先
        idx_set = self.capture_block_indices.get(kind)
        if idx_set is not None:
            return idx in idx_set
        # # 其次按周期
        # k = self.capture_block_every.get(kind)
        # if k is not None and k > 0:
        #     return (idx % k) == 0
        # 默认：若 capture_which 里包含此类，则全部抓
        return (kind in self.capture_which)


    def _process_frame_attention(self, tokens, B, S, P, C, frame_idx, pos=None):
        if tokens.shape != (B * S, P, C):
            tokens = tokens.view(B, S, P, C).view(B * S, P, C)
        if pos is not None and pos.shape != (B * S, P, 2):
            pos = pos.view(B, S, P, 2).view(B * S, P, 2)

        intermediates = []
        K_regs = self.register_token.shape[2]          # 每帧寄存器个数
        patch_start_idx = self.patch_start_idx         # = 1 + K

        for _ in range(self.aa_block_size):
            block = self.frame_blocks[frame_idx]

            if self.capture_attn and self._should_capture('frame', frame_idx) and ('frame' in self.capture_which):
                self._install_attn_hook_once(
                    block=self.frame_blocks[frame_idx],
                    kind='frame',
                    idx=frame_idx,
                    pos_tensor=pos,
                    B=B, S=S, P=P,
                    K_regs=self.register_token.shape[2],
                    patch_start_idx=self.patch_start_idx,
                )

            tokens = checkpoint(block, tokens, pos, use_reentrant=self.use_reentrant) if self.training \
                    else block(tokens, pos=pos)
            frame_idx += 1
            intermediates.append(tokens.view(B, S, P, C))
        return tokens, frame_idx, intermediates


    # def _process_global_attention(self, tokens, B, S, P, C, global_idx, pos=None):
    #     """
    #     Process global attention blocks. We keep tokens in shape (B, S*P, C).
    #     """
    #     if tokens.shape != (B, S * P, C):
    #         tokens = tokens.view(B, S, P, C).view(B, S * P, C)

    #     if pos is not None and pos.shape != (B, S * P, 2):
    #         pos = pos.view(B, S, P, 2).view(B, S * P, 2)

    #     intermediates = []

    #     # by default, self.aa_block_size=1, which processes one block at a time
    #     for _ in range(self.aa_block_size):
    #         if self.training:
    #             tokens = checkpoint(self.global_blocks[global_idx], tokens, pos, use_reentrant=self.use_reentrant)
    #         else:
    #             tokens = self.global_blocks[global_idx](tokens, pos=pos)
    #         global_idx += 1
    #         intermediates.append(tokens.view(B, S, P, C))

    #     return tokens, global_idx, intermediates
    
    def _process_global_attention(self, tokens, B, S, P, C, global_idx, pos=None):
        if tokens.shape != (B, S * P, C):
            tokens = tokens.view(B, S, P, C).view(B, S * P, C)
        if pos is not None and pos.shape != (B, S * P, 2):
            pos = pos.view(B, S, P, 2).view(B, S * P, 2)

        intermediates = []
        K_regs = self.register_token.shape[2]
        patch_start_idx = self.patch_start_idx

        for _ in range(self.aa_block_size):
            block = self.global_blocks[global_idx]

            if self.capture_attn and self._should_capture('global', global_idx) and ('global' in self.capture_which):
                self._install_attn_hook_once(
                    block=self.global_blocks[global_idx],
                    kind='global',
                    idx=global_idx,
                    pos_tensor=pos,
                    B=B, S=S, P=P,
                    K_regs=self.register_token.shape[2],
                    patch_start_idx=self.patch_start_idx,
                )

            tokens = checkpoint(block, tokens, pos, use_reentrant=self.use_reentrant) if self.training \
                    else block(tokens, pos=pos)
            global_idx += 1
            intermediates.append(tokens.view(B, S, P, C))
        return tokens, global_idx, intermediates
    
    def _install_attn_hook_once(
        self,
        block,
        kind: str,                   # 'frame' or 'global'
        idx: int,                    # block index (用于调试)
        pos_tensor: Optional[torch.Tensor],
        B: int, S: int, P: int,      # B: 批，S: 帧数，P: 每帧 token 数(=1+K+H_p*W_p)
        K_regs: int,
        patch_start_idx: int,
    ):
        """
        在 block.attn 上注册一次性 forward hook：
        - 自动区分 frame-wise (B*S,P,C) 与 global (B,S*P,C)
        - 只抽取需要的 query 行（cam / cam+reg / all / indices）
        - 复算 attn = softmax((QK^T)/sqrt(d))，可按 head 求均值
        结果追加到 self.attn_records[kind] 列表。
        """
        handle_box = {'h': None}

        # 读取配置（若 Aggregator 未设置这些属性，则给默认）
        capture_queries = getattr(self, 'capture_queries', 'cam+reg')          # 'cam' | 'cam+reg' | 'all' | 'indices'
        capture_query_indices = getattr(self, 'capture_query_indices', None)    # 仅当 'indices' 时生效
        head_reduce = getattr(self, 'head_reduce', 'mean')                      # 'mean' | 'none'

        def hook(mod, inp, out):
            # inp = (x, pos)
            x = inp[0]                                   # (B*S, P, C) 或 (B, S*P, C)
            pos = inp[1] if len(inp) > 1 else None       # 与 forward 一致的 RoPE 位置 (同形状)
            Bcur, N, C = x.shape
            Hh, Dh = mod.num_heads, mod.head_dim

            # === 与 Attention.forward 对齐的投影 / 归一化 / RoPE ===
            qkv = mod.qkv(x).reshape(Bcur, N, 3, Hh, Dh).permute(2, 0, 3, 1, 4)
            q, k, _ = qkv.unbind(0)
            q, k = mod.q_norm(q), mod.k_norm(k)
            if getattr(mod, 'rope', None) is not None and pos is not None:
                q = mod.rope(q, pos)
                k = mod.rope(k, pos)

            # === 判定当前是 frame-wise 还是 global 路径 ===
            # frame-wise: (B*S, P, C), N == P
            # global    : (B, S*P, C),   N == S*P
            is_frame = (Bcur == B * S) and (N == P)

            # === 构造查询行索引（避免越界） ===
            q_indices: List[int] = []
            if is_frame:
                # 本层序列只有一帧，索引范围 [0, P)
                if capture_queries in ('cam', 'cam+reg', 'all', 'indices'):
                    cam = 0
                    regs = list(range(1, 1 + K_regs))
                    if capture_queries == 'cam':
                        q_indices.append(cam)
                    elif capture_queries == 'cam+reg':
                        q_indices.append(cam); q_indices.extend(regs)
                    elif capture_queries == 'all':
                        q_indices.extend(range(0, P))
                    elif capture_queries == 'indices':
                        if capture_query_indices is None:
                            raise ValueError("capture_query_indices is None but capture_queries='indices'")
                        q_indices = list(capture_query_indices)  # 必须在 [0, P) 内
            else:
                # global：序列包含 S 帧拼接，索引范围 [0, S*P)
                if capture_queries in ('cam', 'cam+reg', 'all', 'indices'):
                    if capture_queries == 'indices':
                        if capture_query_indices is None:
                            raise ValueError("capture_query_indices is None but capture_queries='indices'")
                        q_indices = list(capture_query_indices)  # 允许 [0, S*P)
                    else:
                        for f in range(S):
                            base = f * P
                            cam = base + 0
                            regs = list(range(base + 1, base + 1 + K_regs))
                            if capture_queries == 'cam':
                                q_indices.append(cam)
                            elif capture_queries == 'cam+reg':
                                q_indices.append(cam); q_indices.extend(regs)
                            elif capture_queries == 'all':
                                q_indices.extend(range(base, base + P))

            # 防御性检查
            if len(q_indices) == 0:
                raise RuntimeError(f"[{kind}] no query indices selected (is_frame={is_frame}, N={N}, P={P}, S={S})")
            max_q = max(q_indices)
            if max_q >= N or min(q_indices) < 0:
                raise RuntimeError(f"[{kind}] query index out of bounds: max={max_q}, N={N}, P={P}, S={S}, is_frame={is_frame}")

            # === 只算所需 query 行的注意力（省显存） ===
            scale = Dh ** -0.5
            q_sel = q[:, :, q_indices, :]                          # (Bcur, Hh, lenQ, Dh)
            attn_logits = (q_sel * scale) @ k.transpose(-2, -1)    # (Bcur, Hh, lenQ, N)
            attn = attn_logits.softmax(dim=-1)

            if head_reduce == 'mean':
                attn = attn.mean(dim=1)                            # (Bcur, lenQ, N)

            # 记录下来（CPU）
            # self.attn_records[kind].append(attn.detach().cpu())
            self.attn_records[kind].append({
                'attn': attn.detach().cpu(),     # (Bcur, lenQ, N) 或 (Bcur,H,lenQ,N) → head_reduce 后建议是 (Bcur, lenQ, N)
                'q_indices': q_indices,          # 选到的绝对行索引（frame: [0,P)，global: [0,S*P)）
                'is_frame': is_frame,            # True=frame-wise, False=global
                'P': P, 'S': S, 'K': K_regs, 'patch_start_idx': patch_start_idx,
                'block_idx': idx   # ← 新增：这是第几个 block（frame_idx 或 global_idx）
            })

            # 一次性 hook：触发一次后卸载
            if handle_box['h'] is not None:
                handle_box['h'].remove()
                handle_box['h'] = None

        attn_mod = getattr(block, 'attn', None)
        if attn_mod is None:
            raise AttributeError("Block 内未找到 'attn' 子模块")
        handle_box['h'] = attn_mod.register_forward_hook(hook)





def slice_expand_and_flatten(token_tensor, B, S):
    """
    Processes specialized tokens with shape (1, 2, X, C) for multi-frame processing:
    1) Uses the first position (index=0) for the first frame only
    2) Uses the second position (index=1) for all remaining frames (S-1 frames)
    3) Expands both to match batch size B
    4) Concatenates to form (B, S, X, C) where each sequence has 1 first-position token
       followed by (S-1) second-position tokens
    5) Flattens to (B*S, X, C) for processing

    Returns:
        torch.Tensor: Processed tokens with shape (B*S, X, C)
    """

    # Slice out the "query" tokens => shape (1, 1, ...)
    query = token_tensor[:, 0:1, ...].expand(B, 1, *token_tensor.shape[2:])
    # Slice out the "other" tokens => shape (1, S-1, ...)
    others = token_tensor[:, 1:, ...].expand(B, S - 1, *token_tensor.shape[2:])
    # Concatenate => shape (B, S, ...)
    combined = torch.cat([query, others], dim=1)

    # Finally flatten => shape (B*S, ...)
    combined = combined.view(B * S, *combined.shape[2:])
    return combined
