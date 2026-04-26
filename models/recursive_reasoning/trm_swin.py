from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel

from models.common import trunc_normal_init_
from models.layers import (
    rms_norm,
    SwiGLU,
    RotaryEmbedding,
    CosSin,
    CastedEmbedding,
    CastedLinear,
)
from models.sparse_embedding import CastedSparseEmbedding


IGNORE_LABEL_ID = -100


@dataclass
class TinyRecursiveReasoningModelSwin_ACTV1InnerCarry:
    z_H: torch.Tensor
    z_L: torch.Tensor


@dataclass
class TinyRecursiveReasoningModelSwin_ACTV1Carry:
    inner_carry: TinyRecursiveReasoningModelSwin_ACTV1InnerCarry
    steps: torch.Tensor
    halted: torch.Tensor
    current_data: Dict[str, torch.Tensor]


class TinyRecursiveReasoningModelSwin_ACTV1Config(BaseModel):
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int

    H_cycles: int
    L_cycles: int
    H_layers: int  # ignored, kept for config compatibility
    L_layers: int

    hidden_size: int
    expansion: float
    num_heads: int

    pos_encodings: str
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    halt_max_steps: int
    halt_exploration_prob: float
    forward_dtype: str = "bfloat16"

    puzzle_emb_len: int = 16
    no_ACT_continue: bool = True

    # Swin-specific
    window_size: int = 3
    shift_size: int = 1
    prefix_mlp: bool = True


def _window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    x: [B, H, W, D]
    returns: [B * num_windows, window_size * window_size, D]
    """
    B, H, W, D = x.shape
    x = x.view(
        B,
        H // window_size,
        window_size,
        W // window_size,
        window_size,
        D,
    )
    windows = (
        x.permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .view(-1, window_size * window_size, D)
    )
    return windows


def _window_reverse(
    windows: torch.Tensor,
    window_size: int,
    H: int,
    W: int,
    B: int,
) -> torch.Tensor:
    """
    windows: [B * num_windows, window_size * window_size, D]
    returns: [B, H, W, D]
    """
    D = windows.shape[-1]
    x = windows.view(
        B,
        H // window_size,
        W // window_size,
        window_size,
        window_size,
        D,
    )
    x = (
        x.permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .view(B, H, W, D)
    )
    return x


class WindowAttention(nn.Module):
    def __init__(self, dim: int, window_size: int, num_heads: int):
        super().__init__()
        assert dim % num_heads == 0, "hidden_size must be divisible by num_heads"
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

        # Relative position bias
        num_rel = (2 * window_size - 1) * (2 * window_size - 1)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(num_rel, num_heads)
        )

        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # [2, Wh, Ww]
        coords_flatten = torch.flatten(coords, 1)  # [2, Wh*Ww]
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [2, N, N]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [N, N, 2]
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)  # [N, N]
        self.register_buffer("relative_position_index", relative_position_index, persistent=False)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: [B*nW, N, D]
        mask: [nW, N, N] or None
        """
        B_, N, D = x.shape
        residual_dtype = x.dtype

        qkv_dtype = self.qkv.weight.dtype
        x_proj = x.to(qkv_dtype)
        qkv = (
            self.qkv(x_proj)
            .reshape(B_, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )  # [3, B_, heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # [B_, heads, N, N]

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(N, N, self.num_heads)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # [heads, N, N]
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(0).unsqueeze(2).to(attn.dtype)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B_, N, D)
        out = self.proj(out)
        return out.to(residual_dtype)


class SwinMixerBlock(nn.Module):
    """
    TRM-compatible Swin-style block.
    - Applies shifted-window attention only on the square puzzle tokens.
    - Keeps prefix puzzle-embedding tokens outside the window partition.
    - Then applies a full-sequence SwiGLU MLP like TRM.
    """

    def __init__(
        self,
        config: TinyRecursiveReasoningModelSwin_ACTV1Config,
        shift_size: int,
    ) -> None:
        super().__init__()
        self.config = config
        self.norm_eps = config.rms_norm_eps
        self.window_size = config.window_size
        self.shift_size = shift_size

        grid_size = int(math.isqrt(config.seq_len))
        if grid_size * grid_size != config.seq_len:
            raise ValueError(
                f"Swin-TRM expects seq_len to be a perfect square, got {config.seq_len}"
            )
        self.grid_size = grid_size

        if self.shift_size >= self.window_size:
            raise ValueError("shift_size must be < window_size")

        self.window_attn = WindowAttention(
            dim=config.hidden_size,
            window_size=config.window_size,
            num_heads=config.num_heads,
        )

        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )

        self.prefix_mlp = None
        if config.prefix_mlp:
            self.prefix_mlp = SwiGLU(
                hidden_size=config.hidden_size,
                expansion=config.expansion,
            )

    def _post_norm(self, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        return rms_norm(x + out, variance_epsilon=self.norm_eps)

    def _make_attn_mask(
        self,
        Hp: int,
        Wp: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if self.shift_size == 0:
            return None

        img_mask = torch.zeros((1, Hp, Wp, 1), device=device)
        cnt = 0

        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )

        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = _window_partition(img_mask, self.window_size)  # [nW, ws*ws, 1]
        mask_windows = mask_windows.squeeze(-1)  # [nW, N]
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # [nW, N, N]
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, 0.0)
        return attn_mask

    def forward(self, cos_sin: Optional[CosSin], hidden_states: torch.Tensor) -> torch.Tensor:
        del cos_sin  # Swin block ignores RoPE; use pos_encodings=none or learned.

        B, L_total, D = hidden_states.shape
        prefix_len = L_total - self.config.seq_len

        if prefix_len > 0:
            prefix = hidden_states[:, :prefix_len, :]
            grid_tokens = hidden_states[:, prefix_len:, :]
        else:
            prefix = None
            grid_tokens = hidden_states

        # Let prefix tokens condition the grid
        if prefix is not None:
            grid_tokens = grid_tokens + prefix.mean(dim=1, keepdim=True)

        H = W = self.grid_size
        x = grid_tokens.view(B, H, W, D)

        # Pad so H and W are divisible by window_size
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        Hp, Wp = x.shape[1], x.shape[2]

        attn_mask = self._make_attn_mask(Hp, Wp, x.device)

        residual_grid = x

        # Shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        # Window attention
        x_windows = _window_partition(x, self.window_size)  # [B*nW, ws*ws, D]
        attn_windows = self.window_attn(x_windows, mask=attn_mask)
        x = _window_reverse(attn_windows, self.window_size, Hp, Wp, B)

        # Reverse shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        # Remove padding
        if pad_h > 0 or pad_w > 0:
            x = x[:, :H, :W, :]

        x = self._post_norm(residual_grid[:, :H, :W, :].reshape(B, H * W, D), x.reshape(B, H * W, D))

        if prefix is not None:
            if self.prefix_mlp is not None:
                prefix = self._post_norm(prefix, self.prefix_mlp(prefix))
            hidden_states = torch.cat([prefix, x], dim=1)
        else:
            hidden_states = x

        # Full-sequence MLP, same spirit as original TRM block
        hidden_states = self._post_norm(hidden_states, self.mlp(hidden_states))
        return hidden_states


class TinyRecursiveReasoningModelSwin_ACTV1ReasoningModule(nn.Module):
    def __init__(self, config: TinyRecursiveReasoningModelSwin_ACTV1Config):
        super().__init__()
        layers = []
        for i in range(config.L_layers):
            shift = 0 if (i % 2 == 0) else config.shift_size
            layers.append(SwinMixerBlock(config, shift_size=shift))
        self.layers = nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs) -> torch.Tensor:
        hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)
        return hidden_states


class TinyRecursiveReasoningModelSwin_ACTV1_Inner(nn.Module):
    def __init__(self, config: TinyRecursiveReasoningModelSwin_ACTV1Config) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
            init_std=embed_init_std,
            cast_to=self.forward_dtype,
        )
        self.lm_head = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.q_head = CastedLinear(self.config.hidden_size, 2, bias=True)

        self.puzzle_emb_len = (
            -(self.config.puzzle_emb_ndim // -self.config.hidden_size)
            if self.config.puzzle_emb_len == 0
            else self.config.puzzle_emb_len
        )

        if self.config.puzzle_emb_ndim > 0:
            self.puzzle_emb = CastedSparseEmbedding(
                self.config.num_puzzle_identifiers,
                self.config.puzzle_emb_ndim,
                batch_size=self.config.batch_size,
                init_std=0,
                cast_to=self.forward_dtype,
            )

        # Keep same config knobs as TRM
        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(
                dim=self.config.hidden_size // self.config.num_heads,
                max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
                base=self.config.rope_theta,
            )
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(
                self.config.seq_len + self.puzzle_emb_len,
                self.config.hidden_size,
                init_std=embed_init_std,
                cast_to=self.forward_dtype,
            )

        self.L_level = TinyRecursiveReasoningModelSwin_ACTV1ReasoningModule(self.config)

        self.register_buffer(
            "H_init",
            trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1),
            persistent=True,
        )
        self.register_buffer(
            "L_init",
            trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1),
            persistent=True,
        )

        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
        embedding = self.embed_tokens(input.to(torch.int32))

        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)
            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))
            embedding = torch.cat(
                (
                    puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size),
                    embedding,
                ),
                dim=-2,
            )

        if self.config.pos_encodings == "learned":
            embedding = 0.707106781 * (
                embedding + self.embed_pos.embedding_weight.to(self.forward_dtype)
            )

        return self.embed_scale * embedding

    def empty_carry(self, batch_size: int):
        full_len = self.config.seq_len + self.puzzle_emb_len
        return TinyRecursiveReasoningModelSwin_ACTV1InnerCarry(
            z_H=torch.empty(batch_size, full_len, self.config.hidden_size, dtype=self.forward_dtype),
            z_L=torch.empty(batch_size, full_len, self.config.hidden_size, dtype=self.forward_dtype),
        )

    def reset_carry(self, reset_flag: torch.Tensor, carry: TinyRecursiveReasoningModelSwin_ACTV1InnerCarry):
        return TinyRecursiveReasoningModelSwin_ACTV1InnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), self.H_init, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), self.L_init, carry.z_L),
        )

    def forward(
        self,
        carry: TinyRecursiveReasoningModelSwin_ACTV1InnerCarry,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[
        TinyRecursiveReasoningModelSwin_ACTV1InnerCarry,
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor],
    ]:
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )

        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

        z_H, z_L = carry.z_H, carry.z_L

        # Same recursive schedule as stock TRM
        with torch.no_grad():
            for _H_step in range(self.config.H_cycles - 1):
                for _L_step in range(self.config.L_cycles):
                    z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
                    z_H = self.L_level(z_H, z_L, **seq_info)

        for _L_step in range(self.config.L_cycles):
            z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
            z_H = self.L_level(z_H, z_L, **seq_info)

        new_carry = TinyRecursiveReasoningModelSwin_ACTV1InnerCarry(
            z_H=z_H.detach(),
            z_L=z_L.detach(),
        )

        output = self.lm_head(z_H)[:, self.puzzle_emb_len:]
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)

        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])


class TinyRecursiveReasoningModelSwin_ACTV1(nn.Module):
    """
    ACT wrapper compatible with pretrain.py
    """

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = TinyRecursiveReasoningModelSwin_ACTV1Config(**config_dict)
        self.inner = TinyRecursiveReasoningModelSwin_ACTV1_Inner(self.config)

    @property
    def puzzle_emb(self):
        return getattr(self.inner, "puzzle_emb", None)

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]
        return TinyRecursiveReasoningModelSwin_ACTV1Carry(
            inner_carry=self.inner.empty_carry(batch_size),
            steps=torch.zeros((batch_size,), dtype=torch.int32),
            halted=torch.ones((batch_size,), dtype=torch.bool),
            current_data={k: torch.empty_like(v) for k, v in batch.items()},
        )

    def forward(
        self,
        carry: TinyRecursiveReasoningModelSwin_ACTV1Carry,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[TinyRecursiveReasoningModelSwin_ACTV1Carry, Dict[str, torch.Tensor]]:
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        new_steps = torch.where(carry.halted, 0, carry.steps)
        new_current_data = {
            k: torch.where(
                carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)),
                batch[k],
                v,
            )
            for k, v in carry.current_data.items()
        }

        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(
            new_inner_carry, new_current_data
        )

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits,
        }

        with torch.no_grad():
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps
            halted = is_last_step

            if self.training and (self.config.halt_max_steps > 1):
                if self.config.no_ACT_continue:
                    halted = halted | (q_halt_logits > 0)
                else:
                    halted = halted | (q_halt_logits > q_continue_logits)

                min_halt_steps = (
                    (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob)
                    * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)
                )
                halted = halted & (new_steps >= min_halt_steps)

                if not self.config.no_ACT_continue:
                    _, _, (next_q_halt_logits, next_q_continue_logits) = self.inner(
                        new_inner_carry, new_current_data
                    )
                    outputs["target_q_continue"] = torch.sigmoid(
                        torch.where(
                            is_last_step,
                            next_q_halt_logits,
                            torch.maximum(next_q_halt_logits, next_q_continue_logits),
                        )
                    )

        return (
            TinyRecursiveReasoningModelSwin_ACTV1Carry(
                new_inner_carry, new_steps, halted, new_current_data
            ),
            outputs,
        )
