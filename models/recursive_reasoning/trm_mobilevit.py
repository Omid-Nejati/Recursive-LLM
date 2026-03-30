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
    Attention,
    RotaryEmbedding,
    CosSin,
    CastedEmbedding,
    CastedLinear,
)
from models.sparse_embedding import CastedSparseEmbedding


IGNORE_LABEL_ID = -100


@dataclass
class TinyRecursiveReasoningModelMobileViT_ACTV1InnerCarry:
    z_H: torch.Tensor
    z_L: torch.Tensor


@dataclass
class TinyRecursiveReasoningModelMobileViT_ACTV1Carry:
    inner_carry: TinyRecursiveReasoningModelMobileViT_ACTV1InnerCarry
    steps: torch.Tensor
    halted: torch.Tensor
    current_data: Dict[str, torch.Tensor]


class TinyRecursiveReasoningModelMobileViT_ACTV1Config(BaseModel):
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int

    H_cycles: int
    L_cycles: int
    H_layers: int  # unused, kept for config compatibility
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

    # MobileViT-style options
    conv_kernel_size: int = 3
    mobilevit_depth: int = 1
    prefix_mlp: bool = True
    use_global_attention: bool = True


class ConvBNAct(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=pad, groups=dim, bias=False)
        self.pw = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, D, H, W]
        x = self.conv(x)
        x = self.pw(x)
        x = self.act(x)
        return x


class MobileViTTokenBlock(nn.Module):
    """
    Token mixer used inside the MobileViT-style unit.
    """
    def __init__(self, hidden_size: int, num_heads: int, expansion: float, use_global_attention: bool):
        super().__init__()
        self.use_global_attention = use_global_attention
        if use_global_attention:
            self.attn = Attention(
                hidden_size=hidden_size,
                head_dim=hidden_size // num_heads,
                num_heads=num_heads,
                num_key_value_heads=num_heads,
                causal=False,
            )
        else:
            self.attn = None

        self.mlp = SwiGLU(hidden_size=hidden_size, expansion=expansion)

    def forward(self, x: torch.Tensor, cos_sin: Optional[CosSin], norm_eps: float) -> torch.Tensor:
        # x: [B, N, D]
        if self.attn is not None:
            x = rms_norm(x + self.attn(cos_sin=cos_sin, hidden_states=x), variance_epsilon=norm_eps)
        x = rms_norm(x + self.mlp(x), variance_epsilon=norm_eps)
        return x


class MobileViTMixerBlock(nn.Module):
    """
    MobileViT-style TRM block:
      1) local conv on 2D grid tokens
      2) flatten grid tokens and apply a lightweight global token mixer
      3) fuse local + global features with conv
      4) full-sequence MLP
    Prefix puzzle tokens are kept outside the 2D conv path and can be processed by a small MLP.
    """

    def __init__(self, config: TinyRecursiveReasoningModelMobileViT_ACTV1Config):
        super().__init__()
        self.config = config
        self.norm_eps = config.rms_norm_eps

        grid_size = int(math.isqrt(config.seq_len))
        if grid_size * grid_size != config.seq_len:
            raise ValueError(
                f"MobileViT-TRM expects seq_len to be a perfect square, got {config.seq_len}"
            )
        self.grid_size = grid_size

        self.local_rep = ConvBNAct(config.hidden_size, kernel_size=config.conv_kernel_size)

        self.token_blocks = nn.ModuleList(
            [
                MobileViTTokenBlock(
                    hidden_size=config.hidden_size,
                    num_heads=config.num_heads,
                    expansion=config.expansion,
                    use_global_attention=config.use_global_attention,
                )
                for _ in range(config.mobilevit_depth)
            ]
        )

        self.fuse_in = nn.Conv2d(config.hidden_size * 2, config.hidden_size, kernel_size=1, bias=False)
        self.fuse_out = nn.Conv2d(config.hidden_size, config.hidden_size, kernel_size=3, padding=1, bias=False)
        self.fuse_act = nn.GELU()

        self.full_mlp = SwiGLU(
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

    def forward(self, cos_sin: Optional[CosSin], hidden_states: torch.Tensor) -> torch.Tensor:
        B, L_total, D = hidden_states.shape
        prefix_len = L_total - self.config.seq_len

        if prefix_len > 0:
            prefix = hidden_states[:, :prefix_len, :]
            grid_tokens = hidden_states[:, prefix_len:, :]
        else:
            prefix = None
            grid_tokens = hidden_states

        if prefix is not None:
            # Light conditioning from prefix into grid path
            grid_tokens = grid_tokens + prefix.mean(dim=1, keepdim=True)

        H = W = self.grid_size
        x = grid_tokens.view(B, H, W, D).permute(0, 3, 1, 2).contiguous()  # [B, D, H, W]

        residual_grid = grid_tokens
        local_feat = self.local_rep(x)  # [B, D, H, W]

        global_tokens = local_feat.permute(0, 2, 3, 1).contiguous().view(B, H * W, D)
        for blk in self.token_blocks:
            global_tokens = blk(global_tokens, cos_sin=cos_sin, norm_eps=self.norm_eps)

        global_feat = global_tokens.view(B, H, W, D).permute(0, 3, 1, 2).contiguous()  # [B, D, H, W]

        fused = torch.cat([local_feat, global_feat], dim=1)
        fused = self.fuse_in(fused)
        fused = self.fuse_out(fused)
        fused = self.fuse_act(fused)

        fused_tokens = fused.permute(0, 2, 3, 1).contiguous().view(B, H * W, D)
        grid_tokens = self._post_norm(residual_grid, fused_tokens)

        if prefix is not None:
            if self.prefix_mlp is not None:
                prefix = self._post_norm(prefix, self.prefix_mlp(prefix))
            hidden_states = torch.cat([prefix, grid_tokens], dim=1)
        else:
            hidden_states = grid_tokens

        hidden_states = self._post_norm(hidden_states, self.full_mlp(hidden_states))
        return hidden_states


class TinyRecursiveReasoningModelMobileViT_ACTV1ReasoningModule(nn.Module):
    def __init__(self, config: TinyRecursiveReasoningModelMobileViT_ACTV1Config):
        super().__init__()
        self.layers = nn.ModuleList(
            [MobileViTMixerBlock(config) for _ in range(config.L_layers)]
        )

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs) -> torch.Tensor:
        hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)
        return hidden_states


class TinyRecursiveReasoningModelMobileViT_ACTV1_Inner(nn.Module):
    def __init__(self, config: TinyRecursiveReasoningModelMobileViT_ACTV1Config) -> None:
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

        self.L_level = TinyRecursiveReasoningModelMobileViT_ACTV1ReasoningModule(self.config)

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
        return TinyRecursiveReasoningModelMobileViT_ACTV1InnerCarry(
            z_H=torch.empty(batch_size, full_len, self.config.hidden_size, dtype=self.forward_dtype),
            z_L=torch.empty(batch_size, full_len, self.config.hidden_size, dtype=self.forward_dtype),
        )

    def reset_carry(self, reset_flag: torch.Tensor, carry: TinyRecursiveReasoningModelMobileViT_ACTV1InnerCarry):
        return TinyRecursiveReasoningModelMobileViT_ACTV1InnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), self.H_init, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), self.L_init, carry.z_L),
        )

    def forward(
        self,
        carry: TinyRecursiveReasoningModelMobileViT_ACTV1InnerCarry,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[
        TinyRecursiveReasoningModelMobileViT_ACTV1InnerCarry,
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor],
    ]:
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )

        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

        z_H, z_L = carry.z_H, carry.z_L

        with torch.no_grad():
            for _H_step in range(self.config.H_cycles - 1):
                for _L_step in range(self.config.L_cycles):
                    z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
                    z_H = self.L_level(z_H, z_L, **seq_info)

        for _L_step in range(self.config.L_cycles):
            z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
            z_H = self.L_level(z_H, z_L, **seq_info)

        new_carry = TinyRecursiveReasoningModelMobileViT_ACTV1InnerCarry(
            z_H=z_H.detach(),
            z_L=z_L.detach(),
        )

        output = self.lm_head(z_H)[:, self.puzzle_emb_len:]
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)

        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])


class TinyRecursiveReasoningModelMobileViT_ACTV1(nn.Module):
    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = TinyRecursiveReasoningModelMobileViT_ACTV1Config(**config_dict)
        self.inner = TinyRecursiveReasoningModelMobileViT_ACTV1_Inner(self.config)

    @property
    def puzzle_emb(self):
        return getattr(self.inner, "puzzle_emb", None)

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]
        return TinyRecursiveReasoningModelMobileViT_ACTV1Carry(
            inner_carry=self.inner.empty_carry(batch_size),
            steps=torch.zeros((batch_size,), dtype=torch.int32),
            halted=torch.ones((batch_size,), dtype=torch.bool),
            current_data={k: torch.empty_like(v) for k, v in batch.items()},
        )

    def forward(
        self,
        carry: TinyRecursiveReasoningModelMobileViT_ACTV1Carry,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[TinyRecursiveReasoningModelMobileViT_ACTV1Carry, Dict[str, torch.Tensor]]:
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
            TinyRecursiveReasoningModelMobileViT_ACTV1Carry(
                new_inner_carry, new_steps, halted, new_current_data
            ),
            outputs,
        )