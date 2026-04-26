from typing import Tuple, Dict, Optional
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

try:
    from mamba_ssm import Mamba2
except ImportError as e:
    raise ImportError(
        "trm_mambavision requires `mamba-ssm` and `causal-conv1d`.\n"
        "Install with:\n"
        "  pip install causal-conv1d\n"
        "  pip install --no-build-isolation mamba-ssm\n"
    ) from e


IGNORE_LABEL_ID = -100


@dataclass
class TinyRecursiveReasoningModelMambaVision_ACTV1InnerCarry:
    z_H: torch.Tensor
    z_L: torch.Tensor


@dataclass
class TinyRecursiveReasoningModelMambaVision_ACTV1Carry:
    inner_carry: TinyRecursiveReasoningModelMambaVision_ACTV1InnerCarry
    steps: torch.Tensor
    halted: torch.Tensor
    current_data: Dict[str, torch.Tensor]


class TinyRecursiveReasoningModelMambaVision_ACTV1Config(BaseModel):
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int

    H_cycles: int
    L_cycles: int
    H_layers: int
    L_layers: int

    hidden_size: int
    expansion: float
    num_heads: int

    pos_encodings: str
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    halt_max_steps: int
    halt_exploration_prob: float
    forward_dtype: str = "float32"

    puzzle_emb_len: int = 16
    no_ACT_continue: bool = True

    conv_kernel_size: int = 3
    use_local_cpe: bool = True

    mamba_blocks_per_layer: int = 1
    attn_last_k: int = 1

    mamba_d_state: int = 64
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    mamba_headdim: int = 64
    mamba_ngroups: int = 1
    use_mem_eff_path: bool = False


class LocalCPE(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.dw = nn.Conv2d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=padding,
            groups=dim,
            bias=False,
        )
        self.pw = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.pw(self.dw(x)))


class MambaVisionStyleBlock(nn.Module):
    def __init__(
        self,
        config: TinyRecursiveReasoningModelMambaVision_ACTV1Config,
        use_attention: bool,
    ) -> None:
        super().__init__()
        self.config = config
        self.use_attention = use_attention
        self.norm_eps = config.rms_norm_eps
        self.forward_dtype = getattr(torch, config.forward_dtype)

        grid_size = int(math.isqrt(config.seq_len))
        if grid_size * grid_size != config.seq_len:
            raise ValueError(
                f"MambaVision-TRM expects seq_len to be a perfect square, got {config.seq_len}"
            )
        self.grid_size = grid_size

        self.local_cpe = None
        if config.use_local_cpe:
            self.local_cpe = LocalCPE(
                dim=config.hidden_size,
                kernel_size=config.conv_kernel_size,
            )

        self.mamba_blocks = nn.ModuleList(
            [
                Mamba2(
                    d_model=config.hidden_size,
                    d_state=config.mamba_d_state,
                    d_conv=config.mamba_d_conv,
                    expand=config.mamba_expand,
                    headdim=config.mamba_headdim,
                    ngroups=config.mamba_ngroups,
                    use_mem_eff_path=config.use_mem_eff_path,
                ).to(dtype=self.forward_dtype)
                for _ in range(config.mamba_blocks_per_layer)
            ]
        )

        self.self_attn = None
        if use_attention:
            self.self_attn = Attention(
                hidden_size=config.hidden_size,
                head_dim=config.hidden_size // config.num_heads,
                num_heads=config.num_heads,
                num_key_value_heads=config.num_heads,
                causal=False,
            )

        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )

    def _post_norm(self, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        return rms_norm(x + out, variance_epsilon=self.norm_eps)

    def _apply_local_cpe(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.local_cpe is None:
            return hidden_states

        B, L_total, D = hidden_states.shape
        prefix_len = L_total - self.config.seq_len

        if prefix_len > 0:
            prefix = hidden_states[:, :prefix_len, :]
            grid_tokens = hidden_states[:, prefix_len:, :]
        else:
            prefix = None
            grid_tokens = hidden_states

        H = W = self.grid_size
        x = grid_tokens.view(B, H, W, D).permute(0, 3, 1, 2).contiguous()
        cpe_delta = self.local_cpe(x)
        x = x + cpe_delta
        x = x.permute(0, 2, 3, 1).contiguous().view(B, H * W, D)

        if prefix is not None:
            return torch.cat([prefix, x], dim=1)
        return x

    def forward(self, cos_sin: Optional[CosSin], hidden_states: torch.Tensor) -> torch.Tensor:
        if self.local_cpe is not None:
            cpe_out = self._apply_local_cpe(hidden_states)
            hidden_states = self._post_norm(hidden_states, cpe_out - hidden_states)

        for mamba in self.mamba_blocks:
            residual = hidden_states
            mamba_dtype = next(mamba.parameters()).dtype
            mamba_out = mamba(hidden_states.to(mamba_dtype))
            mamba_out = mamba_out.to(residual.dtype)
            hidden_states = self._post_norm(residual, mamba_out)

        if self.self_attn is not None:
            hidden_states = self._post_norm(
                hidden_states,
                self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states),
            )

        hidden_states = self._post_norm(hidden_states, self.mlp(hidden_states))
        return hidden_states


class TinyRecursiveReasoningModelMambaVision_ACTV1ReasoningModule(nn.Module):
    def __init__(self, config: TinyRecursiveReasoningModelMambaVision_ACTV1Config):
        super().__init__()
        layers = []
        for i in range(config.L_layers):
            use_attention = i >= max(0, config.L_layers - config.attn_last_k)
            layers.append(MambaVisionStyleBlock(config, use_attention=use_attention))
        self.layers = nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs) -> torch.Tensor:
        hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)
        return hidden_states


class TinyRecursiveReasoningModelMambaVision_ACTV1_Inner(nn.Module):
    def __init__(self, config: TinyRecursiveReasoningModelMambaVision_ACTV1Config) -> None:
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
            if self.config.puzzle_emb_ndim > 0 and self.config.puzzle_emb_len == 0
            else (self.config.puzzle_emb_len if self.config.puzzle_emb_ndim > 0 else 0)
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

        self.L_level = TinyRecursiveReasoningModelMambaVision_ACTV1ReasoningModule(self.config)

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
        return TinyRecursiveReasoningModelMambaVision_ACTV1InnerCarry(
            z_H=torch.empty(
                batch_size,
                self.config.seq_len + self.puzzle_emb_len,
                self.config.hidden_size,
                dtype=self.forward_dtype,
            ),
            z_L=torch.empty(
                batch_size,
                self.config.seq_len + self.puzzle_emb_len,
                self.config.hidden_size,
                dtype=self.forward_dtype,
            ),
        )

    def reset_carry(
        self,
        reset_flag: torch.Tensor,
        carry: TinyRecursiveReasoningModelMambaVision_ACTV1InnerCarry,
    ):
        return TinyRecursiveReasoningModelMambaVision_ACTV1InnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), self.H_init, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), self.L_init, carry.z_L),
        )

    def forward(
        self,
        carry: TinyRecursiveReasoningModelMambaVision_ACTV1InnerCarry,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[
        TinyRecursiveReasoningModelMambaVision_ACTV1InnerCarry,
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

        new_carry = TinyRecursiveReasoningModelMambaVision_ACTV1InnerCarry(
            z_H=z_H.detach(),
            z_L=z_L.detach(),
        )

        output = self.lm_head(z_H)[:, self.puzzle_emb_len:]
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)

        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])


class TinyRecursiveReasoningModelMambaVision_ACTV1(nn.Module):
    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = TinyRecursiveReasoningModelMambaVision_ACTV1Config(**config_dict)
        self.inner = TinyRecursiveReasoningModelMambaVision_ACTV1_Inner(self.config)

    @property
    def puzzle_emb(self):
        return getattr(self.inner, "puzzle_emb", None)

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]
        return TinyRecursiveReasoningModelMambaVision_ACTV1Carry(
            inner_carry=self.inner.empty_carry(batch_size),
            steps=torch.zeros((batch_size,), dtype=torch.int32),
            halted=torch.ones((batch_size,), dtype=torch.bool),
            current_data={k: torch.empty_like(v) for k, v in batch.items()},
        )

    def forward(
        self,
        carry: TinyRecursiveReasoningModelMambaVision_ACTV1Carry,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[TinyRecursiveReasoningModelMambaVision_ACTV1Carry, Dict[str, torch.Tensor]]:
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
            TinyRecursiveReasoningModelMambaVision_ACTV1Carry(
                new_inner_carry,
                new_steps,
                halted,
                new_current_data,
            ),
            outputs,
        )