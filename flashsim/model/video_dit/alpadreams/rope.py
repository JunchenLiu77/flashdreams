import torch
from einops import repeat
from torch import Tensor
from torch.distributed import ProcessGroup
from torch.distributed.tensor.device_mesh import DeviceMesh

from flashsim.distributed.context_parallel import split_inputs_cp

try:
    from transformer_engine.pytorch.attention.rope import apply_rotary_pos_emb
except ImportError:
    from transformer_engine.pytorch.attention import apply_rotary_pos_emb


def _compute_freqs(
    dim: int,
    extrapolation_ratio: float = 1.0,
    device: torch.device = torch.device("cuda"),
) -> Tensor:
    """Compute base frequencies for one RoPE dimension with NTK extrapolation.

    Args:
        dim: Number of frequency components (typically dim // 2 of head_dim).
        extrapolation_ratio: Scale factor for extrapolation; > 1 extends context length.

    Returns:
        Base frequencies of shape ``[dim // 2]``.
    """
    dim_range = (
        torch.arange(0, dim, 2, dtype=torch.float32, device=device)[: (dim // 2)] / dim
    )
    ntk_factor = extrapolation_ratio ** (dim / (dim - 2))
    theta = 10000.0 * ntk_factor
    freqs = 1.0 / (theta**dim_range)
    return freqs


class RotaryPositionEmbedding3D:
    """3D rotary position embedding for (t, h, w) sequences.

    Splits head_dim into three parts for time, height, and width. Supports
    context parallelism and time-shift for causal / streaming use.
    """

    raw_freqs_h: Tensor
    raw_freqs_w: Tensor
    raw_freqs_t: Tensor
    freqs_h: Tensor
    freqs_w: Tensor
    freqs_t: Tensor

    def __init__(
        self,
        head_dim: int,
        len_h: int,
        len_w: int,
        len_t: int,
        h_extrapolation_ratio: float = 1.0,
        w_extrapolation_ratio: float = 1.0,
        t_extrapolation_ratio: float = 1.0,
        device: torch.device = torch.device("cuda"),
    ) -> None:
        """Build 3D RoPE for the given sequence lengths and head dimension.

        Args:
            head_dim: Attention head dimension; split into h/w/t sub-dims (2:2:2 ratio).
            len_h: Sequence length along height.
            len_w: Sequence length along width.
            len_t: Sequence length along time.
            h_extrapolation_ratio: NTK extrapolation ratio for height.
            w_extrapolation_ratio: NTK extrapolation ratio for width.
            t_extrapolation_ratio: NTK extrapolation ratio for time.
        """
        self.device = device

        dim_w = dim_h = head_dim // 6 * 2
        dim_t = head_dim - (dim_h + dim_w)

        self.raw_freqs_h = _compute_freqs(dim_h, h_extrapolation_ratio, device)
        self.raw_freqs_w = _compute_freqs(dim_w, w_extrapolation_ratio, device)
        self.raw_freqs_t = _compute_freqs(dim_t, t_extrapolation_ratio, device)

        seq_t = torch.arange(len_t, dtype=torch.float32, device=device)
        seq_h = torch.arange(len_h, dtype=torch.float32, device=device)
        seq_w = torch.arange(len_w, dtype=torch.float32, device=device)

        # Align with the patchify pattern (t, h, w).
        self.freqs_t = repeat(
            torch.outer(seq_t, self.raw_freqs_t),
            "t d -> (t h w) 1 1 d",
            h=len_h,
            w=len_w,
        )
        self.freqs_h = repeat(
            torch.outer(seq_h, self.raw_freqs_h),
            "h d -> (t h w) 1 1 d",
            t=len_t,
            w=len_w,
        )
        self.freqs_w = repeat(
            torch.outer(seq_w, self.raw_freqs_w),
            "w d -> (t h w) 1 1 d",
            t=len_t,
            h=len_h,
        )

        self.device_mesh: DeviceMesh | None = None
        self.freqs_t_cp: Tensor | None = None
        self.freqs_h_cp: Tensor | None = None
        self.freqs_w_cp: Tensor | None = None

    def set_context_parallel_group(self, cp_group: ProcessGroup | None) -> None:
        """Enable or disable context parallelism by splitting frequency buffers along seq dim.

        Currently we assume the sequence length is L = T * H * W. The memory layout is (T, H, W).

        Args:
            cp_group: Process group for context parallel; use None to disable CP.
        """
        if cp_group is None:
            self.device_mesh = None
            self.freqs_t_cp = None
            self.freqs_h_cp = None
            self.freqs_w_cp = None
        else:
            self.device_mesh = DeviceMesh.from_group(cp_group, device_type="cuda")
            self.freqs_t_cp = split_inputs_cp(
                self.freqs_t, seq_dim=0, cp_group=cp_group
            )
            self.freqs_h_cp = split_inputs_cp(
                self.freqs_h, seq_dim=0, cp_group=cp_group
            )
            self.freqs_w_cp = split_inputs_cp(
                self.freqs_w, seq_dim=0, cp_group=cp_group
            )

    def is_context_parallel_enabled(self) -> bool:
        """Return True if context parallelism is active."""
        return self.device_mesh is not None

    def context_parallel_size(self) -> int:
        """Return the context parallel world size, or 1 if CP is disabled."""
        return self.device_mesh.size() if self.device_mesh is not None else 1

    def shift_t(self, offset: int) -> Tensor:
        """Shift the time dimension by the given offset (e.g. for streaming or causal steps).

        Args:
            offset: Integer offset to add to the time position indices.

        Returns:
            Concatenated RoPE frequencies of shape ``[L, 1, 1, head_dim // 2]``,
            where L is the sequence length T * H * W. The memory layout is (T, H, W).
        """
        if self.is_context_parallel_enabled():
            freqs_t = self.freqs_t_cp + offset * self.raw_freqs_t
            freqs_h = self.freqs_h_cp
            freqs_w = self.freqs_w_cp
        else:
            freqs_t = self.freqs_t + offset * self.raw_freqs_t
            freqs_h = self.freqs_h
            freqs_w = self.freqs_w
        freqs = torch.cat([freqs_t, freqs_h, freqs_w] * 2, dim=-1)
        return freqs


def apply_rope_freqs(x: Tensor, freqs: Tensor, interleaved: bool = False) -> Tensor:
    """Apply RoPE frequencies to the input tensor.

    Args:
        x: Input tensor of shape ``[B, S, H, D]``.
        freqs: RoPE frequencies of shape ``[S, 1, 1, D // 2]``.

    Returns:
        Output tensor of shape ``[B, S, H, D]``.
    """
    return apply_rotary_pos_emb(
        x, freqs, tensor_format="bshd", fused=True, interleaved=interleaved
    )
