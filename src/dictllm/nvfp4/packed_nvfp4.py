"""Exact, rotation-free NVFP4 storage and opt-in QuTLASS execution.

Packing is implemented here (not by QuTLASS's re-quantizing export API).
The byte encoding and block-scale layout are compatible with local QuTLASS.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys

import torch
from torch import nn
import torch.nn.functional as F

from .fp4_reference import DTYPES, MAGNITUDES


SCHEMA_VERSION = 1
MANIFEST = "packed_nvfp4_manifest.json"


def padded_shape(shape):
    rows, columns = shape
    if rows < 1 or columns < 1:
        raise ValueError("Need a nonempty matrix.")
    # BF16 output vector alignment and a conservative Blackwell GEMM K tile.
    return ((rows + 7) // 8 * 8, (columns + 127) // 128 * 128)


def pack_codes(codes):
    if codes.dtype != torch.uint8 or codes.ndim != 2 or codes.shape[1] % 2:
        raise ValueError("Need 2D uint8 codes with an even column count.")
    if (codes > 15).any():
        raise ValueError("E2M1 codes must fit a nibble.")
    return (codes[:, ::2] | (codes[:, 1::2] << 4)).contiguous()


def unpack_codes(qweight):
    return torch.stack((qweight & 15, qweight >> 4), dim=-1).flatten(-2)


def decode(qweight, scale_bytes, global_scale_bytes, *, shape=None, dtype=torch.float32):
    codes = unpack_codes(qweight)
    grid = torch.tensor(MAGNITUDES, dtype=torch.float32, device=codes.device)
    levels = grid[(codes & 7).long()]
    levels = torch.where((codes & 8) != 0, -levels, levels)
    scales = scale_bytes.contiguous().view(torch.float8_e4m3fn).float()
    G = global_scale_bytes.contiguous().view(torch.float32)
    # Match the simulator: compute effective scales BEFORE multiplying levels.
    effective = scales / G
    weight = (levels.reshape(scales.shape[0], scales.shape[1], 16)
              * effective.unsqueeze(-1)).flatten(-2)
    if shape is not None:
        weight = weight[:shape[0], :shape[1]]
    return weight.to(dtype)


def encode_verified(after, details, global_scale):
    """Pack an already-verified grid without computing new scales/rounding."""
    if not details["valid"].all():
        raise ValueError("Cannot pack off-grid weights.")
    rows, columns = after.shape
    codes = details["magnitude_indices"].to(torch.uint8)
    codes |= ((after < 0).to(torch.uint8) << 3)
    # Canonicalize +/-0: a floating-point export need not retain a zero's code.
    codes[after == 0] = 0
    codes = F.pad(codes, (0, (-columns) % 16))
    scale_bytes = details["encoded_scales"].to(torch.float8_e4m3fn).view(torch.uint8).contiguous()
    global_bytes = torch.tensor([global_scale], dtype=torch.float32).view(torch.uint8)
    payload = {"qweight": pack_codes(codes), "scale_bytes": scale_bytes,
               "global_scale_bytes": global_bytes}
    restored = decode(**payload, shape=(rows, columns), dtype=after.dtype)
    if not torch.equal(restored, after):
        raise ValueError("Grid fits within verifier tolerance but is not exactly packable; refusing to re-quantize.")
    return payload


def blocked_scales(scale_bytes):
    """CPU/CUDA value-preserving scale swizzle used by local QuTLASS.

    Equivalent to qutlass.utils.to_blocked after zero-padding to [128, 4].
    Keeping this pure Torch lets the storage verifier run without QuTLASS.
    """
    rows, columns = scale_bytes.shape
    padded = F.pad(scale_bytes, (0, (-columns) % 4, 0, (-rows) % 128))
    blocks = padded.view(padded.shape[0] // 128, 128, padded.shape[1] // 4, 4).permute(0, 2, 1, 3)
    rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)
    return rearranged.contiguous().flatten().view(torch.float8_e4m3fn)


def quantize_activations(x):
    """Reference GPU/CPU W4A4 input preparation with NO rotation.

    Uses ordinary Torch operations, not QuTLASS fusedQuantizeNv (which performs
    a matrix transform). This is a correctness-first, unfused runtime path.
    """
    if x.ndim != 2 or x.shape[1] % 16:
        raise ValueError("Activation input must be padded to 16 columns.")
    original = x.to(torch.bfloat16).float()
    if not torch.isfinite(original).all():
        raise ValueError("Activation input contains non-finite entries.")
    amax = original.reshape(x.shape[0], -1, 16).abs().amax(-1)
    maximum = amax.max()
    G = 2688.0 / torch.where(maximum > 0, maximum, torch.ones_like(maximum))
    G = torch.where(maximum > 0, G, torch.ones_like(G)).reshape(1)
    if not torch.isfinite(G).all() or (G <= 0).any():
        raise ValueError("Invalid activation-global scale.")
    encoded = (amax / 6.0 * G).clamp(max=448).to(torch.float8_e4m3fn).float()
    encoded = torch.where((amax > 0) & (encoded == 0), 2.0 ** -9, encoded)
    effective = encoded / G
    effective = torch.where(amax == 0, torch.ones_like(effective), effective)
    normalized = original / effective.repeat_interleave(16, dim=1)
    boundaries = torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5.], device=x.device)
    magnitude = normalized.abs()
    indices = torch.bucketize(magnitude.contiguous(), boundaries)
    midpoint = boundaries[indices.clamp(max=6)]
    # Nearest-even ties, rather than bucketize's default lower-bin convention.
    indices += ((indices < 7) & (magnitude == midpoint) & ((indices % 2) == 1)).long()
    codes = indices.to(torch.uint8) | ((normalized < 0).to(torch.uint8) << 3)
    codes[indices == 0] = 0
    return pack_codes(codes), encoded.to(torch.float8_e4m3fn).view(torch.uint8), G


def require_qutlass(device):
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("The qutlass runtime requires CUDA; no dense fallback is used.")
    capability = torch.cuda.get_device_capability(device)
    if capability[0] not in {10, 12}:
        raise RuntimeError(f"Local NVFP4 QuTLASS needs Blackwell; device capability is {capability}.")
    try:
        library = importlib.import_module("qutlass")
        if not callable(getattr(library, "matmul_nvf4_bf16_tn", None)):
            raise ImportError("qutlass.matmul_nvf4_bf16_tn is missing")
    except ImportError as exc:
        raise RuntimeError("Install/use the working qbasis or qutlass CUDA environment.") from exc
    return library


class PackedNVFP4Linear(nn.Module):
    def __init__(self, shape, storage_shape, source_dtype, *, runtime="dequantize", cache_dequantized=False):
        super().__init__()
        if runtime not in {"dequantize", "qutlass"}:
            raise ValueError("Runtime must be dequantize or qutlass.")
        if source_dtype not in DTYPES or tuple(storage_shape) != padded_shape(shape):
            raise ValueError("Invalid packed shape/dtype specification.")
        self.out_features, self.in_features = shape
        self.storage_shape = tuple(storage_shape)
        self.source_dtype = source_dtype
        self.runtime = runtime
        self.cache_dequantized = cache_dequantized
        rows, columns = self.storage_shape
        self.register_buffer("qweight", torch.empty(rows, columns // 2, dtype=torch.uint8))
        self.register_buffer("scale_bytes", torch.empty(rows, columns // 16, dtype=torch.uint8))
        # Byte storage keeps the FP32 scalar exact across model.to(BF16/FP16).
        self.register_buffer("global_scale_bytes", torch.empty(4, dtype=torch.uint8))
        self._dense_cache = None
        self._blocked_cache = None

    def _apply(self, fn, recurse=True):
        self._dense_cache = self._blocked_cache = None
        return super()._apply(fn, recurse=recurse)

    def dequantize_weight(self, dtype=None):
        return decode(self.qweight, self.scale_bytes, self.global_scale_bytes,
                      shape=(self.out_features, self.in_features),
                      dtype=dtype or DTYPES[self.source_dtype])

    def forward(self, x):
        if x.shape[-1] != self.in_features or not x.is_floating_point():
            raise ValueError("Packed linear input has the wrong shape/dtype.")
        if x.numel() == 0:
            return x.new_empty(*x.shape[:-1], self.out_features)
        if self.runtime == "dequantize":
            if self.cache_dequantized:
                key = (x.dtype, x.device)
                if self._dense_cache is None or self._dense_cache[0] != key:
                    self._dense_cache = (key, self.dequantize_weight(x.dtype))
                weight = self._dense_cache[1]
            else:
                weight = self.dequantize_weight(x.dtype)
            return F.linear(x, weight)
        library = require_qutlass(x.device)
        if self.qweight.device != x.device:
            raise RuntimeError("Move the packed model to the same CUDA device as its inputs.")
        flat = x.reshape(-1, self.in_features)
        flat = F.pad(flat, (0, self.storage_shape[1] - self.in_features))
        activation_codes, activation_scales, activation_G = quantize_activations(flat)
        if self._blocked_cache is None:
            self._blocked_cache = blocked_scales(self.scale_bytes)
        weight_G = self.global_scale_bytes.view(torch.float32)
        alpha = (1.0 / (activation_G * weight_G)).contiguous()
        if not torch.isfinite(alpha).all() or (alpha <= 0).any():
            raise ValueError("Non-finite or underflowed QuTLASS GEMM alpha.")
        output = library.matmul_nvf4_bf16_tn(
            activation_codes, self.qweight, blocked_scales(activation_scales),
            self._blocked_cache, alpha,
        )
        return output[:, :self.out_features].reshape(*x.shape[:-1], self.out_features).to(x.dtype)


def install_packed_modules(model, records, *, runtime="dequantize", cache_dequantized=False):
    """Replace EVERY module alias with one shared packed dictionary instance."""
    from .checkpoint_adapter import capture_dictionary_bindings, restore_dictionary_bindings

    bindings = capture_dictionary_bindings(model)
    seen = set()
    for record in records:
        packed = PackedNVFP4Linear(record["shape"], record["storage_shape"], record["dtype"],
                                   runtime=runtime, cache_dequantized=cache_dequantized)
        for alias in record["aliases"]:
            if not alias.endswith(".weight") or alias in seen:
                raise ValueError(f"Invalid/duplicate alias: {alias}")
            seen.add(alias)
            module_path = alias.removesuffix(".weight")
            original = model.get_submodule(module_path)
            if not isinstance(original, nn.Linear) or original.bias is not None:
                raise ValueError(f"Only bias-free linear factors are supported: {alias}")
            if list(original.weight.shape) != record["shape"]:
                raise ValueError(f"Model/manifest shape mismatch: {alias}")
            parent_name, attribute = module_path.rsplit(".", 1)
            setattr(model.get_submodule(parent_name), attribute, packed)
    restore_dictionary_bindings(model, bindings)


def load_packed_model(checkpoint, *, runtime="dequantize", device="cpu", dtype=None,
                      cache_dequantized=False):
    """Offline, strict custom loader; vanilla HF from_pretrained is not supported."""
    from safetensors.torch import load_file
    checkpoint = Path(checkpoint).expanduser().resolve()
    manifest = json.loads((checkpoint / MANIFEST).read_text())
    if (manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("status") != "verified"
            or manifest.get("transform_class") != "identity"):
        raise ValueError("Need a verified, supported identity packed checkpoint.")
    if runtime == "qutlass":
        require_qutlass(device)
    from .checkpoint_adapter import base_config, model_class, restore_base_config, restore_native_rotary_buffers
    config_data = restore_base_config(json.loads((checkpoint / "config.json").read_text()))
    cls = model_class(config_data)
    # Avoid allocating a full floating-point model merely to overwrite it.
    with torch.device("meta"):
        model = cls(base_config(config_data))
        install_packed_modules(model, manifest["per_tensor"], runtime=runtime,
                               cache_dequantized=cache_dequantized)
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())
    expected = set(manifest["state_keys"])
    if set(index["weight_map"]) != expected:
        raise ValueError("Manifest and shard index disagree.")
    loaded = set()
    expected_shapes = {name: tuple(value.shape) for name, value in model.state_dict().items()}
    for filename in sorted(set(index["weight_map"].values())):
        path = (checkpoint / filename).resolve()
        if path.parent != checkpoint:
            raise ValueError("Shard filenames must stay inside the checkpoint.")
        state = load_file(path, device="cpu")
        for name, value in state.items():
            if name in loaded or index["weight_map"].get(name) != filename:
                raise ValueError(f"Duplicate/misindexed tensor: {name}")
            if name not in expected_shapes or tuple(value.shape) != expected_shapes[name]:
                raise ValueError(f"Unexpected tensor or shape: {name}")
            if name.endswith((".qweight", ".scale_bytes", ".global_scale_bytes")) and value.dtype != torch.uint8:
                raise ValueError(f"Packed buffers must be uint8: {name}")
            loaded.add(name)
        model.load_state_dict(state, strict=False, assign=True)
    if loaded != expected or any(value.device.type == "meta" for value in model.state_dict().values()):
        raise ValueError("Checkpoint did not initialize the complete model, including shared aliases.")
    # Nonpersistent rotary buffers originate on meta and must be rebuilt.
    for module in model.modules():
        if hasattr(module, "rope_init_fn") and hasattr(module, "inv_freq"):
            inv_freq, module.attention_scaling = module.rope_init_fn(module.config, torch.device("cpu"))
            module.register_buffer("inv_freq", inv_freq, persistent=False)
            module.original_inv_freq = module.inv_freq
    restore_native_rotary_buffers(model)
    model.eval().requires_grad_(False)
    if dtype is not None:
        model.to(dtype=dtype)
    model.to(device)
    return model
