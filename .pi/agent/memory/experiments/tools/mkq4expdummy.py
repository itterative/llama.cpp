#!/usr/bin/env python3
"""Synthesize a shape-faithful qwen4exp GGUF with zero-filled weights.

Purpose: a decode/prefill harness for the dev box that matches the real model's per-layer
dims, tensor names and quantized types, so perf deltas transfer. Layer count and the n-gram
table row count are the only reduced quantities.

Geometry is dictated by the consumer, src/models/qwen4exp.cpp, not by the exporter:
create_tensor() validates dims against hparams and throws on mismatch, and
done_getting_tensors() throws if the file holds any tensor the arch never asks for.

usage: mkq4expdummy.py --out <gguf> [--layers 4] [--ple-head-rows 1250000]
                       [--repo .] [--srcdir <dir with config.json + tokenizer files>]
"""

from __future__ import annotations

import argparse
import json
import re
import zlib
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--repo", type=Path, default=Path("."), help="llama.cpp checkout (for gguf-py)")
    ap.add_argument("--srcdir", type=Path, required=True, help="dir with config.json + tokenizer files")
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--fill", choices=("pattern", "zeros"), default="pattern",
                    help="pattern = seeded per-tensor payload, needed for logit fingerprints")
    ap.add_argument("--seed", type=int, default=20260917)
    ap.add_argument("--ctx", type=int, default=40960)
    ap.add_argument("--ple-head-rows", type=int, default=1_250_000)
    ap.add_argument("--experts", type=int, default=0, help="override num_experts (bring-up tests only)")
    args = ap.parse_args()

    sys.path.insert(0, str(args.repo / "gguf-py"))
    import gguf
    from gguf.constants import GGML_QUANT_SIZES, GGUFValueType
    from gguf.gguf_writer import GGUFWriter
    from gguf.vocab import BpeVocab, SpecialVocab

    cfg = json.loads((args.srcdir / "config.json").read_text())
    t = cfg["text_config"]

    n_embd      = int(t["hidden_size"])
    n_head      = int(t["num_attention_heads"])
    n_head_kv   = int(t["num_key_value_heads"])
    head_dim    = int(t["head_dim"])
    n_vocab     = int(t["vocab_size"])
    n_ff        = int(t["moe_intermediate_size"])
    n_ff_shexp  = int(t["shared_expert_intermediate_size"])
    n_expert    = int(t["num_experts"])
    if args.experts:
        n_expert = args.experts
    n_expert_us = int(t["num_experts_per_tok"])
    n_layer     = args.layers
    interval    = int(t.get("full_attention_interval", 4))
    hc          = int(t["hc_count"])
    hc_lr       = int(t["hc_lowrank"])
    hc_dim      = hc * n_embd
    n_rot_pct     = float(t.get("rope_parameters", {}).get("partial_rotary_factor", 1.0))
    rope_theta    = float(t.get("rope_parameters", {}).get("rope_theta", 10000.0))
    mrope         = list(t.get("rope_parameters", {}).get("mrope_section", []))

    gdn_k_heads = int(t["linear_num_key_heads"])
    gdn_k_dim   = int(t["linear_key_head_dim"])
    gdn_v_heads = int(t["linear_num_value_heads"])
    gdn_v_dim   = int(t["linear_value_head_dim"])
    gdn_conv    = int(t["linear_conv_kernel_dim"])
    key_dim     = gdn_k_heads * gdn_k_dim
    value_dim   = gdn_v_heads * gdn_v_dim
    conv_dim    = 2 * key_dim + value_dim

    idx_heads   = int(t["indexer_n_heads"])
    idx_head_dim = int(t["indexer_head_dim"])
    idx_top_k   = int(t["indexer_budget"])
    cmp_ratio   = int(t["indexer_compress_ratio"])

    ngram_size  = int(t["ngram_size"])
    heads_per_ng = int(t["heads_per_ngram"])
    ple_n_heads = (ngram_size - 1) * heads_per_ng
    ple_head_dim = n_embd // ple_n_heads
    ple_rows    = ple_n_heads * args.ple_head_rows

    n_rot       = int(n_rot_pct * head_dim)


    def is_recr(il: int) -> bool:
        return (il + 1) % interval != 0

    def is_ple(il: int) -> bool:
        return any(int(x) - 1 == il for x in t.get("ple_layer_ids", [])) and n_layer > il

    print(f"qwen4exp dummy: {n_layer} layers "
          f"({sum(1 for i in range(n_layer) if is_recr(i))} linear, "
          f"{sum(1 for i in range(n_layer) if not is_recr(i))} full), "
          f"PLE at {[i for i in range(n_layer) if is_ple(i)]}")
    print(f"  n_embd {n_embd} head_dim {head_dim} n_head {n_head}/{n_head_kv} n_rot {n_rot} "
          f"mrope {mrope}")
    print(f"  gdn key {key_dim} value {value_dim} conv_dim {conv_dim} | experts {n_expert} used "
          f"{n_expert_us} ffn {n_ff} shared {n_ff_shexp}")
    print(f"  hc {hc} low_rank {hc_lr} dim {hc_dim} | indexer {idx_heads}x{idx_head_dim} top_k "
          f"{idx_top_k} ratio {cmp_ratio}")
    print(f"  ple: heads {ple_n_heads} head_dim {ple_head_dim} rows/heads {args.ple_head_rows} "
          f"total rows {ple_rows} ({ple_rows * ple_head_dim / 1e9:.2f} B params)")

    # use_temp_file keeps each tensor's bytes out of RAM: without it add_tensor retains
    # the numpy buffer until write_tensors_to_file, which is the whole payload
    gw = GGUFWriter(str(args.out), "qwen4exp", use_temp_file=True)
    gw.add_name("qwen4exp-dummy")
    gw.add_quantization_version(2)
    gw.add_context_length(args.ctx)
    gw.add_embedding_length(n_embd)
    gw.add_feed_forward_length(n_ff)
    gw.add_block_count(n_layer)
    gw.add_head_count(n_head)
    gw.add_head_count_kv(n_head_kv)
    gw.add_rope_freq_base(rope_theta)
    gw.add_layer_norm_rms_eps(float(t.get("rms_norm_eps", 1e-6)))
    gw.add_expert_count(n_expert)
    gw.add_expert_used_count(n_expert_us)
    gw.add_expert_feed_forward_length(n_ff)
    gw.add_expert_shared_feed_forward_length(n_ff_shexp)

    def kv_uint32(key: str, val) -> None:
        if isinstance(val, (list, tuple)):
            gw.add_key_value(key, [np.uint32(v) for v in val], GGUFValueType.ARRAY,
                             sub_type=GGUFValueType.UINT32)
        else:
            gw.add_uint32(key, val)

    def kv_uint64(key: str, val) -> None:
        gw.add_key_value(key, [np.uint64(v) for v in val], GGUFValueType.ARRAY,
                         sub_type=GGUFValueType.UINT64)

    kv = "qwen4exp."
    # head_dim is not n_embd/n_head here (2560/24 = 106), so both must be explicit
    gw.add_uint32(kv + "attention.key_length", head_dim)
    gw.add_uint32(kv + "attention.value_length", head_dim)
    gw.add_uint32(kv + "rope.dimension_count", n_rot)
    # rope.dimension_sections is read as exactly 4 entries; conversion/base.py pads with 0
    gw.add_key_value(kv + "rope.dimension_sections", [np.int32(v) for v in (mrope + [0, 0, 0, 0])[:4]],
                     GGUFValueType.ARRAY, sub_type=GGUFValueType.INT32)
    gw.add_uint32(kv + "full_attention_interval", interval)
    gw.add_uint32(kv + "ssm.conv_kernel", gdn_conv)
    # ssm keys carry the GDN geometry: inner_size == value_dim, group == k heads, rank == v heads
    gw.add_uint32(kv + "ssm.inner_size", value_dim)
    gw.add_uint32(kv + "ssm.state_size", gdn_k_dim)
    gw.add_uint32(kv + "ssm.time_step_rank", gdn_v_heads)
    gw.add_uint32(kv + "ssm.group_count", gdn_k_heads)
    gw.add_uint32(kv + "hyper_connection.count", hc)
    gw.add_uint32(kv + "hyper_connection.low_rank", hc_lr)
    gw.add_uint32(kv + "attention.indexer.head_count", idx_heads)
    gw.add_uint32(kv + "attention.indexer.key_length", idx_head_dim)
    gw.add_uint32(kv + "attention.indexer.top_k", idx_top_k)
    kv_uint32(kv + "attention.compress_ratios", [0 if is_recr(i) else cmp_ratio for i in range(n_layer)])

    if ple_rows:
        kv_uint32(kv + "embedding_length_per_layer_input", ple_head_dim)
        # gguf forbids empty arrays, and a sub-2-layer build has no PLE layer at all
        ple_layers = [i for i in range(n_layer) if is_ple(i)]
        if ple_layers:
            kv_uint32(kv + "ple.layers", ple_layers)
        gw.add_uint32(kv + "ple.ngram_size", ngram_size)
        gw.add_uint32(kv + "ple.heads_per_ngram", heads_per_ng)
        gw.add_uint32(kv + "ple.conv_kernel", int(t.get("ple_conv_kernel_size", 4)))
        gen = json.loads((args.srcdir / "generation_config.json").read_text()) if \
            (args.srcdir / "generation_config.json").exists() else {}
        eos = gen.get("eos_token_id", 0)
        gw.add_uint32(kv + "ple.eos_token_id", int(eos[0] if isinstance(eos, list) else eos))
        if "image_token_id" in cfg:
            gw.add_uint32(kv + "ple.image_token_id", int(cfg["image_token_id"]))
        # odd multipliers below LLAMA_MAX_PLE_NGRAM; values are synthetic but must be nonzero
        kv_uint64(kv + "ple.layer_multipliers", [572907115, 681096679, 582687391][:ngram_size])
        kv_uint64(kv + "ple.head_offsets", [h * args.ple_head_rows for h in range(ple_n_heads)])
        kv_uint64(kv + "ple.head_vocab_sizes", [args.ple_head_rows] * ple_n_heads)

    # BpeVocab reads vocab.json (slow) or falls back to tokenizer.json (fast); SpecialVocab
    # carries merges, added tokens, bos/eos. tokenizer.ggml.pre is the Qwen BPE splitter and
    # only affects tokenisation, not shapes or timing.
    vocab = BpeVocab(args.srcdir)
    tokens, scores, toktypes = [list(x) for x in zip(*vocab.all_tokens())]
    if n_vocab > len(tokens):
        # added/special ids live above the BPE range (bos here is 248044), so the table must
        # reach config vocab_size or those ids fall outside n_vocab
        for i in range(len(tokens), n_vocab):
            tokens.append(f"<unused.{i}>")
            scores.append(0.0)
            toktypes.append(gguf.TokenType.CONTROL)
    gw.add_tokenizer_model(vocab.tokenizer_model)
    gw.add_token_list(list(tokens))
    gw.add_token_scores(list(scores))
    gw.add_token_types(list(toktypes))
    gw.add_string("tokenizer.ggml.pre", "qwen2")
    gw.add_bool("tokenizer.ggml.byte_fallback", True)
    # SpecialVocab owns merges, added tokens, bos/eos/pad ids and the chat template
    SpecialVocab(args.srcdir, load_merges=True, n_vocab=len(tokens)).add_to_gguf(gw)
    # the loader's n_vocab is the token list length, so the embedding rows must follow it
    n_vocab = len(tokens)
    print(f"  vocab: {n_vocab} tokens (config vocab_size {t['vocab_size']})")

    # --- tensors ---------------------------------------------------------------------------
    # all weights are emitted as f16; llama-quantize then picks per-tensor types the same way
    # it did for the real model, so no dtype guessing happens here
    G = gguf.GGMLQuantizationType
    by_type: dict[str, int] = {}
    total = 0
    n_written = 0

    def row_bytes(gtype: gguf.GGMLQuantizationType, ne0: int) -> int:
        blck, size = GGML_QUANT_SIZES[gtype.value]
        assert ne0 % blck == 0, f"{ne0} not a multiple of {blck} for {gtype.name}"
        return ne0 // blck * size

    # Payload fill. Zeros make a fast file but a blind one: a wrong gather row or expert index
    # returns the same value, so model-level bugs are invisible. "pattern" instead writes
    # pseudo-random bytes seeded per tensor name, which is position-sensitive and what makes the
    # logits usable as a correctness fingerprint. Quantized payloads are filled in the byte
    # domain because quantizing ~10 B params through numpy would need a 40 GB f32 source; the
    # fields a block format reads as a scale are pinned to small constants: random bytes there
    # can land on inf/NaN encodings, and a K-quant sub-block scale of zero would erase the
    # payload instead of shrinking it
    # since random bytes there can land on inf/NaN encodings (see block_q* in
    # ggml/src/ggml-common.h - note Q6_K stores d last, at byte 208 of 210).
    DBYTES = np.frombuffer(np.float16(0.001 if args.fill == "pattern" else 0.0).tobytes(), np.uint8)
    PIN = {  # block size in bytes -> [(offset, length, byte value or f16 bytes)]
        34:  [(0, 2, DBYTES)],                 # Q8_0: d
        22:  [(0, 2, DBYTES)],                 # Q5_0: d
        144: [(0, 2, DBYTES), (2, 2, 0), (4, 12, 0x11)],   # Q4_K: d, dmin, scales+mins
        176: [(0, 2, DBYTES), (2, 2, 0), (4, 12, 0x11)],   # Q5_K: d, dmin, scales
        210: [(192, 16, 0x01), (208, 2, DBYTES)],          # Q6_K: scales, then d
    }
    CHUNK = 1 << 28

    def quant_payload(name: str, nbytes: int, size: int) -> np.ndarray:
        data = np.empty(nbytes, dtype=np.uint8)
        rng = np.random.default_rng([args.seed, zlib.crc32(name.encode())])
        for off in range(0, nbytes, CHUNK):
            n = min(CHUNK, nbytes - off)
            data[off:off + n] = rng.integers(0, 256, n, dtype=np.uint8)
        if args.fill == "pattern":
            blocks = data.reshape(-1, size)
            for pos, ln, val in PIN[size]:
                blocks[:, pos:pos + ln] = val
        return data

    def float_payload(name: str, ne: tuple[int, ...], gtype: gguf.GGMLQuantizationType) -> np.ndarray:
        n = int(np.prod(ne))
        if args.fill == "zeros":
            dt = np.float32 if gtype == G.F32 else np.float16
            return np.zeros(n, dtype=dt)
        # gammas and norms want to sit near 1.0 or they squash the signal they scale; everything
        # else, router logits included, stays small so the softmax paths do not saturate
        rng = np.random.default_rng([args.seed, zlib.crc32(name.encode())])
        if name.endswith(".ssm_a"):
            # the GDN log-decay has to be negative: near +1 the state integrates over the whole
            # sequence instead of decaying, which makes the logits explode and the fingerprint
            # useless (first attempt measured PPL 4.3e74 +/- 3.6e74)
            x = (-rng.uniform(0.5, 2.0, n)).astype(np.float32)
        else:
            scale, base = (0.05, 1.0) if name.endswith("_norm.weight") else (0.02, 0.0)
            x = (rng.standard_normal(n) * scale + base).astype(np.float32)
        if gtype == G.F32:
            return x
        if gtype == G.F16:
            return x.astype(np.float16)
        return (x.view(np.uint32) >> 16).astype(np.uint16)  # bf16 by truncation

    def add(name: str, ne: tuple[int, ...], gtype: gguf.GGMLQuantizationType) -> None:
        # ne is ggml order (ne[0] fastest); gguf-py wants HF/numpy order and, for quantized
        # payloads, a uint8 buffer whose last axis is byte-count (see conversion/base.py:1111)
        nonlocal total, n_written
        hf_shape = list(reversed(ne))
        # BF16 has a higher enum value than Q4_0, so "quantized" cannot be tested by range
        if gtype in (G.F32, G.F16, G.BF16):
            data = float_payload(name, ne, gtype)
            nbytes = data.nbytes
        else:
            _, size = GGML_QUANT_SIZES[gtype.value]
            assert size in PIN, f"unpinned scale fields for {gtype.name}"
            hf_shape[-1] = row_bytes(gtype, ne[0])
            nbytes = int(np.prod(hf_shape))
            data = quant_payload(name, nbytes, size)
        gw.add_tensor(name, data, raw_shape=tuple(hf_shape), raw_dtype=gtype)
        total += nbytes
        n_written += 1

    def w(name: str, ne: tuple[int, ...]) -> None:
        by_type["F16"] = by_type.get("F16", 0) + 1
        add(name, ne, G.F16)

    # per-tensor types copied from the real bartowski Q4_K_M build (see
    # results/user/gguf-dump.log). That file is mixed precision, not uniform Q4_K, so
    # llama-quantize's own tables would not reproduce it. Patterns listed as VARIES in the
    # real file use the majority choice, which tracks a typical layer rather than the
    # high-precision first few.
    TYPES = {
        "token_embd.weight": "Q4_K", "output.weight": "Q6_K",
        "per_layer_token_embd.weight": "Q5_0",
        "output_hc_norm.weight": "F32", "output_hc_down.weight": "BF16", "output_hc_up.weight": "BF16",
        "blk.attn_q.weight": "Q4_K", "blk.attn_k.weight": "Q8_0", "blk.attn_v.weight": "Q8_0",
        "blk.attn_output.weight": "Q5_K", "blk.attn_gate.weight": "Q4_K",
        "blk.attn_q_norm.weight": "F32", "blk.attn_k_norm.weight": "F32",
        "blk.attn_qkv.weight": "Q6_K",
        "blk.indexer.q_proj.weight": "BF16", "blk.indexer.k_proj.weight": "BF16",
        "blk.indexer.q_norm.weight": "F32", "blk.indexer.k_norm.weight": "F32",
        "blk.ssm_conv1d.weight": "F32", "blk.ssm_dt.bias": "F32", "blk.ssm_a": "F32",
        "blk.ssm_beta.weight": "F32", "blk.ssm_alpha.weight": "F32", "blk.ssm_norm.weight": "F32",
        "blk.ssm_out.weight": "Q4_K",
        "blk.ple_key.weight": "Q4_K", "blk.ple_value.weight": "Q4_K",
        "blk.ple_norm_key.weight": "F32", "blk.ple_norm_query.weight": "F32",
        "blk.ple_norm_conv.weight": "F32", "blk.ple_conv1d.weight": "F16",
        "blk.ffn_gate_inp.weight": "F32", "blk.ffn_gate_inp_shexp.weight": "F32",
        "blk.ffn_gate_exps.weight": "Q4_K", "blk.ffn_up_exps.weight": "Q4_K",
        "blk.ffn_down_exps.weight": "Q5_0",
        "blk.ffn_gate_shexp.weight": "Q6_K", "blk.ffn_up_shexp.weight": "Q6_K",
        "blk.ffn_down_shexp.weight": "Q8_0",
    }
    for pre in ("hc_attn", "hc_ffn"):
        TYPES[f"blk.{pre}_norm.weight"] = "F32"
        TYPES[f"blk.{pre}_inject.weight"] = "BF16"
        TYPES[f"blk.{pre}_down.weight"] = "Q4_K"
        TYPES[f"blk.{pre}_up.weight"] = "Q5_0"

    def e(name: str, ne: tuple[int, ...]) -> None:
        key = re.sub(r"^blk\.\d+\.", "blk.", name)
        gt = G[TYPES[key]]
        by_type[gt.name] = by_type.get(gt.name, 0) + 1
        add(name, ne, gt)

    e("token_embd.weight", (n_embd, n_vocab))
    e("output.weight", (n_embd, n_vocab))
    e("output_hc_norm.weight", (hc_dim,))
    e("output_hc_down.weight", (hc_dim, hc_lr))
    e("output_hc_up.weight", (hc_lr, hc_dim))

    if ple_rows:
        # the gather target: ne = {ple_head_dim, rows}, and rows must cover max(offset+size)
        e("per_layer_token_embd.weight", (ple_head_dim, ple_rows))

    for il in range(n_layer):
        for pre in ("hc_attn", "hc_ffn"):
            e(f"blk.{il}.{pre}_norm.weight", (hc_dim,))
            e(f"blk.{il}.{pre}_down.weight", (hc_dim, hc_lr))
            e(f"blk.{il}.{pre}_up.weight", (hc_lr, hc_dim))
            e(f"blk.{il}.{pre}_inject.weight", (hc_dim, hc))

        if is_recr(il):
            e(f"blk.{il}.attn_qkv.weight", (n_embd, conv_dim))
            e(f"blk.{il}.attn_gate.weight", (n_embd, value_dim))
            e(f"blk.{il}.ssm_conv1d.weight", (gdn_conv, conv_dim))
            e(f"blk.{il}.ssm_dt.bias", (gdn_v_heads,))
            e(f"blk.{il}.ssm_a", (gdn_v_heads,))
            e(f"blk.{il}.ssm_beta.weight", (n_embd, gdn_v_heads))
            e(f"blk.{il}.ssm_alpha.weight", (n_embd, gdn_v_heads))
            e(f"blk.{il}.ssm_norm.weight", (gdn_v_dim,))
            e(f"blk.{il}.ssm_out.weight", (value_dim, n_embd))
        else:
            n_q = head_dim * n_head * 2
            e(f"blk.{il}.attn_q.weight", (n_embd, n_q))
            e(f"blk.{il}.attn_k.weight", (n_embd, head_dim * n_head_kv))
            e(f"blk.{il}.attn_v.weight", (n_embd, head_dim * n_head_kv))
            e(f"blk.{il}.attn_output.weight", (head_dim * n_head, n_embd))
            e(f"blk.{il}.attn_q_norm.weight", (head_dim,))
            e(f"blk.{il}.attn_k_norm.weight", (head_dim,))
            e(f"blk.{il}.indexer.q_proj.weight", (n_embd, idx_heads * idx_head_dim))
            e(f"blk.{il}.indexer.k_proj.weight", (n_embd, idx_head_dim))
            e(f"blk.{il}.indexer.q_norm.weight", (idx_head_dim,))
            e(f"blk.{il}.indexer.k_norm.weight", (idx_head_dim,))

        if is_ple(il):
            e(f"blk.{il}.ple_key.weight", (n_embd, hc_dim))
            e(f"blk.{il}.ple_value.weight", (n_embd, n_embd))
            for nm in ("ple_norm_key", "ple_norm_query", "ple_norm_conv"):
                e(f"blk.{il}.{nm}.weight", (hc_dim,))
            e(f"blk.{il}.ple_conv1d.weight", (int(t.get("ple_conv_kernel_size", 4)), hc_dim))

        e(f"blk.{il}.ffn_gate_inp.weight", (n_embd, n_expert))
        e(f"blk.{il}.ffn_gate_exps.weight", (n_embd, n_ff, n_expert))
        e(f"blk.{il}.ffn_up_exps.weight", (n_embd, n_ff, n_expert))
        e(f"blk.{il}.ffn_down_exps.weight", (n_ff, n_embd, n_expert))
        e(f"blk.{il}.ffn_gate_inp_shexp.weight", (n_embd,))
        e(f"blk.{il}.ffn_gate_shexp.weight", (n_embd, n_ff_shexp))
        e(f"blk.{il}.ffn_up_shexp.weight", (n_embd, n_ff_shexp))
        e(f"blk.{il}.ffn_down_shexp.weight", (n_ff_shexp, n_embd))

    gw.write_header_to_file()
    gw.write_kv_data_to_file()
    gw.write_tensors_to_file()
    gw.close()

    print(f"wrote {n_written} tensors, {total / 1e9:.2f} GB of payload -> {args.out}")
    print("types come from results/user/gguf-dump.log; no llama-quantize pass needed")
    print("  types: " + ", ".join(f"{k}={v}" for k, v in sorted(by_type.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
