#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 AEON-7
# Original AEON work.
"""Bake AEON ablation variants into stock Inkling-2-Mini weights (CPU only).

Input: one or more AEON variant files (aeon_variants.py format):
  dirs, scales              o_proj (wo_ud) edit      [D]|[24,D]|[48,D]|[48,K,D], [48]|[48,K]
  resid_dirs, resid_scales  residual-stream edit     same shapes
Output: a weight overlay (only the edited tensors, original names, dtypes and
shapes; loads with AEON_WEIGHT_OVERLAY) or a full checkpoint directory (the
stock files copied, edited tensors patched in place byte for byte; loads in
the stock architecture anywhere: stock vLLM, transformers, converters). No
new modules or tensors.

Folds (layer application a = 0..47; loop layers 6..17 run 3 passes,
a = 6 + 12 * pass + (layer - 6)):
  o_proj edit at a      W_o' = P_{K-1} ... P_0 W_o  on layers.L.attn.wo_ud
                        (P_k = I - s_k r_k r_k^T). Loop layers need the same
                        dirs and scales at a, a+12, a+24 (shared weights).
  residual edit at a    the runtime hook edits the attention-input mix of
                        application a+1, whose only readers are that layer's
                        input projections: W' = W P_{K-1} ... P_0 on
                        wq_du, wr_du, gate_attn, attn_sink_proj, wk_dv, wv_dv
                        (+ k_sconv.weight_proj, v_sconv.weight_proj on global
                        layers). a = 47 (final hidden) folds into the unembed.
                        Bakeable apps: 0-4 (layers 1-5) and 41-46 (layers
                        18-23), 47 (unembed). Apps 5-40 feed the shared loop
                        layers (and pass restarts / pre-norm fp32 mixes) and
                        are refused.
R6 all-readers mode (--readers all; default --readers consumer is the R5
fold above, unchanged). Residual apps 41-47 only (no loop layer reads the
stream after app 41; no controller is downstream). The R5 runtime hook is
LOCAL: it edits only the attention-input mix of visit a+1 (or the final
hidden at a = 47); the block stack, the MLP-input mixes and every later mix
are untouched. --readers all instead treats the edit as "removed from the
stream for every later bf16/fp32 reader", a stronger, different edit:
  for every app a with an edit, every reader of the stream after a:
    attention inputs of visits a+1..47 (layers a-23..23; on global layers
    q-stream rows wq/wr/gate/sink read q_input_norm, wk_dv + k_sconv taps
    read k_input_norm, wv_dv + v_sconv taps read v_input_norm),
    the MoE router mlp.gate.weight of visits a+1..47 (behind mlp_norm;
    the fp32 bias is unchanged),
    the unembed (behind the final norm),
    optionally (--fold-queries) the fp32 AttnRes queries of those mixes
    (attn_queries[L-1], mlp_queries[L], output_query; value-normalised
    blocks, no gain).
  Expert inputs (NVFP4 / MXFP8 w13) also read the MLP-input mix but are NOT
  edited (needs a requant); measure what they still see at runtime.
  Gains (--gain-mode oblique, default). v lives in the post-gain space of
  app a's own consumer (gain g = attn_norm, or q_input_norm on global
  layer 23, or the final norm for a = 47). In normed-stream space the hook
  is the oblique edit x -> x - s (v/g) ((g*v) . x). A reader behind gain h
  folds W' = W (I - s p q^T), p = h*v/g, q = g*v/h (p . q = 1, so s = 1
  still annihilates exactly). The own consumer's tensors (every stream)
  get the plain W (I - s v v^T), identical to the R5 fold and the hook.
  Queries read normed blocks with no gain: h = 1. |h| is clamped at 1e-3.
  --gain-mode plain folds W (I - s v v^T) into every reader, ignoring gains.
  Norm-scale approximation: a real stream edit would also change each
  mix's RMS (factor 1/sqrt(1 - s(2-s) c^2), c = the normed v-component) and
  the value-normalised block norms (hence the softmax weights unless
  queries are folded). The reader-side fold ignores both; for |c| < 0.1
  the RMS factor is < 0.5%.
  Composition: per reader tensor the edits of all apps a < visit are
  applied in ascending app order, within an app in ascending k:
  W' = W E_last ... E_first, E = I - s p q^T. Several variants are merged
  in app order (variant order within an app).
  Loop layers / apps 5-40 stay refused in both modes (shared weights read
  through fp32 pre-norm mixes and message blends; per-pass dirs differ).
Math in fp64, one rounding to the stored dtype. The stock directory is only
read. All consumers in the stock checkpoint are bf16; a quantized consumer is
refused (a requant residual would have to be reported, see R5_RESIDUAL.md).

Usage:
  r5_bake.py --stock /path/Inkling-2-Mini --variant v.safetensors [--variant w.safetensors]
             (--out-overlay out.safetensors | --out-checkpoint /new/dir) [--report r.json]
             [--readers consumer|all] [--gain-mode oblique|plain] [--fold-queries]
"""
from __future__ import annotations

import argparse, json, os, shutil, struct, sys

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

NUM_APPS, NUM_LAYERS = 48, 24
LOOP_START, LOOP_LEN, LOOP_PASSES = 6, 12, 3
PREFIX = "model.llm."
ATTN_INPUTS = ("attn.wq_du.weight", "attn.wr_du.weight", "attn.gate_attn.weight",
               "attn.attn_sink_proj.weight", "attn.wk_dv.weight", "attn.wv_dv.weight")
GLOBAL_EXTRA = ("attn.k_sconv.weight_proj", "attn.v_sconv.weight_proj")
RESID_BAKEABLE = set(range(0, 5)) | set(range(41, 48))
FLOAT_DTYPES = (torch.bfloat16, torch.float16, torch.float32)
RESID_ALL_READERS = set(range(41, 48))
GLOBAL_LAYERS = (5, 11, 17, 23)
# global-layer attention tensors -> the input stream (norm) they read
STREAM_NORM = {"attn.wq_du.weight": "q_input_norm", "attn.wr_du.weight": "q_input_norm",
               "attn.gate_attn.weight": "q_input_norm", "attn.attn_sink_proj.weight": "q_input_norm",
               "attn.wk_dv.weight": "k_input_norm", "attn.k_sconv.weight_proj": "k_input_norm",
               "attn.wv_dv.weight": "v_input_norm", "attn.v_sconv.weight_proj": "v_input_norm"}
GAIN_CLAMP = 1e-3


class NotBakeable(ValueError):
    pass


def app_to_layer(app: int) -> int:
    loop_end = LOOP_START + LOOP_LEN * LOOP_PASSES
    if app < LOOP_START:
        return app
    if app < loop_end:
        return LOOP_START + (app - LOOP_START) % LOOP_LEN
    return app - LOOP_LEN * (LOOP_PASSES - 1)


def layer_apps(layer: int) -> list[int]:
    if LOOP_START <= layer < LOOP_START + LOOP_LEN:
        return [layer + LOOP_LEN * p for p in range(LOOP_PASSES)]
    return [layer if layer < LOOP_START else layer + LOOP_LEN * (LOOP_PASSES - 1)]


def resid_consumer(app: int) -> int | None:
    """Layer whose attention reads the mix edited at app (None = unembed)."""
    if app == NUM_APPS - 1:
        return None
    return app_to_layer(app + 1)


def canonicalize(dirs, scales):
    """Same rules as aeon_variants.canonicalize -> [48,K,D] unit rows (fp64), [48,K]."""
    dirs = torch.as_tensor(dirs).double()
    scales = torch.as_tensor(scales).double()
    if dirs.dim() == 1:
        dirs = dirs.expand(NUM_APPS, dirs.shape[0])
    if dirs.dim() == 2:
        if dirs.shape[0] == NUM_LAYERS:
            dirs = dirs[[app_to_layer(a) for a in range(NUM_APPS)]]
        if dirs.shape[0] != NUM_APPS:
            raise ValueError(f"dirs must be [D], [24, D], [48, D] or [48, K, D], got {tuple(dirs.shape)}")
        dirs = dirs[:, None, :]
    if dirs.dim() != 3 or dirs.shape[0] != NUM_APPS:
        raise ValueError(f"dirs must be [D], [24, D], [48, D] or [48, K, D], got {tuple(dirs.shape)}")
    k = dirs.shape[1]
    if scales.dim() == 1:
        if scales.shape[0] != NUM_APPS:
            raise ValueError(f"scales must be [48] or [48, K], got {tuple(scales.shape)}")
        scales = scales[:, None].expand(NUM_APPS, k)
    if tuple(scales.shape) != (NUM_APPS, k):
        raise ValueError(f"scales must be [48] or [48, {k}], got {tuple(scales.shape)}")
    nrm = dirs.norm(dim=-1, keepdim=True)
    dirs = torch.where(nrm > 1e-12, dirs / nrm.clamp_min(1e-12), torch.zeros_like(dirs))
    scales = torch.where(nrm[..., 0] > 1e-12, scales, torch.zeros_like(scales))
    return dirs.contiguous(), scales.contiguous()


def read_variant(path: str):
    t = load_file(path)
    o = r = None
    if ("dirs" in t) != ("scales" in t) or ("resid_dirs" in t) != ("resid_scales" in t):
        raise ValueError(f"{path}: give both dirs and scales (or both resid_*), or neither")
    if "dirs" in t:
        o = canonicalize(t["dirs"], t["scales"])
    if "resid_dirs" in t:
        r = canonicalize(t["resid_dirs"], t["resid_scales"])
    if o is None and r is None:
        raise ValueError(f"{path}: no dirs/scales or resid_dirs/resid_scales")
    return o, r


def plan(variants: list[tuple[str, tuple | None, tuple | None]]):
    """-> list of edits (tensor_name_suffix, side, [(r, s), ...]) in application
    order; raises NotBakeable with every reason found."""
    edits, errors = [], []
    for path, o, r in variants:
        if o is not None:
            d, s = o
            for layer in range(NUM_LAYERS):
                apps = layer_apps(layer)
                if all(not bool((s[a] != 0).any()) for a in apps):
                    continue
                a0 = apps[0]
                for a in apps[1:]:
                    if not (torch.equal(s[a], s[a0]) and torch.allclose(d[a], d[a0], atol=1e-6, rtol=0)):
                        errors.append(f"{os.path.basename(path)}: o_proj edit on shared loop layer {layer} "
                                      f"differs between apps {apps} (per-pass scales/dirs cannot be baked "
                                      "into weights the passes share)")
                        break
                else:
                    pairs = [(d[a0, k], float(s[a0, k])) for k in range(d.shape[1]) if s[a0, k] != 0]
                    edits.append((f"layers.{layer}.attn.wo_ud.weight", "out", pairs, path, apps))
        if r is not None:
            d, s = r
            bad = [a for a in range(NUM_APPS) if bool((s[a] != 0).any()) and a not in RESID_BAKEABLE]
            if bad:
                errors.append(f"{os.path.basename(path)}: residual edits at apps {bad} are not bakeable "
                              "(apps 5-40 feed the shared loop layers / pass restarts; only 0-4 and 41-47 fold)")
                continue
            for a in sorted(RESID_BAKEABLE):
                if not bool((s[a] != 0).any()):
                    continue
                pairs = [(d[a, k], float(s[a, k])) for k in range(d.shape[1]) if s[a, k] != 0]
                L = resid_consumer(a)
                targets = (["unembed.weight"] if L is None else
                           [f"layers.{L}.{n}" for n in ATTN_INPUTS + GLOBAL_EXTRA])
                for t in targets:
                    edits.append((t, "in", pairs, path, [a]))
    if errors:
        raise NotBakeable("not bakeable:\n  " + "\n  ".join(errors))
    return edits


def attn_reader_gain(layer: int, tensor: str) -> str:
    """Norm-gain tensor suffix the attention reader `tensor` of `layer` sits behind."""
    if layer in GLOBAL_LAYERS:
        return f"layers.{layer}.{STREAM_NORM[tensor]}.weight"
    return f"layers.{layer}.attn_norm.weight"


def own_gain(app: int) -> str:
    """Gain of app a's own consumer (stream 0): the space the hook's v lives in."""
    L = resid_consumer(app)
    if L is None:
        return "norm.weight"
    return attn_reader_gain(L, "attn.wq_du.weight")


def readers_after(app: int, fold_queries: bool):
    """[(tensor_suffix, gain_suffix | None (= no gain), own)] read the stream after app a."""
    out = []
    for b in range(app + 1, NUM_APPS):
        L = app_to_layer(b)
        own = b == app + 1
        for t in ATTN_INPUTS + GLOBAL_EXTRA:
            if t in GLOBAL_EXTRA and L not in GLOBAL_LAYERS:
                continue
            out.append((f"layers.{L}.{t}", attn_reader_gain(L, t), own))
        out.append((f"layers.{L}.mlp.gate.weight", f"layers.{L}.mlp_norm.weight", False))
        if fold_queries:
            out.append((f"attn_res.attn_queries.{L - 1}", None, False))
            out.append((f"attn_res.mlp_queries.{L}", None, False))
    out.append(("unembed.weight", "norm.weight", app == NUM_APPS - 1))
    if fold_queries:
        out.append(("attn_res.output_query", None, False))
    return out


def plan_all(variants, fold_queries: bool):
    """--readers all: -> list of ("in_all", tensor, [(app, gain_suffix|None, own, r, s), ...])
    with each reader's edits in ascending app order (variant order within an app)."""
    errors, per_app = [], []
    for vi, (path, o, r) in enumerate(variants):
        if r is None:
            continue
        d, s = r
        bad = [a for a in range(NUM_APPS) if bool((s[a] != 0).any()) and a not in RESID_ALL_READERS]
        if bad:
            errors.append(f"{os.path.basename(path)}: residual edits at apps {bad}: --readers all folds "
                          "apps 41-47 only (earlier apps feed the shared loop layers / prelude readers)")
            continue
        for a in sorted(RESID_ALL_READERS):
            for k in range(d.shape[1]):
                if s[a, k] != 0:
                    per_app.append((a, vi, k, d[a, k], float(s[a, k]), path))
    if errors:
        raise NotBakeable("not bakeable:\n  " + "\n  ".join(errors))
    per_app.sort(key=lambda e: (e[0], e[1], e[2]))
    readers: dict[str, list] = {}
    order: list[str] = []
    for a, vi, k, v, sc, path in per_app:
        for t, gain, own in readers_after(a, fold_queries):
            if t not in readers:
                readers[t] = []; order.append(t)
            readers[t].append((a, gain, own, v, sc, path))
    return [(t, readers[t]) for t in order]


def oblique_pairs(entries, gains, mode: str):
    """[(app, gain, own, v, s, path)] -> [(p, q, s)]: E = I - s p q^T in the reader's input space."""
    out = []
    for a, gain, own, v, sc, path in entries:
        if mode == "plain" or own:
            out.append((v, v, sc)); continue
        g = gains[own_gain(a)]
        h = torch.ones_like(v) if gain is None else gains[gain]
        p = h * v / g
        q = g * v / h
        out.append((p, q, sc))
    return out


def apply_in_pq(W: torch.Tensor, pqs) -> torch.Tensor:
    """W' = W E_last ... E_first, E = I - s p q^T (input-side, applied in stream order)."""
    for p, q, s in reversed(pqs):
        W = W - s * torch.outer(W @ p, q)
    return W


def apply_out(W: torch.Tensor, pairs) -> torch.Tensor:
    """W' = P_{K-1} ... P_0 W (rows live in the residual space)."""
    for r, s in pairs:
        W = W - s * torch.outer(r, r @ W)
    return W


def apply_in(W: torch.Tensor, pairs) -> torch.Tensor:
    """W' = W P_{K-1} ... P_0 (columns read the residual space)."""
    for r, s in reversed(pairs):
        W = W - s * torch.outer(W @ r, r)
    return W


def find_tensor(stock: str):
    """name -> file for every tensor in the stock checkpoint."""
    idx = os.path.join(stock, "model.safetensors.index.json")
    if os.path.exists(idx):
        wm = json.load(open(idx))["weight_map"]
        return {k: os.path.join(stock, v) for k, v in wm.items()}
    out = {}
    for fn in sorted(os.listdir(stock)):
        if fn.endswith(".safetensors") and fn.startswith("model"):
            with safe_open(os.path.join(stock, fn), "pt") as f:
                for k in f.keys():
                    out[k] = os.path.join(stock, fn)
    return out


def _load_gains(where, names):
    out = {}
    for n in names:
        name = PREFIX + n
        if name not in where:
            raise NotBakeable(f"norm gain {name} not in the stock checkpoint")
        with safe_open(where[name], "pt") as f:
            g = f.get_tensor(name).double()
        out[n] = torch.where(g.abs() < GAIN_CLAMP, torch.full_like(g, GAIN_CLAMP) * torch.sign(g + 1e-30), g)
    return out


def bake(stock: str, variant_paths: list[str], readers: str = "consumer",
         gain_mode: str = "oblique", fold_queries: bool = False):
    """-> (baked {name: tensor in stock dtype}, report)."""
    if readers == "consumer":
        return _bake_consumer(stock, variant_paths)
    if readers != "all" or gain_mode not in ("oblique", "plain"):
        raise ValueError(f"readers={readers!r} gain_mode={gain_mode!r}")
    variants = [(p, *read_variant(p)) for p in variant_paths]
    out_edits = [e for e in plan([(p, o, None) for p, o, r in variants]) if e[1] == "out"]
    in_edits = plan_all(variants, fold_queries)
    where = find_tensor(stock)
    gnames = set()
    for t, entries in in_edits:
        for a, gain, own, v, sc, path in entries:
            gnames.add(own_gain(a))
            if gain is not None:
                gnames.add(gain)
    gains = _load_gains(where, sorted(gnames))
    orig, work, log, stats = {}, {}, [], {}
    for suffix, side, pairs, path, apps in out_edits:
        name = PREFIX + suffix
        if name not in where:
            raise NotBakeable(f"writer tensor {name} not in the stock checkpoint")
        if name not in orig:
            with safe_open(where[name], "pt") as f:
                orig[name] = f.get_tensor(name)
            if orig[name].dtype not in FLOAT_DTYPES:
                raise NotBakeable(f"{name}: {orig[name].dtype} is quantized/packed")
            work[name] = orig[name].double()
        work[name] = apply_out(work[name], pairs)
        log.append(dict(tensor=name, side="out", apps=apps, rank=len(pairs), variant=os.path.basename(path)))
    for suffix, entries in in_edits:
        name = PREFIX + suffix
        if name not in where:
            raise NotBakeable(f"reader tensor {name} not in the stock checkpoint")
        with safe_open(where[name], "pt") as f:
            t0 = f.get_tensor(name)
        if t0.dtype not in FLOAT_DTYPES or t0.dim() not in (1, 2):
            raise NotBakeable(f"{name}: {t0.dtype} {tuple(t0.shape)} is quantized/packed; "
                              "baking would need a requant (not supported)")
        D = entries[0][3].numel()
        if t0.shape[-1] != D:
            raise NotBakeable(f"{name}: residual dim {t0.shape[-1]} != direction dim {D}")
        orig[name] = t0
        W = t0.double().reshape(-1, D)
        pqs = oblique_pairs(entries, gains, gain_mode)
        W = apply_in_pq(W, pqs)
        work[name] = W.reshape(t0.shape)
        W0, Wq = t0.double().reshape(-1, D), W.to(t0.dtype).double()
        stats[name] = dict(dtype=str(t0.dtype), shape=list(t0.shape),
                           apps=[e[0] for e in entries], own=[bool(e[2]) for e in entries],
                           rel_change=float((W - W0).norm() / W0.norm().clamp_min(1e-30)),
                           rounding_rel_err=float((Wq - W).norm() / W.norm().clamp_min(1e-30)))
        log.append(dict(tensor=name, side="in", apps=sorted({e[0] for e in entries}), rank=len(entries),
                        gain_mode=gain_mode, variant=sorted({os.path.basename(e[5]) for e in entries})))
    for n in work:
        if n not in stats:   # wo_ud
            W0, We = orig[n].double(), work[n]
            stats[n] = dict(dtype=str(orig[n].dtype), shape=list(orig[n].shape),
                            rel_change=float((We - W0).norm() / W0.norm()),
                            rounding_rel_err=float((We.to(orig[n].dtype).double() - We).norm() / We.norm()))
    baked = {n: work[n].to(orig[n].dtype).contiguous() for n in work}
    report = dict(variants=[os.path.basename(p) for p in variant_paths], readers="all", gain_mode=gain_mode,
                  fold_queries=fold_queries, edits=log, tensors=stats, n_tensors=len(baked),
                  not_edited=("expert w13 (NVFP4/MXFP8) reads the MLP-input mix: edit not folded; "
                              "router bias fp32 unchanged"))
    return baked, report


def _bake_consumer(stock: str, variant_paths: list[str]):
    variants = [(p, *read_variant(p)) for p in variant_paths]
    edits = plan(variants)
    where = find_tensor(stock)
    work: dict[str, torch.Tensor] = {}
    orig: dict[str, torch.Tensor] = {}
    log = []
    for suffix, side, pairs, path, apps in edits:
        name = PREFIX + suffix
        if name not in where:
            if suffix.endswith(GLOBAL_EXTRA):
                continue                                   # local layer: no dynamic taps
            raise NotBakeable(f"consumer tensor {name} not in the stock checkpoint")
        if name not in orig:
            with safe_open(where[name], "pt") as f:
                t = f.get_tensor(name)
            if t.dtype not in FLOAT_DTYPES or t.dim() != 2:
                raise NotBakeable(f"{name}: {t.dtype} {tuple(t.shape)} is quantized/packed; "
                                  "baking would need a requant (not supported)")
            orig[name] = t
            work[name] = t.double()
        D = pairs[0][0].numel()
        dim = work[name].shape[0 if side == "out" else 1]
        if dim != D:
            raise NotBakeable(f"{name}: residual dim {dim} != direction dim {D}")
        work[name] = (apply_out if side == "out" else apply_in)(work[name], pairs)
        log.append(dict(tensor=name, side=side, apps=apps, rank=len(pairs), variant=os.path.basename(path)))
    baked = {n: work[n].to(orig[n].dtype).contiguous() for n in work}
    # rounding report: share of each removed direction left after storage rounding
    stats = {}
    for suffix, side, pairs, path, apps in edits:
        n = PREFIX + suffix
        if n not in baked:
            continue
        W0, Wq, We = orig[n].double(), baked[n].double(), work[n]
        rec = stats.setdefault(n, dict(dtype=str(orig[n].dtype), shape=list(orig[n].shape),
                                       rel_change=float((We - W0).norm() / W0.norm()),
                                       rounding_rel_err=float((Wq - We).norm() / We.norm()), leak=[]))
        for r, s in pairs:
            comp0 = (r @ W0) if side == "out" else (W0 @ r)
            compq = (r @ Wq) if side == "out" else (Wq @ r)
            # left fraction of the removed component (s = 1 -> ideal 0)
            rec["leak"].append(dict(apps=apps, s=s, left=float(compq.norm() / comp0.norm().clamp_min(1e-30)),
                                    ideal=abs(1 - s)))
    report = dict(variants=[os.path.basename(p) for p in variant_paths], edits=log, tensors=stats,
                  n_tensors=len(baked))
    return baked, report


def _check_out(stock: str, out: str):
    s, o = os.path.realpath(stock), os.path.realpath(out)
    if o == s or o.startswith(s + os.sep):
        raise SystemExit(f"refusing to write inside the stock directory {stock}")


def write_overlay(baked, report, out: str, stock: str):
    _check_out(stock, out)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    tmp = out + ".tmp"
    save_file(baked, tmp, metadata={"format": "pt", "aeon_r5_bake": json.dumps(report["edits"], default=str)[:60000]})
    os.replace(tmp, out)


def _header(path: str):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def write_checkpoint(baked, report, out: str, stock: str):
    """Copy the stock directory, then overwrite each edited tensor's bytes in
    place (same name, dtype, shape -> same byte range; headers untouched)."""
    _check_out(stock, out)
    if os.path.exists(out):
        raise SystemExit(f"{out} exists; give a new directory")
    os.makedirs(out)
    where = find_tensor(stock)
    for fn in sorted(os.listdir(stock)):
        src = os.path.join(stock, fn)
        if os.path.isdir(src):
            shutil.copytree(src, os.path.join(out, fn))
        else:
            shutil.copyfile(src, os.path.join(out, fn))      # data only: copies are writable
    for name, t in baked.items():
        path = os.path.join(out, os.path.basename(where[name]))
        hdr, base = _header(path)
        e = hdr[name]
        a, b = e["data_offsets"]
        raw = t.contiguous().view(-1).view(torch.uint8).numpy().tobytes()
        assert len(raw) == b - a, (name, len(raw), b - a)
        with open(path, "r+b") as f:
            f.seek(base + a)
            f.write(raw)
    with open(os.path.join(out, "AEON_R5_BAKE.json"), "w") as f:
        json.dump(report, f, indent=1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stock", required=True)
    ap.add_argument("--variant", action="append", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--out-overlay")
    g.add_argument("--out-checkpoint")
    ap.add_argument("--report")
    ap.add_argument("--readers", choices=("consumer", "all"), default="consumer",
                    help="consumer: R5 fold (hook-equivalent); all: every later bf16/fp32 reader, apps 41-47")
    ap.add_argument("--gain-mode", choices=("oblique", "plain"), default="oblique")
    ap.add_argument("--fold-queries", action="store_true", help="--readers all: also fold fp32 AttnRes queries")
    a = ap.parse_args(argv)
    try:
        baked, report = bake(a.stock, a.variant, a.readers, a.gain_mode, a.fold_queries)
    except NotBakeable as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    if a.out_overlay:
        write_overlay(baked, report, a.out_overlay, a.stock)
    else:
        write_checkpoint(baked, report, a.out_checkpoint, a.stock)
    if a.report:
        json.dump(report, open(a.report, "w"), indent=1)
    if a.readers == "consumer":
        worst = max((l["left"] - l["ideal"] for t in report["tensors"].values() for l in t["leak"]), default=0.0)
        print(f"baked {report['n_tensors']} tensors from {len(a.variant)} variant(s); "
              f"worst storage-rounding leak {worst:.4f} of the removed component")
    else:
        worst = max((t["rounding_rel_err"] for t in report["tensors"].values()), default=0.0)
        print(f"baked {report['n_tensors']} tensors (readers=all, gain={a.gain_mode}, "
              f"queries={a.fold_queries}) from {len(a.variant)} variant(s); worst rounding rel err {worst:.2e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
