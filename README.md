# AEON vLLM Ultimate ROCm (x86)

Early access. Pending community testing. This is not a final release.

`ghcr.io/aeon-7/aeon-vllm-ultimate-rocm`

Sibling images:

| Host | Image |
|---|---|
| ROCm x86 (this repo) | `ghcr.io/aeon-7/aeon-vllm-ultimate-rocm` |
| NVIDIA Blackwell x86 | `ghcr.io/aeon-7/aeon-vllm-ultimate-rtx` |
| DGX Spark | `ghcr.io/aeon-7/aeon-vllm-ultimate` |

## What this image is

Official vLLM **v0.31.0** ROCm image, digest `sha256:749f6f3f944f12af49966ac523c1f8573e4b229594b541954bd5f879b4496b1f`, plus two AEON Python patches. Nothing was recompiled. ROCm inside the base image is 7.2.3. The base already contains gfx90a, gfx942, gfx950, gfx1100, gfx1101, gfx1150, gfx1151, gfx1200, and gfx1201.

| Patch | What it changes | Default |
|---|---|---|
| MXFP6 dequant-once | The Gated DeltaNet MXFP6 projections are unpacked once at load, with the same function stock vLLM calls on every forward. Activations stay quantize-dequantize. Output matches stock emulation. | `AEON_MXFP6_DEQUANT_AT_LOAD=1` |
| vLLM #55601 | Hybrid Mamba state is indexed by `mamba_block_size`, so a DFlash2 drafter does not point prefix-cache hits at the wrong state. | Always applied |

Not in this early-access build: a gfx12 packed MXFP4 kernel, vllm-omni speech-out, the reasoning-EOS force-end port, and any W4A8 / FP8-KV / quantized-allreduce path. Those change numerics or are not validated. AITER stays off (`VLLM_ROCM_USE_AITER=0`) except where you opt in later.

16 GB cards do not have room for the unpacked MXFP6 copy. Set `AEON_MXFP6_DEQUANT_AT_LOAD=0` there. Output stays the same and decode is slower.

## Which weights

ROCm does not load the NVIDIA checkpoint.

| Checkpoint | Where it runs |
|---|---|
| [AEON-7/Qwen3.8-27B-AEON-ULTIMATE-UNCENSORED-MXFP4-MXFP6-ROCm](https://huggingface.co/AEON-7/Qwen3.8-27B-AEON-ULTIMATE-UNCENSORED-MXFP4-MXFP6-ROCm) | This image. Quark MXFP4 MLP + MXFP6 GDN. Gated early access. |
| [AEON-7/Qwen3.8-27B-AEON-ULTIMATE-UNCENSORED-NVFP4-MIXED](https://huggingface.co/AEON-7/Qwen3.8-27B-AEON-ULTIMATE-UNCENSORED-NVFP4-MIXED) | CUDA only (DGX Spark, RTX 5090, RTX PRO 6000). Do not point this image at it. |

The Hugging Face card above is the source of truth for the weights. The serve kit's scripts still run on stock `vllm/vllm-openai-rocm:v0.31.0` by bind-mounting these same two files. This image bakes those files in, so the mount is unnecessary.

DFlash2 on ROCm uses the **BF16** drafter, not the NVFP4 one. The current drafter card is [AEON-7/AEON-DFlash2-Qwen3.8-27B](https://huggingface.co/AEON-7/AEON-DFlash2-Qwen3.8-27B) (`bf16/`). The ROCm weight repo also ships an earlier `dflash2/` folder.

## Pull

```bash
docker pull ghcr.io/aeon-7/aeon-vllm-ultimate-rocm:2026-10-06-v0.31.0-x86-early-access
```

Tags `early-access` and `x86` point at the same build until a tested tag replaces them. Pin the digest once `docker pull` prints it.

## 2× Radeon AI PRO R9700 (gfx1201, 32 GB)

Text-only. TRITON attention on the target and on the draft. RCCL on Simple with P2P off. Context 65,536. DFlash2 K=9 with the batch-size lattice. Prefix caching stays off until the baked #55601 line has been checked on AMD. MXFP6 unpack-once is on.

Keep the iGPU out of `HIP_VISIBLE_DEVICES` and `ROCR_VISIBLE_DEVICES`. The example assumes the two R9700s are `0,1`.

```bash
MODEL=$HOME/models/aeon-mxfp4-rocm
DRAFT=$MODEL/dflash2          # or the bf16/ folder of AEON-DFlash2-Qwen3.8-27B
CACHE=$HOME/.cache/aeon-vllm-rocm
mkdir -p "$CACHE"

docker run -d --name aeon-rocm \
  --device /dev/kfd --device /dev/dri \
  --group-add video --group-add render \
  --cap-add SYS_PTRACE --security-opt seccomp=unconfined --ipc=host \
  --shm-size 16g \
  -p 8000:8000 \
  -e HIP_VISIBLE_DEVICES=0,1 \
  -e ROCR_VISIBLE_DEVICES=0,1 \
  -e NCCL_PROTO=Simple \
  -e NCCL_P2P_DISABLE=1 \
  -e VLLM_ROCM_USE_AITER=0 \
  -e AEON_MXFP6_DEQUANT_AT_LOAD=1 \
  -v "$MODEL":/model:ro \
  -v "$DRAFT":/draft:ro \
  -v "$CACHE":/root/.cache/vllm \
  ghcr.io/aeon-7/aeon-vllm-ultimate-rocm:2026-10-06-v0.31.0-x86-early-access \
  /model \
  --served-model-name aeon \
  --trust-remote-code \
  --dtype bfloat16 \
  --tensor-parallel-size 2 \
  --max-model-len 65536 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.90 \
  --kv-cache-dtype auto \
  --attention-backend TRITON_ATTN \
  --no-enable-prefix-caching \
  --language-model-only \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --override-generation-config '{"temperature":0.6,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"repetition_penalty":1.0}' \
  --speculative-config '{"method":"dflash","model":"/draft","num_speculative_tokens":9,"num_speculative_tokens_per_batch_size":[[1,2,9],[3,8,7],[9,12,6],[13,16,4]],"attention_backend":"TRITON_ATTN","draft_sample_method":"probabilistic","rejection_sample_method":"standard"}'
```

Plain serve, no drafter: drop `--speculative-config` and the `/draft` mount, and you can leave prefix caching on (remove `--no-enable-prefix-caching`). Raise `--max-model-len` to `131072` for that path. One 32 GB card uses `--tensor-parallel-size 1`, `HIP_VISIBLE_DEVICES` of that card only, and no `NCCL_*` variables.

Sampling above is the kit default (`temperature` 0.6). The checkpoint's own generation config may differ. Pass the request temperature you actually want to measure.

## What "early access" means

The two patches match the files in the ROCm serve kit. MXFP6 unpack-once has matching numerics against stock emulation (`torch.equal` on the dequant). It has not yet been the default in a published community run on Radeon: Todd's 2× R9700 numbers were on stock unpack-every-forward, about 1.7 tok/s single stream. The unpack was the slow part of that run (about 8.7 ms per layer versus about 0.2 ms for the BF16 GEMM). This image turns the unpack-once path on by default so that cost is paid at load. Treat speed after this image as unmeasured until someone posts a log.

If load fails with `dict object has no attribute endswith`, the tag is not this image. v0.28 through v0.30 cannot read this checkpoint's `algo_config`.

Support the work: [Patreon membership](https://www.patreon.com/cw/AeonForge7/membership).
