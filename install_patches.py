#!/usr/bin/env python3
"""Bake the AEON ROCm patches into the installed vLLM tree.

The build fails if the file in the base image is not the vLLM v0.31.0 original.
That is the failure mode that let an earlier RTX image ship labeled as patched
when it was still stock.
"""

from __future__ import annotations

import hashlib
import pathlib
import shutil

import vllm

ROOT = pathlib.Path(vllm.__file__).resolve().parent
PATCHES = pathlib.Path("/tmp/aeon-patches")

# relpath under the vllm package -> (stock sha256, patch filename, patched sha256)
EXPECTED = {
    "model_executor/kernels/linear/mxfp6/emulation.py": (
        "6b8f7d0879872c81dae8303728776d711fc9e44b4a94b471613ff42cd3f74b68",
        "mxfp6_emulation.py",
        "b64b67ac772176e64bfa7fd086510eaa05b159c76ce006a0fe263e949b186d30",
    ),
    "v1/worker/gpu/model_states/mamba_hybrid.py": (
        "1ddd7dc921d74789e77d3319dc167820ad7fc47ed51d8d9a35ba3938f5c6a079",
        "mamba_hybrid.py",
        "b141a130ba611d8336e788e961cb0f2aca9cc0fbd24bf81a99aa9ba521c470ea",
    ),
}


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    for rel, (stock, name, patched) in EXPECTED.items():
        dest = ROOT / rel
        src = PATCHES / name
        got = sha256(dest)
        if got != stock:
            raise SystemExit(
                f"refusing to patch {dest}: sha256 {got} is not the v0.31.0 original {stock}"
            )
        if sha256(src) != patched:
            raise SystemExit(f"patch file {src} is not the reviewed AEON copy")
        shutil.copyfile(src, dest)
        if sha256(dest) != patched:
            raise SystemExit(f"copy failed for {dest}")
        print(f"patched {rel}")


if __name__ == "__main__":
    main()
