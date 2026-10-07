# AEON vLLM Ultimate for ROCm x86.
# Early access, pending community testing. Python-only overlay on the official
# vLLM v0.31.0 ROCm image. No HIP or AITER rebuild.
FROM vllm/vllm-openai-rocm:v0.31.0@sha256:749f6f3f944f12af49966ac523c1f8573e4b229594b541954bd5f879b4496b1f

LABEL org.opencontainers.image.title="aeon-vllm-ultimate-rocm" \
      org.opencontainers.image.description="Early access, pending community testing. AEON vLLM Ultimate for ROCm x86 on vLLM v0.31.0 with MXFP6 dequant-once and the DFlash2 prefix-cache fix." \
      org.opencontainers.image.source="https://github.com/AEON-7/vllm-ultimate-rocm" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.vendor="AEON-7" \
      ai.aeon.vendor="rocm" \
      ai.aeon.arch="x86" \
      ai.aeon.target="gfx90a,gfx942,gfx950,gfx1100,gfx1101,gfx1150,gfx1151,gfx1200,gfx1201" \
      ai.aeon.vllm.tag="v0.31.0" \
      ai.aeon.vllm.commit="db9527a46873454610df6dbedf79a36d6bf1a7f6" \
      ai.aeon.base.digest="sha256:749f6f3f944f12af49966ac523c1f8573e4b229594b541954bd5f879b4496b1f" \
      ai.aeon.channel="early-access" \
      ai.aeon.patches="C1-mamba-hybrid-55601,C2-mxfp6-dequant-once"

COPY patches/ /tmp/aeon-patches/
COPY install_patches.py /tmp/aeon-patches/install_patches.py
RUN python3 /tmp/aeon-patches/install_patches.py && rm -rf /tmp/aeon-patches

COPY aeon-entrypoint.sh /usr/local/bin/aeon-entrypoint
RUN sed -i 's/\r$//' /usr/local/bin/aeon-entrypoint && chmod 0755 /usr/local/bin/aeon-entrypoint

ENV AEON_MXFP6_DEQUANT_AT_LOAD=1 \
    VLLM_ROCM_USE_AITER=0 \
    AEON_IMAGE_CHANNEL=early-access

ENTRYPOINT ["/usr/local/bin/aeon-entrypoint"]
