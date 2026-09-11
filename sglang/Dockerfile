# Custom Dynamo SGLang runtime for GLM-5.2-FP8.
#
# Why: GLM-5.2-FP8 (GlmMoeDsaForCausalLM, DeepSeek-style sparse attention) requires
# SGLang >= 0.5.13.post1 (per the model card). No published Dynamo sglang-runtime
# image ships that yet:
#   1.2.1-cuda13       -> SGLang 0.5.11        (q_b_proj shape mismatch at load)
#   1.3.0-dev.1-cuda13 -> SGLang 0.5.12.post1  (still too old)
# So we take the newest Dynamo dev image (keeps the Dynamo runtime + NIXL + sgl_kernel
# wiring) and bump only SGLang to 0.5.13.post1.
#
# Build:  docker build -t glm52-dynamo-sglang:0.5.13post1 -f Dockerfile .
ARG BASE=nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.3.0-dev.1-cuda13
FROM ${BASE}

ARG SGLANG_VERSION=0.5.13.post1

# If your build host is behind a proxy, pip in the build container needs it.
# Pass via --build-arg (see serve.sh). Build with --network=host so the proxy is
# reachable from the container.
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY

# Proxy scoped to this RUN only (not ENV) so it isn't baked into the runtime image.
# --no-deps: keep the base image's torch / flashinfer / sgl-kernel / nixl pins intact
# (0.5.12->0.5.13 is a point release; runtime import is verified after build).
# SGLang 0.5.13.post1 requires sglang-kernel >= 0.4.3 (base image ships 0.4.2.post2).
# The 0.4.3 cu13 wheel is ABI-compatible with the base torch (verified: common_ops imports).
RUN https_proxy=${HTTPS_PROXY} http_proxy=${HTTP_PROXY} no_proxy=${NO_PROXY} \
    pip install --no-deps --break-system-packages \
        "sglang==${SGLANG_VERSION}" "sglang-kernel==0.4.3" \
    && python3 -c "import importlib.metadata as m; \
sg=m.version('sglang'); sk=m.version('sglang-kernel'); \
assert sg.startswith('0.5.13'), sg; assert sk >= '0.4.3', sk; \
print('sglang', sg, '| sglang-kernel', sk)"
# Note: importing sgl_kernel loads GPU-only CUDA ops (common_ops), so we check
# versions via package metadata here; the ops are verified at container run time.

# --- Crash diagnostics: let py-spy dump a hung rank -------------------------
# SGLang's watchdog handler shells out to py-spy to dump every rank's Python
# stack when the scheduler stalls. On 2026-09-03 all 8 ranks hit the 300 s
# watchdog and EVERY dump failed with "Permission Denied", so the hang was
# un-diagnosable. Two separate things block it, and both must be fixed:
#
#   1. The host runs Yama ptrace_scope=1, so a process may only ptrace its own
#      DESCENDANTS. The handler dumps sibling ranks, which is not that.
#      CAP_SYS_PTRACE bypasses Yama.
#   2. The container runs as uid 1000, and Docker sets no ambient capabilities.
#      `cap_add: SYS_PTRACE` in docker-compose.yml therefore only puts the cap
#      in the BOUNDING set (CapBnd) -- CapPrm/CapEff stay 0 and it grants
#      nothing on its own. It is necessary but not sufficient.
#
# A file capability on the binary closes the gap: on exec the kernel computes
# permitted = file_permitted & bounding, so py-spy (and only py-spy) picks the
# cap up as uid 1000 -- which is why the compose-side cap_add must stay.
# Verified: same binary, uid 1000, non-descendant target -> "Permission Denied"
# without this, full stack dump with it.
#
# setcap(8) is not in the image (no libcap2-bin) and apt would need the build
# proxy, so write the xattr directly -- it is a fixed 20-byte struct.
USER root
RUN python3 -c "import struct, os; \
CAP_SYS_PTRACE = 19; \
VFS_CAP_REVISION_2 = 0x02000000; \
VFS_CAP_FLAGS_EFFECTIVE = 0x000001; \
blob = struct.pack('<IIIII', VFS_CAP_REVISION_2 | VFS_CAP_FLAGS_EFFECTIVE, 1 << CAP_SYS_PTRACE, 0, 0, 0); \
os.setxattr('/usr/local/bin/py-spy', b'security.capability', blob)" \
    && python3 -c "import os; \
v = os.getxattr('/usr/local/bin/py-spy', b'security.capability'); \
assert v == bytes.fromhex('0100000200000800000000000000000000000000'), v.hex(); \
print('py-spy cap_sys_ptrace+ep set')"
# Restore the base image's runtime user (uid 1000). Everything above this line
# is the only root window in the build.
USER dynamo
