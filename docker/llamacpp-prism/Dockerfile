# llama.cpp-Server aus dem PrismML-Fork.
#
# Ternary-Bonsai (Q2_0_g128) braucht die Hybrid-Attention-Kernels des Forks —
# das offizielle ggml-org/llama.cpp-Image lädt das GGUF nicht. Verhält sich
# sonst wie das offizielle Server-Image: Entrypoint llama-server,
# Konfiguration zur Laufzeit über LLAMA_ARG_*-Env-Vars.
#
# Build + Push:
#   docker build -t ghcr.io/strfry/llamacpp-prism:prism-b9594-server-cuda \
#     docker/llamacpp-prism
#   docker push ghcr.io/strfry/llamacpp-prism:prism-b9594-server-cuda

ARG CUDA_VERSION=12.8.1

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04 AS build

# Fork-Release pinnen (Tag aus github.com/PrismML-Eng/llama.cpp/releases)
ARG FORK_TAG=prism-b9594-38c66ad
# RTX 4000 Ada = sm_89; mehrere Archs als Liste: "86;89;90"
ARG CUDA_ARCHS=89

RUN apt-get update && apt-get install -y --no-install-recommends \
        git cmake build-essential libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch ${FORK_TAG} \
        https://github.com/PrismML-Eng/llama.cpp /src

WORKDIR /src
# Statisch gegen ggml linken (ein Binary), CURL für -hf/LLAMA_ARG_HF_REPO
RUN cmake -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHS} \
        -DGGML_NATIVE=OFF \
        -DBUILD_SHARED_LIBS=OFF \
        -DLLAMA_CURL=ON \
    && cmake --build build --target llama-server -j

FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /src/build/bin/llama-server /app/llama-server

ENV LLAMA_ARG_HOST=0.0.0.0
EXPOSE 8080

# start-period großzügig: erster Start lädt das GGUF (7.2 GB) von HF
HEALTHCHECK --interval=30s --timeout=5s --start-period=15m \
    CMD curl -sf http://localhost:${LLAMA_ARG_PORT:-8080}/health || exit 1

ENTRYPOINT ["/app/llama-server"]
