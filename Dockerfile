FROM nvcr.io/nvidia/cuda:13.0.1-devel-ubuntu24.04

ARG RC_STABLE_AUDIO_TOOLS_REF=f1f13af77cb9bc34dd6eb001b2430b3c85375ea2

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_PREFER_BINARY=1
ENV PORT=7860
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_ANALYTICS_ENABLED=False
ENV HF_HUB_DISABLE_PROGRESS_BARS=1
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV FOUNDATION_MODELS_ROOT=/models
ENV FOUNDATION_OUTPUT_DIR=/outputs

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    ffmpeg \
    git \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL -o /tmp/miniforge.sh \
      https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh && \
    bash /tmp/miniforge.sh -b -p /opt/conda && \
    rm -f /tmp/miniforge.sh

ENV PATH="/opt/conda/envs/foundation/bin:/opt/conda/bin:${PATH}"

RUN conda create -y -n foundation python=3.10 pip && \
    conda clean -afy

RUN pip install --no-cache-dir --upgrade pip "setuptools<81" wheel && \
    pip install --no-cache-dir \
      torch==2.9.1 \
      torchvision==0.24.1 \
      torchaudio==2.9.1 \
      --index-url https://download.pytorch.org/whl/cu130

RUN git clone https://github.com/RoyalCities/RC-stable-audio-tools.git /opt/RC-stable-audio-tools && \
    cd /opt/RC-stable-audio-tools && \
    git checkout "${RC_STABLE_AUDIO_TOOLS_REF}" && \
    sed -i "s/scipy==1.8.1/scipy>=1.10,<1.14/" setup.py && \
    pip install --no-cache-dir -e . && \
    rm -rf /root/.cache/pip

COPY launch.py /opt/launch.py

WORKDIR /opt/RC-stable-audio-tools

RUN mkdir -p /models /outputs

EXPOSE 7860

CMD ["python", "/opt/launch.py"]
