FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04
# FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
# FROM nvidia/cuda:12.0.1-cudnn8-runtime-ubuntu22.04

RUN apt-get update \
        && apt-get install -y python3-pip espeak gosu libsndfile1-dev emacs git \
        && apt-get clean \
        && rm -rf /var/lib/apt/lists/*

WORKDIR /
ADD dummy /

RUN git clone https://github.com/w-okada/voice-changer.git -b v.1.5.3.13

RUN chmod 0777 /voice-changer/server
COPY ./patches/FairseqHubert.py /voice-changer/server/voice_changer/RVC/embedder/FairseqHubert.py

WORKDIR /voice-changer/server
RUN pip install numpy==1.23.5 \
        && pip install pyworld==0.3.3 --no-build-isolation \
        && pip install -r requirements.txt

# --- Force PyTorch with sm_120 support (RTX 5070 Ti) ---
RUN pip uninstall -y torch torchvision torchaudio || true && \
    pip install --upgrade pip && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

ADD /setup.sh  /voice-changer/server
ADD /exec.sh  /voice-changer/server


ENTRYPOINT ["/bin/bash", "setup.sh"]