# onnx-speech-language-detection

```bash
sudo apt-get update \
&& sudo apt-get upgrade -y \
&& sudo apt-get install -y --no-install-recommends \
    gcc \
    curl \
    wget \
    sudo \
    python3-all-dev \
    python-is-python3 \
    python3-pip \
    ffmpeg \
    portaudio19-dev \
&& pip install -U pip \
    requests==2.31.0 \
    psutil==5.9.5 \
    tqdm==4.65.0 \
    more-itertools==8.10.0 \
    ffmpeg-python==0.2.0 \
    transformers==4.29.2 \
    soundfile==0.12.1 \
    SpeechRecognition==3.10.0 \
    PyAudio==0.2.13 \
    onnx==1.16.2 \
    onnxruntime==1.18.1 \
    onnxsim==0.4.30 \
    protobuf==3.20.3 \
    h5py==3.7.0
```

```bash
curl -o whisper/assets/tiny_decoder_11.onnx https://github.com/PINTO0309/onnx-speech-language-detection/releases/download/1.0/tiny_decoder_11.onnx
curl -o whisper/assets/tiny_encoder_11.onnx https://github.com/PINTO0309/onnx-speech-language-detection/releases/download/1.0/tiny_encoder_11.onnx
```
