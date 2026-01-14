# Voice-Changer on macOS (Apple Silicon)

This guide explains how to run Voice-Changer natively on macOS with Apple Silicon (M1/M2/M3) using MPS (Metal Performance Shaders) for GPU acceleration.

## Prerequisites

- macOS with Apple Silicon (M1, M2, M3, etc.)
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/download)
- Xcode Command Line Tools (install with `xcode-select --install`)

## Quick Start

```bash
# 1. Navigate to server directory
cd server

# 2. Run setup script (creates conda environment)
./setup-mac.sh

# 3. Start the server
./start-mac.sh
```

The web UI will be available at `https://localhost:18888`

> **Note:** Accept the self-signed certificate warning in your browser on first access.

## Manual Setup

If you prefer to set up manually:

```bash
# Create conda environment
conda create -n vcclient python=3.10 -y
conda activate vcclient

# Install PyTorch with MPS support
pip install torch==2.0.1 torchaudio==2.0.2

# Install dependencies
cd server
pip install -r requirements-mac.txt

# Start server
python MMVCServerSIO.py -p 18888 --https true --model_dir model_dir
```

## GPU Acceleration

Voice-Changer automatically uses Apple's Metal Performance Shaders (MPS) for GPU acceleration on Apple Silicon Macs.

### Verify MPS is Working

```bash
conda activate vcclient
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

When the server starts, check the logs for:
```
VoiceChangerV2 Initialized (GPU_NUM(cuda):0, mps_enabled:True, ...)
```

### GPU Usage

- **PyTorch models**: Use MPS (Metal) for GPU acceleration
- **ONNX models**: Run on CPU (ONNX Runtime doesn't support Metal directly)

## Pretrained Models

Pretrained models are automatically downloaded on first run from HuggingFace:

| Model | Size | Purpose |
|-------|------|---------|
| hubert_base.pt | ~360MB | Feature extraction |
| content_vec_500.onnx | ~100MB | Feature extraction (ONNX) |
| rmvpe.pt | ~140MB | Pitch extraction |
| crepe_onnx_full.onnx | ~80MB | Pitch extraction (ONNX) |
| nsf_hifigan | ~55MB | Neural vocoder |

Models are stored in `server/pretrain/`.

## RVC Models

Place your RVC voice models in `server/model_dir/`. Supported formats:
- `.pth` files (PyTorch)
- `.onnx` files (ONNX - recommended for faster inference)

## Configuration

### Change Port

```bash
PORT=8080 ./start-mac.sh
```

Or pass directly:
```bash
python MMVCServerSIO.py -p 8080 --https true --model_dir model_dir
```

### Disable HTTPS

```bash
python MMVCServerSIO.py -p 18888 --https false --model_dir model_dir
```

## Known Limitations

1. **VoRAS model not supported** - VoRAS inference is not available on macOS
2. **RVC runs on CPU** - RVC inference is forced to run on CPU due to MPS compatibility issues. This is slower than GPU but produces correct audio output.
3. **ONNX runs on CPU** - ONNX Runtime uses CPU; CoreML provider not currently supported
4. **No half-precision (FP16)** - CPU inference uses FP32
5. **First run is slow** - Downloads ~500MB+ of pretrained models

## RVC Model Notes

- **"nono" models** (no f0): These models don't support pitch shifting. They convert voice timbre only.
- **f0 models**: Support pitch shifting via the "tran" parameter. May sound more natural.
- **Index files** (`.index`): Use `indexRatio > 0` to enable index-based voice similarity matching for better quality.

## Troubleshooting

### "MPS available: False"

1. Ensure you're on Apple Silicon (M1/M2/M3), not Intel Mac
2. Update macOS to latest version
3. Reinstall PyTorch: `pip install --force-reinstall torch==2.0.1`

### Server crashes with MPS errors

Try running with CPU fallback:
```bash
python MMVCServerSIO.py -p 18888 --https true --model_dir model_dir
# Then set GPU to -1 in the web UI
```

### "Module not found" errors

Ensure the conda environment is activated:
```bash
conda activate vcclient
```

### Slow performance

1. Use ONNX models when available (`.onnx` files)
2. Reduce chunk size in web UI settings
3. Close other GPU-intensive applications

### Port already in use

```bash
# Find process using port
lsof -i :18888

# Kill process
kill -9 <PID>

# Or use different port
PORT=8080 ./start-mac.sh
```

## Performance Tips

1. **Use ONNX RMVPE** - Set pitch extractor to `rmvpe_onnx` for better CPU utilization
2. **Optimize chunk size** - Start with default, adjust based on latency vs quality
3. **Monitor Activity Monitor** - Check GPU usage under "GPU History"

## Comparison: Mac vs CUDA

| Feature | Mac (MPS) | NVIDIA (CUDA) |
|---------|-----------|---------------|
| PyTorch inference | MPS GPU | CUDA GPU |
| ONNX inference | CPU | CUDA GPU |
| Half-precision | No (FP32) | Yes (FP16) |
| VoRAS model | No | Yes |
| Performance | Good | Best |

## Files Created

- `server/requirements-mac.txt` - Mac-specific dependencies
- `server/setup-mac.sh` - Environment setup script
- `server/start-mac.sh` - Server startup script
