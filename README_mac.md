# Voice-Changer on macOS (Apple Silicon)

This guide explains how to run Voice-Changer natively on macOS with Apple Silicon (M1/M2/M3/M4).

> **Note:** RVC inference runs on CPU due to MPS compatibility issues. While MPS is detected, the pipeline forces CPU execution to ensure correct audio output.

## Prerequisites

- macOS with Apple Silicon (M1, M2, M3, M4, etc.)
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

## Manual Setup (Tested & Verified)

These steps have been verified to work with pinned dependencies for reproducibility:

```bash
# 1. Create conda environment with Python 3.10
conda create -n vcclient python=3.10 -y

# 2. Activate the environment
conda activate vcclient

# 3. Install PyTorch with MPS support first (must be installed before other deps)
pip install torch==2.0.1 torchaudio==2.0.2

# 4. Navigate to server directory and install dependencies
cd server
pip install -r requirements-mac.txt

# 5. Start the server
PYTORCH_ENABLE_MPS_FALLBACK=1 python MMVCServerSIO.py -p 18888 --https true --model_dir model_dir
```

### Dependency Pinning

The `requirements-mac.txt` file contains all dependencies with exact pinned versions extracted from a working environment. This ensures reproducibility across different systems. Key dependencies include:

- `torch==2.0.1` - PyTorch with MPS support
- `onnxruntime==1.16.0` - CPU version (no GPU on Mac)
- `fairseq` - Custom fork for macOS compatibility
- `numpy==1.23.5` - Compatible with all audio processing libraries

## GPU Acceleration

While MPS (Metal Performance Shaders) is detected, RVC inference is forced to run on CPU to ensure correct audio output. This is a known compatibility issue with MPS and RVC models.

### Verify Environment

```bash
conda activate vcclient
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

When the server starts, you'll see these log messages confirming CPU fallback:
```
VoiceChangerV2 Initialized (GPU_NUM(cuda):0, mps_enabled:True, onnx_device:CPU)
[PipelineGenerator] MPS detected - forcing CPU for entire pipeline (known MPS compatibility issue)
```

### Current Status

- **RVC models**: Run on CPU (MPS produces distorted audio)
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

1. **RVC runs on CPU** - RVC inference is forced to CPU due to MPS compatibility issues. MPS produces incorrect/distorted audio ("ahhh" sound). CPU inference is slower but produces correct output.
2. **VoRAS model not supported** - VoRAS inference is not available on macOS
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

### Distorted "ahhhh" audio output

This was a known MPS issue that has been fixed. The pipeline now automatically forces CPU execution when MPS is detected. If you still experience this:
1. Ensure you're using the latest code with the MPS fixes
2. Check the logs for: `[PipelineGenerator] MPS detected - forcing CPU for entire pipeline`
3. If the message doesn't appear, the fix may not be applied correctly

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

| Feature | Mac (Apple Silicon) | NVIDIA (CUDA) |
|---------|---------------------|---------------|
| RVC inference | CPU (forced) | CUDA GPU |
| ONNX inference | CPU | CUDA GPU |
| Half-precision | No (FP32) | Yes (FP16) |
| VoRAS model | No | Yes |
| Performance | Slower (CPU) | Best (GPU) |

> **Why CPU on Mac?** MPS backend produces numerically incorrect results for RVC models, resulting in distorted "ahhhh" audio. CPU inference is slower but produces correct voice conversion.

## Files Created

- `server/requirements-mac.txt` - Mac-specific dependencies (all versions pinned)
- `server/setup-mac.sh` - Environment setup script
- `server/start-mac.sh` - Server startup script
- `Claude.md` - Comprehensive codebase documentation
