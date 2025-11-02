# Installation with GPU Support (PyTorch) via UV

By default, this project installs **PyTorch (CPU)**. You can optionally enable **GPU (CUDA)** acceleration for NVIDIA systems.


## Quick Start

### CPU Installation (Default)

```bash
uv sync
```

Installs PyTorch with CPU support on all platforms.

### GPU Installation

1. **Check your CUDA version:**

   ```bash
   nvidia-smi
   ```

   Example output: `CUDA Version: 12.8`

2. **Reinstall with GPU support:**

   ```bash
   # Remove lock to re-resolve dependencies
   rm uv.lock # For windows: Remove-Item uv.lock 

   # Install with matching CUDA version
   uv sync --extra pytorch-cu128
   ```

   Replace `cu128` with your version (`cu130`, `cu126`, `cu121`, etc.)

   **Note:** If your CUDA version isn’t supported “out of the box,” you’ll need to **manually add the corresponding extra** to your `pyproject.toml`, then remove the lock file and reinstall with the correct version.


## Switching Between CPU and GPU

```bash
# Switch to GPU (example: CUDA 12.8)
rm uv.lock # For windows: Remove-Item uv.lock 
uv sync --extra pytorch-cu128

# Switch back to CPU
rm uv.lock # For windows: Remove-Item uv.lock 
uv sync
```


## Common Extras

| Purpose                 | Example Command                             |
| ----------------------- | ------------------------------------------- |
| CPU (default)           | `uv sync`                                   |
| GPU (CUDA 12.8)         | `uv sync --extra pytorch-cu128`             |
| GPU + Dev tools         | `uv sync --extra pytorch-cu128 --extra dev` |
| GPU + All optional deps | `uv sync --extra pytorch-cu128 --extra all` |


## Verify Installation

**CPU (default):**

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# Example → 2.8.0+cpu False
```

**GPU:**

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Example → 2.8.0+cu128 True NVIDIA GeForce RTX 4090
```


## Platform Notes

| Platform    | CPU | GPU           |
| ----------- | --- | ------------- |
| **Windows** | YES   | YES (CUDA only) |
| **Linux**   | YES   | YES (CUDA only) |
| **macOS**   | YES   | NO (CPU only)  |

**macOS users:** PyTorch GPU wheels aren’t available. Use `uv sync` (CPU).


## Troubleshooting

| Issue                       | Solution                                                            |
| --------------------------- | ------------------------------------------------------------------- |
| **Wrong CUDA version**      | Remove `uv.lock` and reinstall with correct `--extra pytorch-cuXXX` |
| **GPU not detected**        | Update NVIDIA drivers and verify CUDA version matches PyTorch       |
| **Multiple GPU extras**     | Only specify one GPU extra per install                              |
| **“No solution found”**     | Remove `uv.lock` and `.uv` cache, then retry                        |
| **macOS GPU install fails** | Use CPU only (`uv sync`)                                            |
