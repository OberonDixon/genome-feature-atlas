# Local LLM Server (vLLM)

## One-time setup

```bash
conda create -n vllm python=3.12
conda activate vllm
pip install 'vllm>=0.22.1'
```

Set HuggingFace cache location in `~/.bashrc`:
```bash
export HF_HOME=/path/to/hf_weights
```

The model weights (~67GB BF16) are cached at `$HF_HOME` and downloaded
automatically on first boot. Subsequent boots load from cache.

## Booting the server

```bash
conda activate vllm

vllm serve Qwen/Qwen3.6-35B-A3B \
  --quantization fp8 \
  --tensor-parallel-size 2 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.80 \
  --enforce-eager \
  --port 8000
```

**Startup takes 5–10 minutes** — the server loads ~67GB of weights into RAM,
quantizes to FP8, and distributes across both GPUs. It's ready when you see:

```
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

## Verifying the server is up

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen3.6-35B-A3B", "messages": [{"role": "user", "content": "Hello"}]}'
```

## Shutting down properly

**Important:** Ctrl+C alone does not reliably free GPU memory. Worker processes
can linger and hold VRAM, causing NCCL errors on the next boot.

Full shutdown procedure:

```bash
# 1. Find all vLLM-related processes
ps aux | grep -E "vllm|VLLM" | grep -v grep
nvidia-smi  # worker PIDs appear as VLLM::Worker_TP0, VLLM::Worker_TP1

# 2. Kill everything
pkill -f vllm
# If workers persist, kill by PID explicitly:
kill -9   ...

# 3. Verify GPUs are free (both should show <500MiB used)
nvidia-smi
```

Only proceed with a new `vllm serve` once `nvidia-smi` confirms both GPUs
are clear.