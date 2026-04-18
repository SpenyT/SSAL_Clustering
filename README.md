# SSAL Clustering

## Setup & Run

### 1. Create a virtual environment

**Windows**
```bash
python -m venv venv
```

**Mac/Linux**
```bash
python3 -m venv venv
```

### 2. Activate the virtual environment

**Windows**
```bash
venv\Scripts\activate
```

**Mac/Linux**
```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

For GPU-accelerated UMAP via RAPIDS cuML (requires a CUDA-capable GPU):

```bash
pip install ".[gpu]"
```

> **Note:** cuML requires Linux or WSL2. On Windows or macOS the CPU fallback (`umap-learn`) is used automatically.

### 4. Run

```bash
python src/main.py
```

### Deactivate when done

```bash
deactivate
```
