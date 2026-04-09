# Python 3.11 for quflow (compatible with 3.9–3.11)
FROM python:3.11-bookworm

# System deps: FFTW (for ducc0/numba), ffmpeg (animations), cartopy (proj, geos)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libfftw3-dev \
    ffmpeg \
    libproj-dev \
    libgeos-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps from quflow requirements (excluding the quflow package line)
COPY requirements.txt /tmp/requirements.txt
RUN sed -i '/^quflow/d' /tmp/requirements.txt && \
    pip install --no-cache-dir -r /tmp/requirements.txt && \
    pip install --no-cache-dir jupyter ipywidgets

# Editable install of quflow is done in postCreateCommand so the workspace mount is available
WORKDIR /workspace
