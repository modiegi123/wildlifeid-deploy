FROM python:3.11-slim

WORKDIR /app

# CPU-only torch wheel - far smaller than the default CUDA build, and this Space has no GPU
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV HOST=0.0.0.0
ENV PORT=7860
# Full L-384 model needs ~800MB+ RAM just for weights - too much for most free hosting tiers.
# T-224 (768-d embeddings, separately recalibrated - see Siphesihle/thresholds.json) fits instead.
ENV WILDLIFEID_BASELINE_MODEL=BVRA/MegaDescriptor-T-224
EXPOSE 7860

CMD ["python", "run.py"]
