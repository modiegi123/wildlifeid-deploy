FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV HOST=0.0.0.0
ENV PORT=7860
# PyTorch alone needs ~300MB RAM just to import, before any model is loaded - too much for this
# host's free tier. Runs a pre-exported ONNX copy of the baseline model via onnxruntime instead
# (see Siphesihle/app.py: OnnxEmbedder), downloaded from the Hub on first use.
ENV WILDLIFEID_ONNX_REPO=modiegimorudu/megadescriptor-t224-onnx
EXPOSE 7860

CMD ["python", "run.py"]
