Below is a **ready-to-use `README.md`** for your uploaded Jupyter Notebook.
You can copy-paste this directly into a `README.md` file in your GitHub repository.

---

# DeepSeek-OCR Gradio Application

This project demonstrates how to build an **OCR (Optical Character Recognition) application** using **DeepSeek-OCR**, **vLLM**, and **Gradio**.
The app allows users to upload images and extract text efficiently using a large multimodal model optimized for GPU usage.

---

## Features

* Image-based OCR using **DeepSeek-OCR**
* Fast inference with **vLLM**
* GPU-optimized configuration (T4 / 16GB compatible)
* Interactive **Gradio** web interface
* Clean, modular, and beginner-friendly code

---

## Tech Stack

* **Python**
* **DeepSeek-OCR**
* **vLLM**
* **Gradio**
* **PyTorch**
* **Pillow (PIL)**

---

## Project Structure

```
├── DeepSeek_OCR.ipynb   # Main notebook containing model loading & Gradio UI
├── README.md           # Project documentation
```

---

## Installation

Install all required dependencies:

```bash
pip install -U gradio vllm pillow torch
```

> **Note:**
>
> * A CUDA-enabled GPU is recommended
> * Tested on **NVIDIA T4 (16GB VRAM)**

---

## Model Configuration

The model is loaded only once and stored globally for efficient reuse:

```python
llm_model = LLM(
    model="deepseek-ai/DeepSeek-OCR",
    enable_prefix_caching=False,
    mm_processor_cache_gb=0,
    trust_remote_code=True,
    gpu_memory_utilization=0.9,
    max_model_len=4096
)
```

This setup ensures:

* Stable GPU memory usage
* Fast inference
* Compatibility with limited VRAM

---

## ▶How to Run

1. Open the notebook:

   ```bash
   jupyter notebook
   ```

2. Run all cells in order.

3. Launch the Gradio interface:

   * Upload an image
   * View extracted text instantly

---

## Example Use Cases

* Document digitization
* Invoice & receipt extraction
* Book & notes OCR
* Multimodal AI experiments

---

## Key Learning Outcomes

* How to use **DeepSeek-OCR** with vLLM
* Serving multimodal models efficiently
* Building simple AI web apps with **Gradio**
* GPU memory optimization techniques

---

## Notes

* Model loading may take **1–2 minutes** initially
* Large images may take longer to process
* Make sure your GPU drivers and CUDA version are properly set up

---

## Acknowledgements

* DeepSeek AI Team
* Open-source community
* vLLM & Gradio contributors

---

## License
This project is intended for **educational and research purposes**.




