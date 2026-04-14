# Financial Document Visual Question Answering

**Multimodal Deep Learning for Financial Document VQA**

Neural Networks and Deep Learning | University of Colorado Boulder | Spring 2026

## Overview

This project evaluates and adapts vision-language transformer architectures for Visual Question Answering (VQA) on financial documents. We benchmark four state-of-the-art models on both generic document benchmarks (DocVQA) and a custom financial test set curated from real SEC 10-K filings, measuring the domain gap and identifying failure modes specific to financial document understanding.

## Key Findings

- **63–83% performance drop** when moving from generic documents to financial documents across all four architectures
- **LayoutLMv3 outperforms OCR-free models** (Pix2Struct, Donut) on financial documents, reversing the DocVQA ranking — explicit OCR with spatial encoding handles dense financial tables better than pixel-based reading
- **Zero numerical reasoning capability** across all models (0% ANLS on 65 numerical questions)
- **Negative transfer from generic fine-tuning** — LoRA fine-tuning on DocVQA decreased financial document performance from 0.137 to 0.084 ANLS
- **Structural failures dominate** (46–65% of errors) — models find text but pick the wrong table row/column

## Results

### DocVQA vs Financial Documents (ANLS)

| Model | DocVQA | Financial | Domain Gap |
|-------|--------|-----------|------------|
| OCR + RoBERTa (baseline) | 0.326 | 0.120 | -63% |
| LayoutLMv3 | 0.468 | 0.154 | -67% |
| Donut | 0.627 | 0.105 | -83% |
| Pix2Struct | 0.655 | 0.137 | -79% |

### Per Question Type on Financial Documents (ANLS)

| Model | Extractive | Layout | Numerical | Chart |
|-------|-----------|--------|-----------|-------|
| OCR + RoBERTa | 0.128 | 0.185 | 0.000 | 0.183 |
| LayoutLMv3 | 0.192 | 0.167 | 0.009 | 0.103 |
| Donut | 0.156 | 0.051 | 0.000 | 0.000 |
| Pix2Struct | 0.195 | 0.056 | 0.017 | 0.118 |

### DePlot + LLM vs Pix2Struct on ChartQA

| Model | ChartQA ANLS |
|-------|-------------|
| Pix2Struct | 0.314 |
| DePlot + FLAN-T5 | 0.441 |

## Custom Financial VQA Test Set

We curate a gold-standard benchmark of **79 images** from 10-K annual reports of **10 major U.S. public companies** (FY2025), annotated with **397 question-answer pairs** across four categories:

- **Extractive** (243): Direct value lookup from tables
- **Layout Understanding** (72): Spatial reasoning over document structure
- **Numerical Reasoning** (65): Arithmetic operations over financial data
- **Chart Interpretation** (17): Visual reasoning over charts and graphs

**Companies:** Apple, Amazon, Goldman Sachs, Microsoft, Tesla, Walmart, Johnson & Johnson, ExxonMobil, Bank of America, JPMorgan Chase

**Sectors:** Technology, Finance, Healthcare, Energy, Retail

## Models Evaluated

| Model | Type | Parameters | Key Mechanism |
|-------|------|-----------|---------------|
| **Pix2Struct** | OCR-free encoder-decoder | 282M | Screenshot parsing pretraining |
| **Donut** | OCR-free encoder-decoder | 259M | Swin Transformer + BART decoder |
| **LayoutLMv3** | OCR-dependent encoder | 126M | Text + image + spatial layout |
| **OCR + RoBERTa** | Two-stage pipeline | ~125M | Tesseract OCR + extractive QA |
| **DePlot + FLAN-T5** | Two-stage pipeline | 530M | Chart-to-table + LLM reasoning |

## Project Structure

```
FinDocVQA/
├── notebooks/
│   ├── 01_setup_and_eda.ipynb
│   ├── 02_baseline_ocr_roberta.ipynb
│   ├── 03_zeroshot_pix2struct.ipynb
│   ├── 04_zeroshot_donut.ipynb
│   ├── 05_zeroshot_layoutlmv3.ipynb
│   ├── 07_deplot_llm_pipeline.ipynb
│   ├── 08_financial_test_set.ipynb
│   ├── 09_financial_evaluation.ipynb
│   ├── 11_financial_evaluation_final.ipynb
│   ├── 12_lora_finetuning.ipynb
│   ├── 13_error_analysis.ipynb
│   └── 14_gradio_demo.ipynb
├── data/
│   ├── docvqa/
│   ├── chartqa/
│   ├── tatqa/
│   └── financial_test/
│       ├── images/manual/        # 79 SEC filing screenshots
│       └── annotations/          # 397 verified QA pairs
├── models/
│   └── pix2struct_lora_best/     # LoRA fine-tuned checkpoint
├── outputs/                      # All result JSONs and charts
└── configs/
```

## Technology Stack

- **Framework:** PyTorch + HuggingFace Transformers
- **Fine-Tuning:** HuggingFace PEFT (LoRA, rank=16, alpha=32)
- **OCR:** Tesseract
- **Demo:** Gradio
- **Compute:** Google Colab Pro (T4 GPU)

## Evaluation Metrics

- **ANLS** (Average Normalized Levenshtein Similarity): Standard DocVQA metric tolerating minor OCR/spelling variations
- **Exact Match**: Strict accuracy requiring exact string match
- **Numerical Accuracy**: Custom metric for numerical reasoning (1% tolerance)

## Demo

A Gradio-based demo allows users to upload financial document images and ask questions, with answers from Pix2Struct, LayoutLMv3, and OCR+RoBERTa displayed side by side.

## References

- Mathew et al. (2021). DocVQA: A Dataset for VQA on Document Images. WACV.
- Kim et al. (2022). OCR-Free Document Understanding Transformer (Donut). ECCV.
- Lee et al. (2023). Pix2Struct: Screenshot Parsing as Pretraining. ICML.
- Huang et al. (2022). LayoutLMv3: Pre-training for Document AI. ACM MM.
- Liu et al. (2023). DePlot: Plot-to-Table Translation. ACL Findings.
- Masry et al. (2022). ChartQA: Question Answering about Charts. ACL Findings.
- Zhu et al. (2021). TAT-QA: Tabular and Textual QA in Finance. ACL.
- Hu et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. ICLR.
- Gurari et al. (2018). VizWiz Grand Challenge. CVPR.

## Author

**Vishwas K.**
M.S. Computer Science, University of Colorado Boulder

## Course

Neural Networks and Deep Learning (Spring 2026)
Professor Danna Gurari
