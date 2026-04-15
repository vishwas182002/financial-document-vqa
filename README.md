# Financial Document Visual Question Answering

**Multimodal Deep Learning for Financial Document VQA**

Neural Networks and Deep Learning | University of Colorado Boulder | Spring 2026

## Overview

This project evaluates and adapts vision-language transformer architectures for Visual Question Answering (VQA) on financial documents. We benchmark four state-of-the-art models on both generic document benchmarks (DocVQA) and a custom financial test set curated from real SEC 10-K filings, measuring the domain gap and identifying failure modes specific to financial document understanding.

## Key Findings

- **63–83% performance drop** when moving from generic documents to financial documents across all four architectures
- **LayoutLMv3 outperforms OCR-free models** (Pix2Struct, Donut) on financial documents, reversing the DocVQA ranking — explicit OCR with spatial encoding handles dense financial tables better than pixel-based reading
- **Zero numerical reasoning capability** across all models (0% ANLS on 65 numerical questions)
- **Negative transfer from generic fine-tuning** — LoRA fine-tuning on DocVQA decreased financial document performance from 0.137 to 0.084 ANLS, demonstrating that domain-specific training data is essential
- **Structural failures dominate** (46–65% of errors) — models find text but pick the wrong table row/column
- **Two-stage DePlot+LLM pipeline outperforms end-to-end models** on chart-specific questions (0.441 vs 0.314 ANLS on ChartQA)

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

### LoRA Fine-Tuning Results

| Setting | Financial ANLS |
|---------|---------------|
| Pix2Struct (zero-shot) | 0.137 |
| Pix2Struct (LoRA on DocVQA) | 0.084 |

Fine-tuning on generic DocVQA data caused **negative transfer**, reducing performance by 39%. This demonstrates that domain-specific financial training data is required for effective adaptation.

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

All images are real screenshots from SEC EDGAR filings. All answers are manually verified against the source documents with numerical reasoning answers calculator-checked.

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
financial-document-vqa/
├── notebooks/
│   ├── 01_setup_and_eda.ipynb          # Environment setup + dataset EDA
│   ├── 02_baseline_ocr_roberta.ipynb   # OCR + RoBERTa baseline on DocVQA
│   ├── 03_zeroshot_pix2struct.ipynb    # Pix2Struct zero-shot on DocVQA
│   ├── 04_zeroshot_donut.ipynb         # Donut zero-shot on DocVQA
│   ├── 05_zeroshot_layoutlmv3.ipynb    # LayoutLMv3 zero-shot on DocVQA
│   ├── 07_deplot_llm_pipeline.ipynb    # DePlot + FLAN-T5 on ChartQA
│   ├── 08_financial_test_set.ipynb     # Financial test set collection
│   ├── 09_financial_evaluation_final.ipynb  # All models on financial test set
│   ├── 10_lora_finetuning.ipynb        # LoRA fine-tuning + evaluation
│   ├── 11_error_analysis.ipynb         # Failure mode taxonomy
│   └── 12_gradio_demo.ipynb            # Interactive demo app
├── data/
│   └── financial_test/
│       ├── annotations/                # 397 verified QA pairs (JSON)
│       └── images/samples/             # Sample SEC filing screenshots
├── outputs/                            # Result JSONs + visualization PNGs
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## Data Access

### Included in this repository
- **Financial VQA test set annotations** (`data/financial_test/annotations/`) — the full 397 QA pairs in JSON format
- **Sample financial document images** (`data/financial_test/images/samples/`) — all 79 SEC filing screenshots
- **All result JSONs and charts** (`outputs/`) — complete experimental results

### Not included (too large for GitHub)
The following datasets must be downloaded separately to reproduce the training and DocVQA/ChartQA evaluations:

- **DocVQA**: Download from [pixparse/docvqa-single-page-questions](https://huggingface.co/datasets/pixparse/docvqa-single-page-questions) on HuggingFace
- **ChartQA**: Download via `datasets.load_dataset("HuggingFaceM4/ChartQA")`
- **TAT-QA**: Clone from [GitHub](https://github.com/NExTplusplus/TAT-QA)
- **Model checkpoints**: LoRA fine-tuned weights are not included. Re-run `10_lora_finetuning.ipynb` to reproduce.

### How to reproduce
1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Open notebooks in Google Colab with GPU runtime
4. Run notebooks in numerical order
5. For notebooks 01-05, download DocVQA/ChartQA datasets first and save to Google Drive as described in notebook 01
6. For notebooks 09+, the financial test set is included in this repository

## Technology Stack

- **Framework:** PyTorch + HuggingFace Transformers
- **Fine-Tuning:** HuggingFace PEFT (LoRA, rank=16, alpha=32)
- **OCR:** Tesseract
- **Demo:** Gradio
- **Compute:** Google Colab Pro (T4 GPU)

## Evaluation Metrics

- **ANLS** (Average Normalized Levenshtein Similarity): Standard DocVQA metric tolerating minor OCR/spelling variations
- **Exact Match**: Strict accuracy requiring exact string match

## Limitations

- The financial test set contains 79 images with 397 QA pairs. Chart interpretation has only 17 questions, which limits statistical significance for that category.
- LoRA fine-tuning was performed on generic DocVQA data, not financial-specific training data. Domain-specific fine-tuning is left as future work.
- Only base-size models were evaluated. Larger models (Pix2Struct-large) or foundation models (GPT-4V, Gemini) may perform differently.
- No human performance baseline was established for the financial test set.

## Future Work

- Curate financial-specific training data for domain-adapted fine-tuning
- Evaluate larger vision-language models and multimodal LLMs
- Expand the financial test set with more chart interpretation questions
- Establish human performance baselines
- Investigate retrieval-augmented approaches for multi-page financial documents

## References

- Mathew et al. (2021). DocVQA: A Dataset for VQA on Document Images. WACV.
- Kim et al. (2022). OCR-Free Document Understanding Transformer (Donut). ECCV.
- Lee et al. (2023). Pix2Struct: Screenshot Parsing as Pretraining. ICML.
- Huang et al. (2022). LayoutLMv3: Pre-training for Document AI. ACM MM.
- Liu et al. (2023). DePlot: Plot-to-Table Translation. ACL Findings.
- Masry et al. (2022). ChartQA: Question Answering about Charts. ACL Findings.
- Zhu et al. (2021). TAT-QA: Tabular and Textual QA in Finance. ACL.
- Hu et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. ICLR.
- Srivastava et al. (2025). Enhancing Financial VQA using Intermediate Structured Representations. arXiv.
- Gurari et al. (2018). VizWiz Grand Challenge. CVPR.

## Author

**Vishwas K.**
M.S. Computer Science, University of Colorado Boulder

## Course

Neural Networks and Deep Learning (Spring 2026)
Professor Danna Gurari
