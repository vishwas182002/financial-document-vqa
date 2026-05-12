# Financial Document Visual Question Answering

**Benchmarking and adapting vision-language models for SEC 10-K document VQA**

## Overview

This project evaluates and adapts vision-language transformer architectures for Visual Question Answering (VQA) on financial documents. I benchmark four representative document VQA models on both generic document benchmarks (DocVQA) and a custom financial test set curated from real SEC 10-K filings, measuring the domain gap and identifying failure modes specific to financial document understanding.

## Key Findings

- **63-83% performance drop** when moving from generic documents to financial documents across all four architectures
- **LayoutLMv3 outperforms OCR-free models** (Pix2Struct, Donut) on financial documents, reversing the DocVQA ranking - explicit OCR with spatial encoding handles dense financial tables better than pixel-based reading
- **Near-zero numerical reasoning capability** across all models (0.000-0.017 ANLS on 65 numerical questions)
- **Negative transfer from generic fine-tuning** - LoRA fine-tuning on DocVQA decreased financial document performance from 0.137 to 0.084 ANLS
- **Domain-specific fine-tuning works** - LoRA fine-tuning on just 307 financial QA pairs improved performance by 22% over zero-shot (0.137 to 0.168 ANLS), demonstrating that the right domain data matters more than a larger generic dataset
- **Structural failures dominate** (46-65% of errors) - models often find the relevant text region but select the wrong table row or column
- **Two-stage DePlot+LLM pipeline outperforms end-to-end models** on chart-specific questions (0.441 vs. 0.314 ANLS on ChartQA)

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

| Setting | Financial ANLS | Training Data |
|---------|---------------|---------------|
| Pix2Struct (zero-shot) | 0.137 | - |
| Pix2Struct (Generic LoRA) | 0.084 | DocVQA (5,000 examples) |
| Pix2Struct (Financial LoRA) | 0.168 | SEC 10-K filings (307 QA pairs) |

Generic fine-tuning caused **negative transfer** (-39%). Domain-specific fine-tuning on just 307 financial QA pairs **improved performance by 22%** over zero-shot, demonstrating that domain-specific training data is essential.

*Note: Zero-shot and Generic LoRA results are on the full 397-pair test set. Financial LoRA results are on a held-out subset of 90 pairs from 3 unseen companies (JPM, MSFT, WMT) to prevent data leakage.*

### DePlot + LLM vs Pix2Struct on ChartQA

| Model | ChartQA ANLS |
|-------|-------------|
| Pix2Struct | 0.314 |
| DePlot + FLAN-T5 | 0.441 |

## Custom Financial VQA Test Set

I curated a benchmark of **79 images** from 10-K annual reports of **10 major U.S. public companies**, annotated with **397 question-answer pairs** across four categories:

- **Extractive** (243): Direct value lookup from tables
- **Layout Understanding** (72): Spatial reasoning over document structure
- **Numerical Reasoning** (65): Arithmetic operations over financial data
- **Chart Interpretation** (17): Visual reasoning over charts and graphs

**Companies:** Apple, Amazon, Goldman Sachs, Microsoft, Tesla, Walmart, Johnson & Johnson, ExxonMobil, Bank of America, JPMorgan Chase

**Sectors:** Technology, Finance, Healthcare, Energy, Retail

All images are real screenshots from SEC EDGAR filings. All answers are manually verified against the source documents, with numerical reasoning answers calculator-checked.

## Models Evaluated

| Model | Type | Parameters | Key Mechanism |
|-------|------|-----------|---------------|
| **Pix2Struct** | OCR-free encoder-decoder | 282M | Screenshot parsing pretraining |
| **Donut** | OCR-free encoder-decoder | 259M | Swin Transformer + BART decoder |
| **LayoutLMv3** | OCR-dependent encoder | 126M | Text + image + spatial layout |
| **OCR + RoBERTa** | Two-stage pipeline | ~125M | Tesseract OCR + extractive QA |
| **DePlot + FLAN-T5** | Two-stage pipeline | 530M | Chart-to-table + LLM reasoning |

## Project Structure

```text
financial-document-vqa/
├── notebooks/
│   ├── 01_setup_and_eda.ipynb
│   ├── 02_baseline_ocr_roberta.ipynb
│   ├── 03_zeroshot_pix2struct.ipynb
│   ├── 04_zeroshot_donut.ipynb
│   ├── 05_zeroshot_layoutlmv3.ipynb
│   ├── 06_deplot_llm_pipeline.ipynb
│   ├── 07_financial_test_set.ipynb
│   ├── 08_financial_evaluation_final.ipynb
│   ├── 09_lora_finetuning.ipynb
│   ├── 10_error_analysis.ipynb
│   ├── 11_gradio_demo.ipynb
│   └── 12_financial_finetuning.ipynb
├── data/
│   └── financial_test/
│       ├── annotations/
│       └── images/samples/
├── outputs/
├── requirements.txt
├── LICENSE
└── README.md
```

## Data Access

### Included in this repository

- **Financial VQA test set annotations** (`data/financial_test/annotations/`) - the full 397 QA pairs in JSON format
- **Financial document images** (`data/financial_test/images/samples/`) - all 79 SEC filing screenshots
- **Result JSONs and charts** (`outputs/`) - experimental outputs and visualizations

### Not included

The following datasets/checkpoints must be downloaded or regenerated separately:

- **DocVQA**: Download from [pixparse/docvqa-single-page-questions](https://huggingface.co/datasets/pixparse/docvqa-single-page-questions)
- **ChartQA**: Download via `datasets.load_dataset("HuggingFaceM4/ChartQA")`
- **TAT-QA**: Clone from [NExTplusplus/TAT-QA](https://github.com/NExTplusplus/TAT-QA)
- **Model checkpoints**: LoRA fine-tuned weights are not included. Re-run `09_lora_finetuning.ipynb` or `12_financial_finetuning.ipynb` to reproduce.

## Reproduction

1. Clone this repository.
2. Install Python dependencies:

```bash
pip install -r requirements.txt
```

3. Install system dependencies if running OCR or PDF/image extraction workflows:
   - **Tesseract OCR** is required for OCR-based experiments using `pytesseract`.
   - **Poppler** may be required by `pdf2image`, depending on your environment.

4. Open notebooks in Google Colab with GPU runtime.
5. Run notebooks in numerical order.
6. For notebooks `01`-`05`, download DocVQA/ChartQA datasets first and save them as described in notebook `01`.
7. For notebooks `08` onward, the financial test set is included in this repository.

Optional note: data-collection notebooks may require additional browser/data-collection tools such as `requests`, `selenium`, or `playwright` if rerunning the original collection workflow.

## Technology Stack

- **Framework:** PyTorch + HuggingFace Transformers
- **Fine-tuning:** HuggingFace PEFT / LoRA (rank=16, alpha=32)
- **OCR:** Tesseract
- **Demo:** Gradio
- **Compute:** Google Colab Pro (T4 GPU)

## Evaluation Metrics

- **ANLS** (Average Normalized Levenshtein Similarity): Standard DocVQA metric tolerating minor OCR/spelling variations
- **Exact Match**: Strict accuracy requiring exact string match after normalization

## Limitations

- The financial test set contains 79 images with 397 QA pairs. Chart interpretation has only 17 questions, limiting statistical power for that category.
- Financial LoRA fine-tuning used only 307 training pairs; a larger financial training set would likely yield stronger improvements.
- Only base-size open VQA models were evaluated. Larger models or closed-source multimodal LLMs such as GPT-4V, Gemini, or Claude may perform differently.
- No human analyst performance baseline was established for the financial test set.
- The best fine-tuned result remains far below a deployment-ready threshold, especially for numerical reasoning.

## Future Work

- Curate larger financial-specific training datasets for more effective domain adaptation
- Evaluate larger vision-language models and multimodal LLMs
- Expand the financial test set with more chart interpretation and numerical reasoning questions
- Establish human performance baselines
- Investigate retrieval-augmented and structure-aware approaches for multi-page financial documents
- Fine-tune LayoutLMv3, the best zero-shot financial performer, with domain-specific data

## References

- Mathew et al. (2021). DocVQA: A Dataset for VQA on Document Images. WACV.
- Kim et al. (2022). OCR-Free Document Understanding Transformer (Donut). ECCV.
- Lee et al. (2023). Pix2Struct: Screenshot Parsing as Pretraining. ICML.
- Huang et al. (2022). LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking. ACM MM.
- Liu et al. (2023). DePlot: One-shot Visual Language Reasoning by Plot-to-Table Translation. ACL Findings.
- Masry et al. (2022). ChartQA: A Benchmark for Question Answering about Charts. ACL Findings.
- Zhu et al. (2021). TAT-QA: A Question Answering Benchmark on a Hybrid of Tabular and Textual Content in Finance. ACL.
- Hu et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. ICLR.
- Srivastava et al. (2025). Enhancing Financial VQA using Intermediate Structured Representations. arXiv.
- Gurari et al. (2018). VizWiz Grand Challenge. CVPR.

## Project Context

Neural Networks and Deep Learning  
University of Colorado Boulder, Spring 2026

## Author

**Vishwas K.**  
M.S. Computer Science, University of Colorado Boulder
