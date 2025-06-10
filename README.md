# On Finetuning Tabular Foundation Models

:scroll: [arXiv](https://arxiv.org)
&nbsp; :books: [Other tabular DL projects](https://github.com/yandex-research/rtdl)

This repository provides tools for fine-tuning TabPFN v2 models on tabular datasets.

> Foundation models are an emerging research direction in tabular deep learning. Notably, TabPFNv2 recently claimed superior performance over traditional GBDT-based methods on small-scale datasets using an in-context learning paradigm, which does not adapt model parameters to target datasets. However, the optimal finetuning approach for adapting tabular foundational models, and how this adaptation reshapes their internal mechanisms, remains underexplored. While prior works studied finetuning for earlier foundational models, inconsistent findings and TabPFNv2's unique architecture necessitate fresh investigation. To address these questions, we first systematically evaluate various finetuning strategies on diverse datasets. Our findings establish full finetuning as the most practical solution for TabPFNv2 in terms of time-efficiency and effectiveness. We then investigate how finetuning alters TabPFNv2's inner mechanisms, drawing an analogy to retrieval-augmented models. We reveal that the success of finetuning stems from the fact that after gradient-based adaptation, the dot products of the query-representations of test objects and the key-representations of in-context training objects more accurately reflect their target similarity. This improved similarity allows finetuned TabPFNv2 to better approximate target dependency by appropriately weighting relevant in-context samples, improving the retrieval-based prediction logic. From the practical perspective, we managed to finetune TabPFNv2 on datasets with up to 50K objects, observing performance improvements on almost all tasks. More precisely, on academic datasets with I.I.D. splits, finetuning allows TabPFNv2 to achieve state-of-the-art results, while on datasets with gradual temporal shifts and rich feature sets, TabPFNv2 is less stable and prior methods remain better.

## Quick Start

### Prerequisites

1. **Install dependencies**
   ```bash
   # Install uv if you haven't already
   pip install uv
   
   # For GPU support (NVIDIA)
   uv sync --extra cu124
   
   # For CPU only (not recommended)
   uv sync
   ```

2. **Download model checkpoints**
   ```bash
   # Download TabPFN v2 models
   wget https://huggingface.co/Prior-Labs/TabPFN-v2-reg/resolve/main/tabpfn-v2-regressor.ckpt?download=true -O tabpfn-v2-regressor.ckpt
   wget https://huggingface.co/Prior-Labs/TabPFN-v2-class/resolve/main/tabpfn-v2-classifier.ckpt?download=true -O tabpfn-v2-classifier.ckpt
   ```

3. **Download datasets**
   ```bash
   mkdir -p local data
   wget https://huggingface.co/datasets/rototoHF/tabm-data/resolve/main/data.tar -O local/tabm-data.tar.gz
   tar -xvf local/tabm-data.tar.gz -C data
   ```

### Running the code

Execute a minimal run with:
```bash
uv run python bin/tabpfnv2_finetune.py exp/full-finetune/adult/evaluation/0.toml --force
```

## 📁 Project Structure

- `bin/` - Training and evaluation scripts
- `exp/` - Experiment configurations and results
- `data/` - Dataset directory (created after download)
- `lib/` - Common utilities and tools

## 🔧 Configuration

Experiments are configured using TOML files located in the `exp/` directory. Each configuration specifies:
- Dataset path and preprocessing
- Model hyperparameters
- Training settings
- Evaluation metrics

## 📊 Results

After training, results are saved in the same directory as the configuration file:
- `report.json` - Evaluation metrics
- Model checkpoints
- Training logs

## 📝 Citation

If you find this work useful, please cite:
```bibtex
@article{yourname2024finetuning,
  title={On Finetuning Tabular Foundation Models},
  author={Ivan Rubachev and Akim Kotelnikov and Nikolay Kartashev and Artem Babenko},
  journal={arXiv preprint arXiv:2024.TODO},
  year={2024}
}
```
