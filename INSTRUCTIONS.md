# DA6401 Assignment 3 — Complete Instructions

---

## Files you have

| File | What it does |
|---|---|
| `model.py` | Every transformer piece (attention, encoder, decoder, scheduler, loss) + `infer()` |
| `dataset.py` | Downloads Multi30k, builds vocab, makes dataloaders |
| `train.py` | Trains the model and runs all 5 W&B experiments |
| `kaggle_notebook.ipynb` | Paste this into Kaggle and run top to bottom |

---

## Step 1 — Get your files into Kaggle

1. Go to **kaggle.com → New Notebook**
2. On the right panel click **"Add data" → "Upload"**
3. Upload these 3 files as a dataset: `model.py`, `dataset.py`, `train.py`
4. Kaggle will put them at `/kaggle/input/your-dataset-name/`
5. In your notebook first cell, copy them to the working directory:

```python
import shutil, os
for f in ['model.py', 'dataset.py', 'train.py']:
    shutil.copy(f'/kaggle/input/YOUR-DATASET-NAME/{f}', f'./{f}')
```

6. Then paste the rest of `kaggle_notebook.ipynb` cells below that.

**Enable T4 x2:** Settings (right sidebar) → Accelerator → **GPU T4 x2**

---

## Step 2 — Run training (follow this order)

Run **Step 5 first** (base model, ~1.5 hrs). This gives you the checkpoint you submit.

Then run the experiments one at a time in separate notebook sessions so you don't hit the 12-hour Kaggle limit. Each experiment flag you can run independently:

```
--experiments noam    # ~2 hrs  (Exp 2.1)
--experiments scale   # ~1.5 hrs (Exp 2.2)
--experiments pe      # ~2 hrs  (Exp 2.4)
--experiments smooth  # ~2 hrs  (Exp 2.5)
```

Exp 2.3 (attention heatmap) runs automatically at the end of `--experiments base`.

---

## Step 3 — Upload weights to Google Drive

1. After training finishes, `best_model.pt` appears in the Kaggle output panel (right sidebar → Output)
2. Download it to your computer
3. Upload it to **Google Drive**
4. Right-click the file → **Share** → change to **"Anyone with the link"**
5. Copy the link. It looks like:
   ```
   https://drive.google.com/file/d/1aBcDeFgHiJkLmNoPqRsTuVwXyZ/view
   ```
6. The part between `/d/` and `/view` is your **File ID**:
   ```
   1aBcDeFgHiJkLmNoPqRsTuVwXyZ
   ```
7. Open `model.py`, find this line near the top:
   ```python
   GDRIVE_FILE_ID = "PASTE_YOUR_GOOGLE_DRIVE_FILE_ID_HERE"
   ```
   Replace with your actual ID:
   ```python
   GDRIVE_FILE_ID = "1aBcDeFgHiJkLmNoPqRsTuVwXyZ"
   ```

---

## Step 4 — Test the autograder flow

Run **Step 9** in the notebook (the final cell). It must print a real English sentence with zero errors. This is literally what the autograder runs:

```python
model = Transformer().to(device)   # downloads weights, loads vocab inside __init__
model.eval()
english_sentence = model.infer(german_sentence)
```

If this works, you are ready to submit.

---

## Step 5 — Gradescope submission

1. Submit `model.py`, `dataset.py`, `train.py` (with the real Drive ID in `model.py`)
2. Submit your **public W&B report link**
3. Do **not** upload `best_model.pt` — the code downloads it automatically

---

## W&B Report checklist

| Section | What to include |
|---|---|
| 2.1 Noam vs Fixed LR | Overlay train loss curves for both runs. Explain warmup prevents large early gradients in QK softmax. |
| 2.2 Scaling factor | Overlay Q and K gradient norm curves for first 1000 steps. Explain vanishing gradient without scaling. |
| 2.3 Attention heads | The heatmap image (auto-logged). Identify which head attends locally vs long-range. |
| 2.4 PE vs Learned | BLEU comparison table. Explain sinusoidal can extrapolate beyond training length; learned cannot. |
| 2.5 Label smoothing | Prediction confidence plot. Explain smoothing as regularizer that hurts training perplexity but helps generalisation. |

---

## Why Pre-LayerNorm (for your report)

The code uses Pre-LN (norm before sublayer, not after). Reason: in Post-LN the residual stream early in training is dominated by unnormalized outputs, causing gradient explosion in deep stacks. Pre-LN normalizes before the attention/FFN operation so gradients flow more evenly through all layers from epoch 1.

---

## Common problems

| Problem | Fix |
|---|---|
| `OSError: de_core_news_sm not found` | Run `!python -m spacy download de_core_news_sm` |
| `gdown` fails downloading weights | Make sure Drive sharing is set to "Anyone with the link" |
| W&B run not public | Go to wandb.ai → your project → Settings → make it public |
| CUDA out of memory | Reduce `batch_size` in `load_data()` from 256 to 128 |
