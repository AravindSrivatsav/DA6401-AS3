import argparse
import os
import random
import shutil

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import wandb

matplotlib.use("Agg")

from dataset import load_data
from lr_scheduler import NoamScheduler
from model import EOS_IDX, PAD_IDX, SOS_IDX, LabelSmoothingLoss, Transformer


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(model, path, src_vocab, trg_vocab, cfg):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "src_vocab": src_vocab,
            "trg_vocab": trg_vocab,
            "config": cfg,
        },
        path,
    )


def train_epoch(model, loader, optimizer, scheduler, criterion, device, log_grads=False, step_offset=0):
    model.train()
    total_loss = 0
    step = step_offset
    grad_log = []

    for src, trg in loader:
        src, trg = src.to(device), trg.to(device)
        trg_in = trg[:, :-1]
        trg_out = trg[:, 1:]

        logits = model(src, trg_in)
        B, T, V = logits.shape
        loss = criterion(logits.reshape(B * T, V), trg_out.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        if log_grads and step < 1000:
            q_norm, k_norm, count = 0.0, 0.0, 0
            inner_model = model.module if isinstance(model, nn.DataParallel) else model
            for layer in inner_model.encoder.layers:
                if layer.self_attn.w_q.weight.grad is not None:
                    q_norm += layer.self_attn.w_q.weight.grad.norm().item() ** 2
                    k_norm += layer.self_attn.w_k.weight.grad.norm().item() ** 2
                    count += 1
            if count:
                grad_log.append({"step": step, "q_grad_norm": q_norm ** 0.5, "k_grad_norm": k_norm ** 0.5})

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        step += 1

    return total_loss / len(loader), step, grad_log


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    for src, trg in loader:
        src, trg = src.to(device), trg.to(device)
        trg_in = trg[:, :-1]
        trg_out = trg[:, 1:]
        logits = model(src, trg_in)
        B, T, V = logits.shape
        total_loss += criterion(logits.reshape(B * T, V), trg_out.reshape(-1)).item()
    return total_loss / len(loader)


@torch.no_grad()
def compute_bleu(model, records, n=400):
    import sacrebleu
    hyps, refs = [], []
    model.eval()
    for r in records[:n]:
        hyps.append(model.infer(r["de"]))
        refs.append(r["en"])
    return sacrebleu.corpus_bleu(hyps, [refs], tokenize="13a").score


@torch.no_grad()
def avg_confidence(model, loader, device, n_batches=20):
    model.eval()
    total, count = 0.0, 0
    for i, (src, trg) in enumerate(loader):
        if i >= n_batches:
            break
        src, trg = src.to(device), trg.to(device)
        trg_in = trg[:, :-1]
        trg_out = trg[:, 1:]
        logits = model(src, trg_in)
        probs = torch.softmax(logits, dim=-1)
        B, T, V = probs.shape
        flat_probs = probs.reshape(B * T, V)
        flat_target = trg_out.reshape(-1)
        mask = flat_target != PAD_IDX
        correct_probs = flat_probs[mask].gather(1, flat_target[mask].unsqueeze(1)).squeeze(1)
        total += correct_probs.sum().item()
        count += correct_probs.numel()
    return total / count if count else 0.0


@torch.no_grad()
def visualize_attention(model, sentence_de, trg_vocab, device):
    model.eval()
    tokens_de = [t.text.lower() for t in model.spacy_de.tokenizer(sentence_de)]
    src_ids = [SOS_IDX] + [model.src_vocab.get(t, 0) for t in tokens_de] + [EOS_IDX]
    src = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(device)
    src_mask = model.make_src_mask(src)
    model.encoder(src, src_mask)

    last_layer = model.encoder.layers[-1]
    attn = last_layer.self_attn.attn_weights[0].cpu().numpy()
    labels = ["<sos>"] + tokens_de + ["<eos>"]

    num_heads = attn.shape[0]
    ncols = 4
    nrows = (num_heads + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = axes.flatten()

    for h in range(num_heads):
        ax = axes[h]
        im = ax.imshow(attn[h], cmap="viridis", aspect="auto")
        ax.set_title(f"Head {h + 1}", fontsize=9)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
        ax.set_yticklabels(labels, fontsize=6)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for h in range(num_heads, len(axes)):
        axes[h].axis("off")

    plt.suptitle("Encoder Last Layer Per-Head Attention", fontsize=11)
    plt.tight_layout()
    path = "attention_heads.png"
    plt.savefig(path, dpi=120)
    plt.close()
    return path


def run(
    run_name,
    train_loader,
    val_loader,
    test_records,
    src_vocab,
    trg_vocab,
    device,
    epochs=25,
    warmup_steps=3000,
    use_noam=True,
    fixed_lr=1e-4,
    smoothing=0.1,
    pe_type="sinusoidal",
    use_scale=True,
    log_grads=False,
    wandb_project="da6401-a3",
):
    cfg = dict(
        d_model=256, num_heads=8, num_layers=3, d_ff=512,
        max_len=150, dropout=0.1, pe_type=pe_type, use_scale=use_scale,
    )

    wandb.init(project=wandb_project, name=run_name, config={**cfg, "smoothing": smoothing, "noam": use_noam})

    model = Transformer(src_vocab=src_vocab, trg_vocab=trg_vocab, **cfg).to(device)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    inner = model.module if isinstance(model, nn.DataParallel) else model
    print(f"[{run_name}] params: {count_params(inner):,}")

    criterion = LabelSmoothingLoss(len(trg_vocab), smoothing=smoothing)
    optimizer = torch.optim.Adam(model.parameters(), lr=0 if use_noam else fixed_lr, betas=(0.9, 0.98), eps=1e-9)

    scheduler = None
    if use_noam:
        scheduler = NoamScheduler(optimizer, d_model=cfg["d_model"], warmup_steps=warmup_steps)

    best_bleu = 0.0
    ckpt_path = f"best_{run_name}.pt"
    global_step = 0

    for epoch in range(1, epochs + 1):
        train_loss, global_step, grad_log = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, device,
            log_grads=log_grads, step_offset=global_step
        )
        val_loss = eval_epoch(model, val_loader, criterion, device)

        log = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        log["lr"] = scheduler.get_lr() if scheduler else fixed_lr

        for g in grad_log:
            wandb.log({"step": g["step"], "q_grad_norm": g["q_grad_norm"], "k_grad_norm": g["k_grad_norm"]})

        if epoch % 5 == 0 or epoch == epochs:
            bleu = compute_bleu(inner, test_records, n=300)
            log["val_bleu"] = bleu
            conf = avg_confidence(model, val_loader, device)
            log["pred_confidence"] = conf
            print(f"Epoch {epoch:3d} | train={train_loss:.4f} val={val_loss:.4f} BLEU={bleu:.2f} conf={conf:.4f}")
            if bleu > best_bleu:
                best_bleu = bleu
                save_checkpoint(inner, ckpt_path, src_vocab, trg_vocab, cfg)
        else:
            print(f"Epoch {epoch:3d} | train={train_loss:.4f} val={val_loss:.4f}")

        wandb.log(log)

    wandb.log({"best_bleu": best_bleu})
    wandb.finish()
    return ckpt_path, best_bleu


def main(args):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | GPUs: {torch.cuda.device_count()}")

    print("Loading data...")
    train_loader, val_loader, test_loader, src_vocab, trg_vocab, spacy_de, spacy_en, test_records = load_data(batch_size=256)
    print(f"src vocab: {len(src_vocab)} | trg vocab: {len(trg_vocab)}")

    common = dict(
        train_loader=train_loader, val_loader=val_loader, test_records=test_records,
        src_vocab=src_vocab, trg_vocab=trg_vocab, device=device,
        wandb_project=args.wandb_project,
    )

    if "all" in args.experiments or "noam" in args.experiments:
        print("\n=== EXP 1: Noam Scheduler ===")
        run("noam_scheduler", use_noam=True, epochs=25, **common)
        run("fixed_lr_1e4", use_noam=False, fixed_lr=1e-4, epochs=25, **common)

    if "all" in args.experiments or "scale" in args.experiments:
        print("\n=== EXP 2: Scaling Factor Ablation ===")
        run("with_scale", use_scale=True, epochs=20, log_grads=True, **common)
        run("no_scale", use_scale=False, epochs=20, log_grads=True, **common)

    if "all" in args.experiments or "base" in args.experiments:
        print("\n=== EXP BASE: Full Training + Attention Rollout ===")
        best_path, best_bleu = run("base_model", use_noam=True, epochs=args.epochs, **common)
        print(f"Best BLEU: {best_bleu:.2f}  saved -> {best_path}")

        ckpt = torch.load(best_path, map_location="cpu")
        vis_model = Transformer(src_vocab=src_vocab, trg_vocab=trg_vocab,
                                d_model=256, num_heads=8, num_layers=3, d_ff=512).to(device)
        vis_model.load_state_dict(ckpt["model_state_dict"])
        sample_de = test_records[0]["de"]
        attn_img = visualize_attention(vis_model, sample_de, trg_vocab, device)

        wandb.init(project=args.wandb_project, name="attention_rollout")
        wandb.log({"attention_heads": wandb.Image(attn_img, caption=sample_de)})
        wandb.finish()

        shutil.copy(best_path, "best_model.pt")
        print("Saved best_model.pt upload this to Google Drive.")

    if "all" in args.experiments or "pe" in args.experiments:
        print("\n=== EXP 4: Positional Encoding ===")
        run("sinusoidal_pe", pe_type="sinusoidal", epochs=25, **common)
        run("learned_pe", pe_type="learned", epochs=25, **common)

    if "all" in args.experiments or "smooth" in args.experiments:
        print("\n=== EXP 5: Label Smoothing ===")
        run("smooth_0.1", smoothing=0.1, epochs=25, **common)
        run("smooth_0.0", smoothing=0.0, epochs=25, **common)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", nargs="+", default=["all"],
                        choices=["all", "base", "noam", "scale", "pe", "smooth"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--wandb_project", type=str, default="da6401-a3")
    args = parser.parse_args()
    main(args)
