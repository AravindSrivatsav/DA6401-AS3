import math
import os

import gdown
import spacy
import torch
import torch.nn as nn
import torch.nn.functional as F

GDRIVE_FILE_ID = "1R7bsbgXm9lhba5r-NYA8WKHcUa1Xqs-v"

PAD_IDX = 1
UNK_IDX = 0
SOS_IDX = 2
EOS_IDX = 3


class ScaledDotProductAttention(nn.Module):
    def __init__(self, use_scale=True):
        super().__init__()
        self.use_scale = use_scale

    def forward(self, q, k, v, mask=None):
        scores = torch.matmul(q, k.transpose(-2, -1))
        if self.use_scale:
            scores = scores / math.sqrt(q.size(-1))
        if mask is not None:
            if mask.dtype == torch.bool:
                scores = scores.masked_fill(mask, -1e9)
            else:
                scores = scores.masked_fill(mask == 0, -1e9)
        weights = F.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)
        return torch.matmul(weights, v), weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1, use_scale=True):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        self.attn_fn = ScaledDotProductAttention(use_scale)
        self.dropout = nn.Dropout(dropout)
        self.attn_weights = None

    def _split(self, x):
        B = x.size(0)
        return x.view(B, -1, self.num_heads, self.d_k).transpose(1, 2)

    def forward(self, q, k, v, mask=None):
        B = q.size(0)
        q = self._split(self.w_q(q))
        k = self._split(self.w_k(k))
        v = self._split(self.w_v(v))
        out, self.attn_weights = self.attn_fn(q, k, v, mask)
        out = out.transpose(1, 2).contiguous().view(B, -1, self.num_heads * self.d_k)
        return self.dropout(self.w_o(out))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, : x.size(1)])


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.embed = nn.Embedding(max_len, d_model)
        nn.init.normal_(self.embed.weight, std=0.02)

    def forward(self, x):
        pos = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        return self.dropout(x + self.embed(pos))


def _make_pe(pe_type, d_model, max_len, dropout):
    if pe_type == "learned":
        return LearnedPositionalEncoding(d_model, max_len, dropout)
    return PositionalEncoding(d_model, max_len, dropout)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, use_scale=True):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout, use_scale)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        n = self.norm1(x)
        x = x + self.drop(self.self_attn(n, n, n, mask))
        x = x + self.drop(self.ff(self.norm2(x)))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, use_scale=True):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout, use_scale)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout, use_scale)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        n = self.norm1(x)
        x = x + self.drop(self.self_attn(n, n, n, trg_mask))
        n = self.norm2(x)
        x = x + self.drop(self.cross_attn(n, enc_out, enc_out, src_mask))
        x = x + self.drop(self.ff(self.norm3(x)))
        return x


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_len, dropout, pe_type, use_scale):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)
        self.pe = _make_pe(pe_type, d_model, max_len, dropout)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout, use_scale) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.scale = math.sqrt(d_model)

    def forward(self, src, mask):
        x = self.pe(self.embed(src) * self.scale)
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_len, dropout, pe_type, use_scale):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)
        self.pe = _make_pe(pe_type, d_model, max_len, dropout)
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout, use_scale) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.scale = math.sqrt(d_model)

    def forward(self, trg, enc_out, src_mask, trg_mask):
        x = self.pe(self.embed(trg) * self.scale)
        for layer in self.layers:
            x = layer(x, enc_out, src_mask, trg_mask)
        return self.norm(x)


class LabelSmoothingLoss(nn.Module):
    def __init__(self, vocab_size, smoothing=0.1, pad_idx=PAD_IDX):
        super().__init__()
        self.smoothing = smoothing
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx

    def forward(self, logits, target):
        log_p = F.log_softmax(logits, dim=-1)
        with torch.no_grad():
            eps = self.smoothing / (self.vocab_size - 2)
            dist = torch.full_like(log_p, eps)
            dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
            dist[:, self.pad_idx] = 0.0
        non_pad = target != self.pad_idx
        loss = -(dist * log_p).sum(dim=-1)
        return loss[non_pad].mean()


class NoamScheduler:
    def __init__(self, optimizer, d_model, warmup_steps, factor=1.0):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.factor = factor
        self.step_num = 0

    def step(self):
        self.step_num += 1
        lr = self._lr(self.step_num)
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def _lr(self, s):
        return self.factor * (self.d_model ** -0.5) * min(s ** -0.5, s * self.warmup_steps ** -1.5)

    def get_lr(self):
        if self.step_num == 0:
            return 0.0
        return self._lr(self.step_num)


class Transformer(nn.Module):
    def __init__(
        self,
        d_model=256,
        num_heads=8,
        num_layers=3,
        d_ff=512,
        max_len=150,
        dropout=0.1,
        pe_type="sinusoidal",
        use_scale=True,
        src_vocab=None,
        trg_vocab=None,
    ):
        super().__init__()

        weights = None
        if src_vocab is None or trg_vocab is None:
            ckpt = self._fetch_ckpt()
            src_vocab = ckpt["src_vocab"]
            trg_vocab = ckpt["trg_vocab"]
            cfg = ckpt.get("config", {})
            d_model = cfg.get("d_model", d_model)
            num_heads = cfg.get("num_heads", num_heads)
            num_layers = cfg.get("num_layers", num_layers)
            d_ff = cfg.get("d_ff", d_ff)
            max_len = cfg.get("max_len", max_len)
            pe_type = cfg.get("pe_type", pe_type)
            use_scale = cfg.get("use_scale", use_scale)
            weights = ckpt["model_state_dict"]

        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.max_len = max_len
        self.idx2trg = {v: k for k, v in trg_vocab.items()}

        self.encoder = Encoder(len(src_vocab), d_model, num_heads, d_ff, num_layers, max_len, dropout, pe_type, use_scale)
        self.decoder = Decoder(len(trg_vocab), d_model, num_heads, d_ff, num_layers, max_len, dropout, pe_type, use_scale)
        self.fc_out = nn.Linear(d_model, len(trg_vocab))

        self._xavier_init()
        self._load_spacy()

        if weights is not None:
            self.load_state_dict(weights)

    def _xavier_init(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _load_spacy(self):
        try:
            self.spacy_de = spacy.load("de_core_news_sm")
        except OSError:
            self.spacy_de = spacy.blank("de")

    def _fetch_ckpt(self, path="best_model.pt"):
        if not os.path.exists(path):
            gdown.download(
                id=GDRIVE_FILE_ID,
                output=path,
                quiet=False,
            )
        return torch.load(path, map_location="cpu")

    def make_src_mask(self, src):
        return (src == PAD_IDX).unsqueeze(1).unsqueeze(2)

    def make_trg_mask(self, trg):
        T = trg.size(1)
        pad = (trg == PAD_IDX).unsqueeze(1).unsqueeze(2)
        causal = torch.triu(
            torch.ones(T, T, device=trg.device, dtype=torch.bool), diagonal=1
        ).unsqueeze(0).unsqueeze(0)
        return pad | causal

    def forward(self, src, trg, src_mask=None, trg_mask=None):
        if src_mask is None:
            src_mask = self.make_src_mask(src)
        if trg_mask is None:
            trg_mask = self.make_trg_mask(trg)
        enc = self.encoder(src, src_mask)
        dec = self.decoder(trg, enc, src_mask, trg_mask)
        return self.fc_out(dec)

    @torch.no_grad()
    def infer(self, german_sentence):
        device = next(self.parameters()).device
        self.eval()
        tokens = [t.text.lower() for t in self.spacy_de.tokenizer(german_sentence)]
        src_ids = [SOS_IDX] + [self.src_vocab.get(t, UNK_IDX) for t in tokens] + [EOS_IDX]
        src = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(device)
        src_mask = self.make_src_mask(src)
        enc = self.encoder(src, src_mask)
        trg_ids = [SOS_IDX]
        for _ in range(self.max_len):
            trg = torch.tensor(trg_ids, dtype=torch.long).unsqueeze(0).to(device)
            trg_mask = self.make_trg_mask(trg)
            dec = self.decoder(trg, enc, src_mask, trg_mask)
            nxt = self.fc_out(dec[:, -1]).argmax(-1).item()
            if nxt == EOS_IDX:
                break
            trg_ids.append(nxt)
        return " ".join(self.idx2trg.get(i, "<unk>") for i in trg_ids[1:])
