from collections import Counter
import spacy
import torch
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

PAD_IDX = 1
UNK_IDX = 0
SOS_IDX = 2
EOS_IDX = 3
SPECIALS = ["<unk>", "<pad>", "<sos>", "<eos>"]

def build_vocab(token_lists, min_freq=2):
    counter = Counter(tok for sent in token_lists for tok in sent)
    kept = [tok for tok, cnt in counter.items() if cnt >= min_freq]
    vocab = {tok: i + len(SPECIALS) for i, tok in enumerate(kept)}
    for i, s in enumerate(SPECIALS):
        vocab[s] = i
    return vocab

class TranslationDataset(Dataset):
    def __init__(self, records, src_vocab, trg_vocab, spacy_de, spacy_en, max_len=150):
        self.records = records
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.spacy_de = spacy_de
        self.spacy_en = spacy_en
        self.max_len = max_len

    def _tok_de(self, text):
        return [t.text.lower() for t in self.spacy_de.tokenizer(text)]

    def _tok_en(self, text):
        return [t.text.lower() for t in self.spacy_en.tokenizer(text)]

    def _encode(self, tokens, vocab):
        ids = [vocab.get(t, UNK_IDX) for t in tokens[: self.max_len - 2]]
        return [SOS_IDX] + ids + [EOS_IDX]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        src = torch.tensor(self._encode(self._tok_de(r["de"]), self.src_vocab), dtype=torch.long)
        trg = torch.tensor(self._encode(self._tok_en(r["en"]), self.trg_vocab), dtype=torch.long)
        return src, trg

def collate_fn(batch):
    srcs, trgs = zip(*batch)
    return (
        pad_sequence(srcs, batch_first=True, padding_value=PAD_IDX),
        pad_sequence(trgs, batch_first=True, padding_value=PAD_IDX),
    )

def load_data(batch_size=256):
    spacy_de = spacy.load("de_core_news_sm")
    spacy_en = spacy.load("en_core_web_sm")

    raw = load_dataset("bentrevett/multi30k")
    train_records = list(raw["train"])
    val_records = list(raw["validation"])
    test_records = list(raw["test"])

    de_tok = [[t.text.lower() for t in spacy_de.tokenizer(r["de"])] for r in train_records]
    en_tok = [[t.text.lower() for t in spacy_en.tokenizer(r["en"])] for r in train_records]

    src_vocab = build_vocab(de_tok, min_freq=2)
    trg_vocab = build_vocab(en_tok, min_freq=2)

    def make_loader(records, shuffle):
        ds = TranslationDataset(records, src_vocab, trg_vocab, spacy_de, spacy_en)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, num_workers=0)

    return (
        make_loader(train_records, shuffle=True),
        make_loader(val_records, shuffle=False),
        make_loader(test_records, shuffle=False),
        src_vocab,
        trg_vocab,
        spacy_de,
        spacy_en,
        test_records,
    )