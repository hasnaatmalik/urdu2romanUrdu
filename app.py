%%writefile app.py
# Replace your current app.py with this corrected version:

import os
import io
import json
import time
import math
import pickle
from typing import List, Optional, Dict, Tuple
import re
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
import matplotlib.pyplot as plt

# -------------------------
# Constants
# -------------------------
DEFAULT_BPE_PATH = "bpe_model.pkl"
DEFAULT_CKPT_PATH = "best_model.pt"
DEFAULT_URDU_FILE = "urdu.txt"
DEFAULT_ROMAN_FILE = "roman.txt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# BPE Implementation
# -------------------------
class UrduRomanBPE:
    def __init__(self):
        self.vocab_size = None
        self.urdu_vocab = {}
        self.roman_vocab = {}
        self.urdu_merges = []
        self.roman_merges = []
        self.special_tokens = ['<PAD>', '<UNK>', '<SOS>', '<EOS>']

    def load_model(self, filepath: str):
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        self.urdu_vocab = data['urdu_vocab']
        self.roman_vocab = data['roman_vocab']
        self.urdu_merges = data['urdu_merges']
        self.roman_merges = data['roman_merges']
        self.vocab_size = data['vocab_size']
        self.special_tokens = data.get('special_tokens', self.special_tokens)

    def _preprocess_urdu_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text.strip())
        text = re.sub(r'[۔؟!]', ' <EOS> ', text)
        text = re.sub(r'[،؍]', ' ', text)
        return text.strip()

    def _preprocess_roman_text(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[.!?]', ' <EOS> ', text)
        text = re.sub(r'[,;:]', ' ', text)
        return text.strip()

    def _get_word_tokens(self, word: str) -> List[str]:
        return list(word) + ['</w>']

    def _apply_bpe(self, word: str, vocab: Dict, merges: List[Tuple[str, str]]) -> List[str]:
        if not word:
            return []
        word_tokens = self._get_word_tokens(word)
        if len(word_tokens) == 1:
            return word_tokens
        
        for merge in merges:
            if len(word_tokens) == 1:
                break
            new_word_tokens = []
            i = 0
            while i < len(word_tokens):
                if (i < len(word_tokens) - 1 and
                    word_tokens[i] == merge[0] and
                    word_tokens[i + 1] == merge[1]):
                    new_word_tokens.append(''.join(merge))
                    i += 2
                else:
                    new_word_tokens.append(word_tokens[i])
                    i += 1
            word_tokens = new_word_tokens
        return word_tokens

    def encode_urdu(self, text: str) -> List[int]:
        text = self._preprocess_urdu_text(text)
        tokens = []
        for word in text.split():
            if word in self.special_tokens:
                tokens.append(self.urdu_vocab.get(word, self.urdu_vocab['<UNK>']))
            else:
                for tok in self._apply_bpe(word, self.urdu_vocab, self.urdu_merges):
                    tokens.append(self.urdu_vocab.get(tok, self.urdu_vocab['<UNK>']))
        return tokens

    def decode_roman(self, token_ids: List[int]) -> str:
        id_to_token = {idx: token for token, idx in self.roman_vocab.items()}
        toks = [id_to_token.get(t, '<UNK>') for t in token_ids]
        text = ''.join(toks).replace('</w>', ' ')
        return text.strip()

# -------------------------
# Model Architecture
# -------------------------
class BiLSTMEncoder(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.bilstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None):
        emb = self.dropout(self.embedding(x))
        if lengths is not None:
            emb = nn.utils.rnn.pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        outputs, (hidden, cell) = self.bilstm(emb)
        if lengths is not None:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        return outputs, hidden, cell

class LSTMDecoder(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int,
                 encoder_hidden_dim: int, num_layers: int = 4, dropout: float = 0.3):
        super().__init__()
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embedding_dim + encoder_hidden_dim * 2,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        # Match the layer names from your trained model
        self.attention = nn.Linear(hidden_dim + encoder_hidden_dim * 2, encoder_hidden_dim * 2)
        self.out_projection = nn.Linear(hidden_dim + encoder_hidden_dim * 2, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor, cell: torch.Tensor,
                encoder_outputs: torch.Tensor, mask: Optional[torch.Tensor] = None):
        emb = self.dropout(self.embedding(x))
        top_hidden = hidden[-1].unsqueeze(1)
        enc_len = encoder_outputs.size(1)
        top_rep = top_hidden.repeat(1, enc_len, 1)
        att_in = torch.cat([top_rep, encoder_outputs], dim=2)
        scores = self.attention(att_in)
        scores = torch.sum(scores * encoder_outputs, dim=2)
        
        # FIX: Use non-in-place operation to avoid gradient issues
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)  # Remove underscore
        
        att_w = F.softmax(scores, dim=1)
        context = torch.bmm(att_w.unsqueeze(1), encoder_outputs)
        lstm_in = torch.cat([emb, context], dim=2)
        lstm_out, (hidden, cell) = self.lstm(lstm_in, (hidden, cell))
        out_in = torch.cat([lstm_out.squeeze(1), context.squeeze(1)], dim=1)
        logits = self.out_projection(out_in)
        return logits, hidden, cell, att_w

class Seq2SeqModel(nn.Module):
    def __init__(self, encoder_vocab_size: int, decoder_vocab_size: int,
                 embedding_dim: int = 256, encoder_hidden_dim: int = 512,
                 decoder_hidden_dim: int = 512, encoder_layers: int = 2,
                 decoder_layers: int = 4, dropout: float = 0.3):
        super().__init__()
        self.encoder = BiLSTMEncoder(encoder_vocab_size, embedding_dim, encoder_hidden_dim,
                                   num_layers=encoder_layers, dropout=dropout)
        self.decoder = LSTMDecoder(decoder_vocab_size, embedding_dim, decoder_hidden_dim,
                                 encoder_hidden_dim, num_layers=decoder_layers, dropout=dropout)
        self.bridge_hidden = nn.Linear(encoder_hidden_dim * 2, decoder_hidden_dim)
        self.bridge_cell = nn.Linear(encoder_hidden_dim * 2, decoder_hidden_dim)
        self.decoder_layers = decoder_layers

    def _bridge(self, h: torch.Tensor, c: torch.Tensor):
        last_h = torch.cat([h[-2], h[-1]], dim=1)
        last_c = torch.cat([c[-2], c[-1]], dim=1)
        dh = self.bridge_hidden(last_h).unsqueeze(0).repeat(self.decoder_layers, 1, 1)
        dc = self.bridge_cell(last_c).unsqueeze(0).repeat(self.decoder_layers, 1, 1)
        return dh, dc

    def translate(self, src: torch.Tensor, max_length: int, sos_token: int, eos_token: int,
                  src_lengths: Optional[torch.Tensor] = None):
        self.eval()
        with torch.no_grad():
            B = src.size(0)
            enc_out, enc_h, enc_c = self.encoder(src, src_lengths)
            dec_h, dec_c = self._bridge(enc_h, enc_c)
            if src_lengths is not None:
                mask = torch.zeros(B, src.size(1), device=src.device)
                for i, L in enumerate(src_lengths):
                    mask[i, :L] = 1
            else:
                mask = (src != 0).float()
            dec_in = torch.full((B, 1), sos_token, device=src.device, dtype=torch.long)
            outputs = []
            attns = []
            for _ in range(max_length):
                logit, dec_h, dec_c, att_w = self.decoder(dec_in, dec_h, dec_c, enc_out, mask)
                pred = logit.argmax(dim=1)
                outputs.append(pred)
                attns.append(att_w)
                dec_in = pred.unsqueeze(1)
                if (pred == eos_token).all():
                    break
            return torch.stack(outputs, dim=1), torch.stack(attns, dim=1)

# -------------------------
# Caching functions
# -------------------------
@st.cache_resource(show_spinner=True)
def load_bpe(bpe_path: str):
    bpe = UrduRomanBPE()
    bpe.load_model(bpe_path)
    return bpe

@st.cache_resource(show_spinner=True)
def load_model(ckpt_path: str, _bpe: UrduRomanBPE):
    enc_vocab = len(_bpe.urdu_vocab)
    dec_vocab = len(_bpe.roman_vocab)
    model = Seq2SeqModel(encoder_vocab_size=enc_vocab, decoder_vocab_size=dec_vocab,
                        embedding_dim=256, encoder_hidden_dim=512, decoder_hidden_dim=512,
                        encoder_layers=2, decoder_layers=4, dropout=0.3)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    history = {
        'train_losses': ckpt.get('train_losses', []),
        'val_losses': ckpt.get('val_losses', []),
        'val_bleus': ckpt.get('val_bleus', []),
        'val_cers': ckpt.get('val_cers', []),
    }
    return model, history

# -------------------------
# Helper functions
# -------------------------
def make_src_tensor(bpe: UrduRomanBPE, urdu_texts: List[str], max_len: int = 50) -> Tuple[torch.Tensor, torch.Tensor]:
    sos = bpe.urdu_vocab['<SOS>']; eos = bpe.urdu_vocab['<EOS>']; pad = bpe.urdu_vocab['<PAD>']
    seqs = []
    lens = []
    for t in urdu_texts:
        ids = bpe.encode_urdu(t)
        ids = [sos] + ids + [eos]
        lens.append(min(len(ids), max_len))
        ids = ids[:max_len]
        if len(ids) < max_len:
            ids = ids + [pad]*(max_len - len(ids))
        seqs.append(ids)
    tens = torch.tensor(seqs, dtype=torch.long, device=DEVICE)
    lens = torch.tensor(lens, dtype=torch.long, device=DEVICE)
    return tens, lens

def decode_pred_tokens(bpe: UrduRomanBPE, pred_ids: List[int]) -> str:
    pad = bpe.roman_vocab['<PAD>']; sos = bpe.roman_vocab['<SOS>']; eos = bpe.roman_vocab['<EOS>']
    cleaned = []
    for t in pred_ids:
        if t == eos:
            break
        if t in (pad, sos):
            continue
        cleaned.append(t)
    return bpe.decode_roman(cleaned)

def draw_attention(attn: np.ndarray, urdu_tokens_vis: List[str], roman_tokens_vis: List[str]):
    fig, ax = plt.subplots(figsize=(min(12, 1 + 0.6*len(urdu_tokens_vis)), min(8, 1 + 0.5*len(roman_tokens_vis))))
    im = ax.imshow(attn, aspect='auto')
    ax.set_xticks(np.arange(len(urdu_tokens_vis)))
    ax.set_xticklabels(urdu_tokens_vis, rotation=45, ha='right')
    ax.set_yticks(np.arange(len(roman_tokens_vis)))
    ax.set_yticklabels(roman_tokens_vis)
    ax.set_xlabel("Source (Urdu tokens incl. <SOS>/<EOS>)")
    ax.set_ylabel("Generated Roman tokens")
    ax.set_title("Attention heatmap")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    st.pyplot(fig)

def urdu_token_visuals(bpe: UrduRomanBPE, urdu_text: str, max_len: int = 50) -> List[str]:
    ids = bpe.encode_urdu(urdu_text)
    ids = [bpe.urdu_vocab['<SOS>']] + ids + [bpe.urdu_vocab['<EOS>']]
    id2tok = {idx: tok for tok, idx in bpe.urdu_vocab.items()}
    toks = [id2tok.get(i, '<UNK>') for i in ids][:max_len]
    return toks

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Urdu → Roman Urdu (Seq2Seq + BPE)", page_icon="🔤", layout="wide")
st.title("🔤 Urdu → Roman Urdu Translator (Seq2Seq + BPE)")
st.caption(f"Device: **{DEVICE}** · Loads your `bpe_model.pkl` + `best_model.pt`")

with st.sidebar:
    st.header("Artifacts")
    bpe_path = st.text_input("Path to BPE model (.pkl)", value=DEFAULT_BPE_PATH)
    ckpt_path = st.text_input("Path to trained checkpoint (.pt)", value=DEFAULT_CKPT_PATH)
    max_len = st.slider("Max sequence length", min_value=16, max_value=200, value=50, step=2)
    show_prob = st.checkbox("(For debug) show raw IDs", value=False)

    load_btn = st.button("Load / Reload Artifacts")

# Load BPE
if 'bpe' not in st.session_state or load_btn:
    try:
        st.session_state.bpe = load_bpe(bpe_path)
        st.success("BPE loaded.")
    except Exception as e:
        st.error(f"Failed to load BPE: {e}")

# Load Model
if 'model' not in st.session_state or load_btn:
    try:
        if 'bpe' in st.session_state:
            st.session_state.model, st.session_state.history = load_model(ckpt_path, st.session_state.bpe)
            st.success("Model loaded.")
        else:
            st.warning("Load BPE first.")
    except Exception as e:
        st.error(f"Failed to load model: {e}")

bpe = st.session_state.get('bpe', None)
model = st.session_state.get('model', None)
history = st.session_state.get('history', {'train_losses':[], 'val_losses':[], 'val_bleus':[], 'val_cers':[]})

col1, col2 = st.columns([1,1])

# Single translation
with col1:
    st.subheader("Single Translation")
    urdu_text = st.text_area("Urdu text", value="تو کبھی خود کو بھی دیکھے گا تو ڈر جائے گا", height=120)
    run_single = st.button("Translate")
    if run_single and bpe is not None and model is not None:
        try:
            src, src_len = make_src_tensor(bpe, [urdu_text], max_len=max_len)
            sos_t = bpe.roman_vocab['<SOS>']; eos_t = bpe.roman_vocab['<EOS>']
            t0 = time.time()
            preds, attns = model.translate(src, max_length=max_len, sos_token=sos_t, eos_token=eos_t, src_lengths=src_len)
            infer_ms = (time.time() - t0)*1000.0
            pred_ids = preds[0].detach().cpu().tolist()
            roman = decode_pred_tokens(bpe, pred_ids)
            st.write(f"**Roman:** {roman}")
            st.caption(f"Inference: {infer_ms:.1f} ms")
            if show_prob:
                st.code(f"Pred IDs: {pred_ids}")

            with st.expander("Show attention heatmap"):
                att_np = attns[0].detach().cpu().numpy()
                ur_vis = urdu_token_visuals(bpe, urdu_text, max_len=max_len)
                roman_vis = roman.split() if roman.strip() else [""]
                tgt_len = min(len(roman_vis), att_np.shape[0])
                src_len_vis = min(len(ur_vis), att_np.shape[1])
                draw_attention(att_np[:tgt_len, :src_len_vis], ur_vis[:src_len_vis], roman_vis[:tgt_len])
        except Exception as e:
            st.error(f"Translation failed: {e}")

# Batch translation
with col2:
    st.subheader("Batch Translation")
    st.caption("Enter one Urdu sentence per line.")
    batch_in = st.text_area("Batch input", value="میں آپ سے محبت کرتا ہوں\nآج موسم بہت اچھا ہے", height=120)
    max_items = st.number_input("Max lines to process", min_value=1, max_value=128, value=8, step=1)
    run_batch = st.button("Run batch")
    if run_batch and bpe is not None and model is not None:
        lines = [x.strip() for x in batch_in.splitlines() if x.strip()][:max_items]
        if not lines:
            st.warning("No non-empty lines found.")
        else:
            try:
                src, src_len = make_src_tensor(bpe, lines, max_len=max_len)
                sos_t = bpe.roman_vocab['<SOS>']; eos_t = bpe.roman_vocab['<EOS>']
                preds, attns = model.translate(src, max_length=max_len, sos_token=sos_t, eos_token=eos_t, src_lengths=src_len)
                out = []
                for i, line in enumerate(lines):
                    ids = preds[i].detach().cpu().tolist()
                    out.append((line, decode_pred_tokens(bpe, ids)))
                st.markdown("**Results**")
                for ur, ro in out:
                    st.write("—")
                    st.write(f"**Urdu:** {ur}")
                    st.write(f"**Roman:** {ro}")
            except Exception as e:
                st.error(f"Batch translation failed: {e}")

st.markdown("---")

# Quick evaluation
st.subheader("Quick Evaluation (optional)")
st.caption("Evaluates on top-N pairs from urdu.txt / roman.txt if those files exist.")

eval_cols = st.columns([1,1,1])
with eval_cols[0]:
    urdu_file = st.text_input("Urdu file", value=DEFAULT_URDU_FILE)
with eval_cols[1]:
    roman_file = st.text_input("Roman file", value=DEFAULT_ROMAN_FILE)
with eval_cols[2]:
    eval_N = st.number_input("Evaluate first N lines", min_value=1, max_value=200, value=20, step=1)

run_eval = st.button("Run quick eval")
if run_eval and bpe is not None and model is not None:
    if not (os.path.exists(urdu_file) and os.path.exists(roman_file)):
        st.error("Files not found.")
    else:
        try:
            with open(urdu_file, 'r', encoding='utf-8') as f:
                ur_lines = [l.strip() for l in f.readlines()]
            with open(roman_file, 'r', encoding='utf-8') as f:
                ro_lines = [l.strip() for l in f.readlines()]
            n = min(eval_N, len(ur_lines), len(ro_lines))
            ur = ur_lines[:n]; ro_ref = ro_lines[:n]
            src, src_len = make_src_tensor(bpe, ur, max_len=max_len)
            sos_t = bpe.roman_vocab['<SOS>']; eos_t = bpe.roman_vocab['<EOS>']
            preds, attns = model.translate(src, max_length=max_len, sos_token=sos_t, eos_token=eos_t, src_lengths=src_len)
            preds_txt = [decode_pred_tokens(bpe, preds[i].detach().cpu().tolist()) for i in range(n)]
            for i in range(n):
                st.write("—")
                st.write(f"**Urdu:** {ur[i]}")
                st.write(f"**Ref:**  {ro_ref[i]}")
                st.write(f"**Pred:** {preds_txt[i]}")
        except Exception as e:
            st.error(f"Evaluation failed: {e}")

st.markdown("---")

# Training curves
st.subheader("Training Curves (from checkpoint)")
c1, c2, c3 = st.columns(3)
with c1:
    if history['train_losses']:
        fig = plt.figure()
        plt.plot(history['train_losses'])
        plt.title("Train Loss")
        plt.xlabel("Epoch"); plt.ylabel("Loss")
        st.pyplot(fig)
with c2:
    if history['val_losses']:
        fig = plt.figure()
        plt.plot(history['val_losses'])
        plt.title("Val Loss")
        plt.xlabel("Epoch"); plt.ylabel("Loss")
        st.pyplot(fig)
with c3:
    if history['val_bleus']:
        fig = plt.figure()
        plt.plot(history['val_bleus'])
        plt.title("Val BLEU")
        plt.xlabel("Epoch"); plt.ylabel("BLEU")
        st.pyplot(fig)

st.caption("If charts are empty, your checkpoint might not contain these histories.")
st.markdown("—")
st.caption("Built for Urdu-Roman transliteration using BiLSTM + BPE")