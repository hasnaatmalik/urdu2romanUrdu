import os
import re
import unicodedata
import time
import pickle
from typing import List, Optional, Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

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
        text = unicodedata.normalize("NFKC", text)
        text = re.sub(r"[^\u0600-\u06FF\s]", " ", text)  # Keep Urdu & spaces
        text = re.sub(r"\s+", " ", text).strip()
        text = text.replace("ك", "ک").replace("ي", "ی").replace("ە", "ہ").replace("ة", "ہ").replace("ئ", "ی")
        return text

    def _preprocess_roman_text(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"\s+", " ", text).strip()
        return text

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
class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, n_layers, dropout, pad_idx):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.emb_drop = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            emb_dim, hid_dim // 2, num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=True, batch_first=True
        )

    def forward(self, src, src_len):
        e = self.emb_drop(self.emb(src))
        packed = nn.utils.rnn.pack_padded_sequence(e, src_len.cpu(), batch_first=True, enforce_sorted=False)
        h, (hn, cn) = self.lstm(packed)
        h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True)
        def _merge(x):
            layers = []
            for l in range(0, x.size(0), 2):
                layers.append(torch.cat([x[l], x[l+1]], dim=-1))
            return torch.stack(layers, dim=0)
        return h, (_merge(hn), _merge(cn))

class LuongAttention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.Wa = nn.Linear(hid_dim, hid_dim, bias=False)

    def forward(self, dec_h, enc_out, src_mask):
        q = self.Wa(dec_h).unsqueeze(1)  # [B, 1, H]
        scores = torch.bmm(q, enc_out.transpose(1, 2)).squeeze(1)  # [B, S]
        scores = scores.masked_fill(~src_mask, -1e9)
        attn = F.softmax(scores, dim=-1)
        ctx = torch.bmm(attn.unsqueeze(1), enc_out).squeeze(1)
        return ctx, attn

class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, n_layers, dropout, pad_idx):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.emb_drop = nn.Dropout(dropout)
        self.lstm = nn.LSTM(emb_dim + hid_dim, hid_dim, num_layers=n_layers,
                            dropout=dropout if n_layers > 1 else 0.0, batch_first=True)
        self.attn = LuongAttention(hid_dim)
        self.out_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(hid_dim, vocab_size)

    def forward(self, dec_inp, enc_out, src_mask, hidden):
        B, T = dec_inp.shape
        e = self.emb_drop(self.emb(dec_inp))
        outputs = []
        h, c = hidden
        for t in range(T):
            dec_h_top = h[-1]
            ctx, _ = self.attn(dec_h_top, enc_out, src_mask)
            x_t = torch.cat([e[:, t, :], ctx], dim=-1).unsqueeze(1)
            o, (h, c) = self.lstm(x_t, (h, c))
            outputs.append(self.proj(self.out_drop(o.squeeze(1))))
        return torch.stack(outputs, dim=1), (h, c)

class Seq2Seq(nn.Module):
    def __init__(self, enc, dec, pad_id_src):
        super().__init__()
        self.enc, self.dec, self.pad_id_src = enc, dec, pad_id_src

    def _bridge(self, hn, cn):
        L_enc = hn.size(0)
        L_dec = self.dec.lstm.num_layers
        if L_enc == L_dec:
            return hn, cn
        if L_enc < L_dec:
            pad_h = hn.new_zeros(L_dec - L_enc, hn.size(1), hn.size(2))
            pad_c = cn.new_zeros(L_dec - L_enc, cn.size(1), cn.size(2))
            return torch.cat([hn, pad_h], 0), torch.cat([cn, pad_c], 0)
        return hn[-L_dec:], cn[-L_dec:]

    def forward(self, src, src_len, dec_inp):
        enc_out, (hn, cn) = self.enc(src, src_len)
        hn, cn = self._bridge(hn, cn)
        src_mask = (src != self.pad_id_src)
        logits, _ = self.dec(dec_inp, enc_out, src_mask, (hn, cn))
        return logits

    @torch.no_grad()
    def translate(self, src: torch.Tensor, src_len: torch.Tensor, sos_token: int, eos_token: int, max_length: int):
        self.eval()
        batch_size = src.size(0)
        enc_out, (hn, cn) = self.enc(src, src_len)
        hn, cn = self._bridge(hn, cn)
        src_mask = (src != self.pad_id_src)
        
        dec_input = torch.full((batch_size, 1), sos_token, device=src.device, dtype=torch.long)
        outputs = []
        
        for _ in range(max_length):
            logits, (hn, cn) = self.dec(dec_input, enc_out, src_mask, (hn, cn))
            predictions = logits[:, -1, :].argmax(dim=-1)
            outputs.append(predictions)
            dec_input = predictions.unsqueeze(1)
            
            if (predictions == eos_token).all():
                break
        
        return torch.stack(outputs, dim=1) if outputs else torch.zeros((batch_size, 1), dtype=torch.long, device=src.device)

# -------------------------
# Caching and Helper Functions
# -------------------------
@st.cache_resource(show_spinner=True)
def load_model(ckpt_path: str, bpe: UrduRomanBPE):
    enc_vocab = len(bpe.urdu_vocab)
    dec_vocab = len(bpe.roman_vocab)
    encoder = Encoder(enc_vocab, 256, 512, 2, 0.3, bpe.urdu_vocab['<PAD>']).to(DEVICE)
    decoder = Decoder(dec_vocab, 256, 512, 3, 0.3, bpe.roman_vocab['<PAD>']).to(DEVICE)
    model = Seq2Seq(encoder, decoder, bpe.urdu_vocab['<PAD>']).to(DEVICE)
    
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model

@st.cache_resource(show_spinner=True)
def load_bpe(bpe_path: str):
    bpe = UrduRomanBPE()
    bpe.load_model(bpe_path)
    return bpe

def make_src_tensor(bpe: UrduRomanBPE, urdu_texts: List[str], max_len: int = 50) -> Tuple[torch.Tensor, torch.Tensor]:
    sequences = []
    lengths = []
    for text in urdu_texts:
        tokens = bpe.encode_urdu(text)[:max_len]
        actual_length = min(len(tokens), max_len)
        lengths.append(actual_length)
        while len(tokens) < max_len:
            tokens.append(bpe.urdu_vocab['<PAD>'])
        sequences.append(tokens)
    src_tensor = torch.tensor(sequences, dtype=torch.long, device=DEVICE)
    len_tensor = torch.tensor(lengths, dtype=torch.long, device=DEVICE)
    return src_tensor, len_tensor

def decode_pred_tokens(bpe: UrduRomanBPE, pred_ids: List[int]) -> str:
    pad = bpe.roman_vocab['<PAD>']
    eos = bpe.roman_vocab['<EOS>']
    cleaned_tokens = [t for t in pred_ids if t not in [pad] and t != eos]
    return bpe.decode_roman(cleaned_tokens)

def draw_attention(attn: np.ndarray, urdu_tokens_vis: List[str], roman_tokens_vis: List[str]):
    if len(roman_tokens_vis) == 0 or len(urdu_tokens_vis) == 0:
        st.warning("No tokens to visualize")
        return
        
    fig, ax = plt.subplots(figsize=(min(12, 1 + 0.6*len(urdu_tokens_vis)), 
                                   min(8, 1 + 0.5*len(roman_tokens_vis))))
    im = ax.imshow(attn, aspect='auto', cmap='Blues')
    
    ax.set_xticks(np.arange(len(urdu_tokens_vis)))
    ax.set_xticklabels(urdu_tokens_vis, rotation=45, ha='right')
    ax.set_yticks(np.arange(len(roman_tokens_vis)))
    ax.set_yticklabels(roman_tokens_vis)
    ax.set_xlabel("Source (Urdu tokens)")
    ax.set_ylabel("Generated Roman tokens")
    ax.set_title("Attention Heatmap")
    
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    st.pyplot(fig)

def urdu_token_visuals(bpe: UrduRomanBPE, urdu_text: str, max_len: int = 50) -> List[str]:
    ids = bpe.encode_urdu(urdu_text)[:max_len]
    id2tok = {idx: tok for tok, idx in bpe.urdu_vocab.items()}
    return [id2tok.get(i, '<UNK>') for i in ids]

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Urdu → Roman Urdu", page_icon="🔤", layout="centered")
st.title("🔤 Urdu → Roman Urdu Translator")
st.caption(f"Device: **{DEVICE}**")

with st.sidebar:
    st.header("Model Settings")
    bpe_path = st.text_input("BPE model path", value=DEFAULT_BPE_PATH)
    ckpt_path = st.text_input("Model checkpoint path", value=DEFAULT_CKPT_PATH)
    max_len = st.slider("Max sequence length", min_value=16, max_value=512, value=256, step=32)
    show_debug = st.checkbox("Show debug info", value=False)
    load_btn = st.button("Load/Reload Models")

# Load models
if 'bpe' not in st.session_state or load_btn:
    try:
        st.session_state.bpe = load_bpe(bpe_path)
        st.success("✅ BPE model loaded")
    except Exception as e:
        st.error(f"❌ BPE loading failed: {e}")

if 'model' not in st.session_state or load_btn:
    try:
        if 'bpe' in st.session_state:
            st.session_state.model = load_model(ckpt_path, st.session_state.bpe)
            st.success("✅ Translation model loaded")
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")

bpe = st.session_state.get('bpe')
model = st.session_state.get('model')

st.subheader("✍️ Enter Misrah (Urdu)")
user_text = st.text_area("", height=120, placeholder="یہ جہاں خواب ہے یا خواب کا جہاں کچھ ہے ...")

if st.button("Transliterate") and bpe and model:
    try:
        with st.spinner("Transliterating..."):
            src_tensor, src_lengths = make_src_tensor(bpe, [user_text], max_len)
            sos_token = bpe.roman_vocab['<SOS>']
            eos_token = bpe.roman_vocab['<EOS>']
            
            start_time = time.time()
            predictions = model.translate(src_tensor, src_lengths, sos_token, eos_token, max_len)
            inference_time = (time.time() - start_time) * 1000
            
            pred_ids = predictions[0].cpu().tolist()
            translation = decode_pred_tokens(bpe, pred_ids)
            
            st.markdown("### 🌿 Roman Transliteration")
            st.code(translation or "(empty)", language=None)
            st.caption(f"⏱️ Inference time: {inference_time:.1f}ms")
            
            if show_debug:
                st.code(f"Token IDs: {pred_ids}")
            
            with st.expander("🔍 Show Attention Heatmap"):
                if hasattr(model.dec, 'attn') and model.dec.attn is not None:
                    # Note: This is a simplification; actual attention weights would need to be captured during decode
                    st.warning("Attention heatmap not fully implemented with this decode method.")
                else:
                    st.warning("No attention weights available")
                    
    except Exception as e:
        st.error(f"❌ Translation failed: {str(e)}")

st.markdown("---")
st.caption("🚀 Urdu-Roman Neural Machine Translation • Built with PyTorch & Streamlit")