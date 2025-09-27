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
        self.attention = nn.Linear(hidden_dim + encoder_hidden_dim * 2, encoder_hidden_dim * 2)
        self.out_projection = nn.Linear(hidden_dim + encoder_hidden_dim * 2, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor, cell: torch.Tensor,
                encoder_outputs: torch.Tensor, mask: Optional[torch.Tensor] = None):
        batch_size = x.size(0)
        enc_seq_len = encoder_outputs.size(1)
        
        emb = self.dropout(self.embedding(x))  # (B, 1, E)
        top_hidden = hidden[-1].unsqueeze(1)   # (B, 1, H)
        
        top_hidden_repeated = top_hidden.repeat(1, enc_seq_len, 1)  # (B, S, H)
        att_input = torch.cat([top_hidden_repeated, encoder_outputs], dim=2)  # (B, S, H + 2H)
        
        att_scores = self.attention(att_input)  # (B, S, 2H)
        
        if mask is not None:
            if mask.shape[1] != enc_seq_len:
                if mask.shape[1] > enc_seq_len:
                    mask = mask[:, :enc_seq_len]
                else:
                    pad_size = enc_seq_len - mask.shape[1]
                    mask = F.pad(mask, (0, pad_size), value=0.0)
            att_scores = att_scores.masked_fill(mask.unsqueeze(2) == 0, -1e9)
        
        att_weights = F.softmax(att_scores.sum(dim=2), dim=1)  # (B, S)
        context = torch.bmm(att_weights.unsqueeze(1), encoder_outputs)  # (B, 1, 2H)
        
        lstm_input = torch.cat([emb, context], dim=2)  # (B, 1, E + 2H)
        lstm_out, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        
        output_input = torch.cat([lstm_out.squeeze(1), context.squeeze(1)], dim=1)  # (B, H + 2H)
        logits = self.out_projection(output_input)  # (B, V)
        
        return logits, hidden, cell, att_weights

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
        last_h = torch.cat([h[-2], h[-1]], dim=1)  # (B, 2H)
        last_c = torch.cat([c[-2], c[-1]], dim=1)   # (B, 2H)
        dh = self.bridge_hidden(last_h).unsqueeze(0).repeat(self.decoder_layers, 1, 1)
        dc = self.bridge_cell(last_c).unsqueeze(0).repeat(self.decoder_layers, 1, 1)
        return dh, dc

    def translate(self, src: torch.Tensor, max_length: int, sos_token: int, eos_token: int,
                  src_lengths: Optional[torch.Tensor] = None):
        self.eval()
        with torch.no_grad():
            batch_size = src.size(0)
            encoder_outputs, encoder_hidden, encoder_cell = self.encoder(src, src_lengths)
            decoder_hidden, decoder_cell = self._bridge(encoder_hidden, encoder_cell)
            
            if src_lengths is not None:
                mask = torch.zeros(batch_size, encoder_outputs.size(1), device=src.device, dtype=torch.float)
                for i, length in enumerate(src_lengths):
                    mask[i, :min(length.item(), encoder_outputs.size(1))] = 1.0
            else:
                mask = (src != 0).float()
            
            decoder_input = torch.full((batch_size, 1), sos_token, device=src.device, dtype=torch.long)
            outputs = []
            attention_weights = []
            
            for _ in range(max_length):
                logits, decoder_hidden, decoder_cell, att_weights = self.decoder(
                    decoder_input, decoder_hidden, decoder_cell, encoder_outputs, mask
                )
                predictions = logits.argmax(dim=1)
                outputs.append(predictions)
                attention_weights.append(att_weights)
                decoder_input = predictions.unsqueeze(1)
                
                if (predictions == eos_token).all():
                    break
            
            final_outputs = torch.stack(outputs, dim=1) if outputs else torch.zeros((batch_size, 1), dtype=torch.long, device=src.device)
            final_attention = torch.stack(attention_weights, dim=1) if attention_weights else torch.zeros((batch_size, 1, encoder_outputs.size(1)), device=src.device)
            
            return final_outputs, final_attention

# -------------------------
# Caching and Helper Functions
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

def make_src_tensor(bpe: UrduRomanBPE, urdu_texts: List[str], max_len: int = 50) -> Tuple[torch.Tensor, torch.Tensor]:
    sos = bpe.urdu_vocab['<SOS>']
    eos = bpe.urdu_vocab['<EOS>']
    pad = bpe.urdu_vocab['<PAD>']
    
    sequences = []
    lengths = []
    
    for text in urdu_texts:
        tokens = bpe.encode_urdu(text)
        tokens = [sos] + tokens + [eos]
        actual_length = min(len(tokens), max_len)
        lengths.append(actual_length)
        
        if len(tokens) > max_len:
            tokens = tokens[:max_len]
            tokens[-1] = eos
        while len(tokens) < max_len:
            tokens.append(pad)
        
        sequences.append(tokens)
    
    src_tensor = torch.tensor(sequences, dtype=torch.long, device=DEVICE)
    len_tensor = torch.tensor(lengths, dtype=torch.long, device=DEVICE)
    
    return src_tensor, len_tensor

def decode_pred_tokens(bpe: UrduRomanBPE, pred_ids: List[int]) -> str:
    pad = bpe.roman_vocab['<PAD>']
    sos = bpe.roman_vocab['<SOS>']
    eos = bpe.roman_vocab['<EOS>']
    
    cleaned_tokens = []
    for token_id in pred_ids:
        if token_id == eos:
            break
        if token_id not in [pad, sos]:
            cleaned_tokens.append(token_id)
    
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
    ids = bpe.encode_urdu(urdu_text)
    ids = [bpe.urdu_vocab['<SOS>']] + ids + [bpe.urdu_vocab['<EOS>']]
    id2tok = {idx: tok for tok, idx in bpe.urdu_vocab.items()}
    tokens = [id2tok.get(i, '<UNK>') for i in ids[:max_len]]
    return tokens

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Urdu → Roman Urdu", page_icon="🔤", layout="wide")
st.title("🔤 Urdu → Roman Urdu Translator")
st.caption(f"Device: **{DEVICE}**")

with st.sidebar:
    st.header("Model Settings")
    bpe_path = st.text_input("BPE model path", value=DEFAULT_BPE_PATH)
    ckpt_path = st.text_input("Model checkpoint path", value=DEFAULT_CKPT_PATH)
    max_len = st.slider("Max sequence length", min_value=16, max_value=100, value=50, step=2)
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
            st.session_state.model, st.session_state.history = load_model(ckpt_path, st.session_state.bpe)
            st.success("✅ Translation model loaded")
        else:
            st.warning("⚠️ Load BPE model first")
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")

bpe = st.session_state.get('bpe')
model = st.session_state.get('model')
history = st.session_state.get('history', {})

# Main interface
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Single Translation")
    urdu_input = st.text_area("Enter Urdu text:", 
                             value="تو کبھی خود کو بھی دیکھے گا تو ڈر جائے گا", 
                             height=100)
    
    if st.button("Translate", type="primary") and bpe and model:
        try:
            with st.spinner("Translating..."):
                src_tensor, src_lengths = make_src_tensor(bpe, [urdu_input], max_len)
                sos_token = bpe.roman_vocab['<SOS>']
                eos_token = bpe.roman_vocab['<EOS>']
                
                start_time = time.time()
                predictions, attention = model.translate(
                    src_tensor, max_length=max_len, 
                    sos_token=sos_token, eos_token=eos_token, 
                    src_lengths=src_lengths
                )
                inference_time = (time.time() - start_time) * 1000
                
                pred_ids = predictions[0].cpu().tolist()
                translation = decode_pred_tokens(bpe, pred_ids)
                
                st.write("**Translation:**")
                st.write(f"🇺🇷 {translation}")
                st.caption(f"⏱️ Inference time: {inference_time:.1f}ms")
                
                if show_debug:
                    st.code(f"Token IDs: {pred_ids}")
            
            with st.expander("🔍 Show Attention Heatmap"):
                if attention is not None and attention.numel() > 0:
                    att_matrix = attention[0].cpu().numpy()
                    src_tokens = urdu_token_visuals(bpe, urdu_input, max_len)
                    tgt_tokens = translation.split() if translation.strip() else ["<empty>"]
                    
                    min_len = min(att_matrix.shape[0], len(tgt_tokens))
                    att_matrix = att_matrix[:min_len, :min(att_matrix.shape[1], len(src_tokens))]
                    draw_attention(att_matrix, src_tokens[:att_matrix.shape[1]], tgt_tokens[:att_matrix.shape[0]])
                else:
                    st.warning("No attention weights available")
                    
        except Exception as e:
            st.error(f"❌ Translation failed: {str(e)}")

with col2:
    st.subheader("Batch Translation")
    st.caption("One sentence per line")
    
    batch_input = st.text_area("Enter multiple Urdu sentences:",
                               value="میں آپ سے محبت کرتا ہوں\nآج موسم بہت اچھا ہے",
                               height=100)
    
    max_batch = st.number_input("Max sentences", min_value=1, max_value=20, value=5)
    
    if st.button("Translate Batch") and bpe and model:
        lines = [line.strip() for line in batch_input.splitlines() if line.strip()]
        lines = lines[:max_batch]
        
        if lines:
            try:
                with st.spinner(f"Translating {len(lines)} sentences..."):
                    src_tensor, src_lengths = make_src_tensor(bpe, lines, max_len)
                    sos_token = bpe.roman_vocab['<SOS>']
                    eos_token = bpe.roman_vocab['<EOS>']
                    
                    predictions, _ = model.translate(
                        src_tensor, max_length=max_len,
                        sos_token=sos_token, eos_token=eos_token,
                        src_lengths=src_lengths
                    )
                    
                    st.write("**Batch Results:**")
                    for i, (urdu_line, pred_ids) in enumerate(zip(lines, predictions)):
                        translation = decode_pred_tokens(bpe, pred_ids.cpu().tolist())
                        st.write(f"**{i+1}.** {urdu_line}")
                        st.write(f"   → {translation}")
                        st.write("---")
                        
            except Exception as e:
                st.error(f"❌ Batch translation failed: {str(e)}")
        else:
            st.warning("No valid sentences found")

# Training curves
if history:
    st.markdown("---")
    st.subheader("📈 Training History")
    
    chart_cols = st.columns(3)
    
    with chart_cols[0]:
        if 'train_losses' in history and history['train_losses']:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(history['train_losses'])
            ax.set_title("Training Loss")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
    
    with chart_cols[1]:
        if 'val_losses' in history and history['val_losses']:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(history['val_losses'])
            ax.set_title("Validation Loss")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
    

st.markdown("---")
st.caption("🚀 Urdu-Roman Neural Machine Translation • Built with PyTorch & Streamlit")