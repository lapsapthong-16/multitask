import os
import io
import torch
import pandas as pd
import numpy as np
import streamlit as st

from typing import List, Tuple, Dict
from torch.nn.functional import softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification

st.set_page_config(page_title="PR Crisis Sentiment & Emotion (XAI Demo)", layout="wide")

import torch.nn as nn
from transformers import AutoModel

MTL_DIR = "models/mtl"  # folder with mtl_model.pt and base_tok/

class SimpleMTL(nn.Module):
    """
    Example MTL: shared encoder + two linear heads.
    Adjust hidden size to match your encoder.
    """
    def __init__(self, encoder_name_or_path, num_sent=3, num_emo=6):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name_or_path)
        hidden = self.encoder.config.hidden_size
        self.sent_head = nn.Linear(hidden, num_sent)
        self.emo_head  = nn.Linear(hidden, num_emo)

    def forward(self, **enc):
        # use [CLS] (or first token) pooled representation
        out = self.encoder(**enc)
        # RoBERTa/BERTweet: take first token's hidden state
        cls = out.last_hidden_state[:, 0, :]
        return self.sent_head(cls), self.emo_head(cls)

@st.cache_resource(show_spinner=True)
def load_mtl():
    """Load MTL model if available"""
    mtl_model_path = os.path.join(MTL_DIR, "mtl_model.pt")
    base_tok_path = os.path.join(MTL_DIR, "base_tok")
    
    if not os.path.exists(mtl_model_path) or not os.path.exists(base_tok_path):
        return None, None
    
    try:
        tok = AutoTokenizer.from_pretrained(base_tok_path, use_fast=True)
        mdl = SimpleMTL(base_tok_path)
        state = torch.load(mtl_model_path, map_location=DEVICE)
        # if keys had 'module.' from DDP: state = {k.replace("module.", ""): v for k,v in state.items()}
        mdl.load_state_dict(state, strict=True)
        mdl.eval().to(DEVICE)
        return tok, mdl
    except Exception as e:
        st.error(f"Failed to load MTL model: {str(e)}")
        return None, None

# =========================
# CONFIG — edit as needed
# =========================
# Base paths - will be combined with backbone choice
SENTIMENT_BASE = "models/sentiment"     
EMOTION_BASE   = "models/emotion"       

# match the exact class order used during training
SENTIMENT_LABELS = ["Negative", "Neutral", "Positive"]
EMOTION_LABELS   = ["Anger", "Fear", "Joy", "Sadness", "Surprise", "No Emotion"]

MAX_LEN = 128
DEVICE = "cpu"  # keep cpu for grading demo; switch to "cuda" if available

# Optional Captum (true Integrated Gradients). If not installed, we'll fall back to a 1-pass saliency.
USE_CAPTUM = False
try:
    import captum
    from captum.attr import IntegratedGradients
    USE_CAPTUM = True
except Exception:
    USE_CAPTUM = False

# =========================
# LOAD MODELS
# =========================
@st.cache_resource(show_spinner=True)
def load_models(backbone="bertweet"):
    """Load models based on backbone choice"""
    # Construct paths with backbone subdirectory
    sentiment_path = os.path.join(SENTIMENT_BASE, backbone)
    emotion_path = os.path.join(EMOTION_BASE, backbone)
    
    # Load sentiment model
    try:
        sent_tok = AutoTokenizer.from_pretrained(sentiment_path, use_fast=True)
        sent_mdl = AutoModelForSequenceClassification.from_pretrained(sentiment_path)
        sent_mdl.eval().to(DEVICE)
    except Exception as e:
        st.error(f"Failed to load sentiment model from {sentiment_path}: {str(e)}")
        return None, None, None, None
    
    # Load emotion model
    try:
        emo_tok = AutoTokenizer.from_pretrained(emotion_path, use_fast=True)
        emo_mdl = AutoModelForSequenceClassification.from_pretrained(emotion_path)
        emo_mdl.eval().to(DEVICE)
    except Exception as e:
        st.error(f"Failed to load emotion model from {emotion_path}: {str(e)}")
        return None, None, None, None

    return sent_tok, sent_mdl, emo_tok, emo_mdl

# Initialize with default backbone
sentiment_tok, sentiment_mdl, emotion_tok, emotion_mdl = load_models("bertweet")

# Check if models loaded successfully
if sentiment_tok is None or sentiment_mdl is None or emotion_tok is None or emotion_mdl is None:
    st.error("Failed to load initial models. Please check the model paths and try again.")
    st.stop()

# Check MTL availability
mtl_available = False
mtl_tok, mtl_mdl = None, None

# Check if MTL models exist for the current backbone
def check_mtl_availability(backbone):
    mtl_path = os.path.join(MTL_DIR, backbone)
    # Check if the directory exists and contains model files
    if not os.path.exists(mtl_path):
        return False
    
    # Look for any model files
    model_files = [f for f in os.listdir(mtl_path) if f.endswith('.pt') or f.endswith('.safetensors')]
    return len(model_files) > 0

# Try to load MTL if available for bertweet (default)
if check_mtl_availability("bertweet"):
    mtl_tok, mtl_mdl = load_mtl()
    if mtl_tok is not None and mtl_mdl is not None:
        mtl_available = True

# Function to reload MTL for different backbones
def reload_mtl_for_backbone(backbone):
    global mtl_tok, mtl_mdl, mtl_available
    if check_mtl_availability(backbone):
        mtl_tok, mtl_mdl = load_mtl_dir(os.path.join(MTL_DIR, backbone))
        mtl_available = mtl_tok is not None and mtl_mdl is not None
    else:
        mtl_tok, mtl_mdl = None, None
        mtl_available = False

# =========================
# INFERENCE HELPERS
# =========================
def predict_single(model, tokenizer, text: str, labels: List[str], max_len=MAX_LEN):
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len)
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc)
        probs = softmax(out.logits, dim=-1).squeeze(0)
        conf, idx = torch.max(probs, dim=-1)
    return labels[idx.item()], float(conf.item()), probs.cpu().tolist(), enc

def predict_mtl(mtl_model, tokenizer, text: str):
    """Predict using MTL model with fallback to single-task style"""
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LEN)
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    
    with torch.no_grad():
        try:
            # Try MTL-style prediction first
            if hasattr(mtl_model, 'sent_head') and hasattr(mtl_model, 'emo_head'):
                s_logits, e_logits = mtl_model(**enc)
                s_probs = softmax(s_logits, dim=-1).squeeze(0)
                e_probs = softmax(e_logits, dim=-1).squeeze(0)
            else:
                # Fallback: treat as single-task model
                outputs = mtl_model(**enc)
                logits = outputs.logits
                
                # Assume first half is sentiment, second half is emotion
                # This is a heuristic and may need adjustment
                if logits.shape[1] >= 9:  # 3 sentiment + 6 emotion
                    s_logits = logits[:, :3]
                    e_logits = logits[:, 3:9]
                else:
                    # If we can't determine, use the full logits for both
                    s_logits = logits
                    e_logits = logits
                
                s_probs = softmax(s_logits, dim=-1).squeeze(0)
                e_probs = softmax(e_logits, dim=-1).squeeze(0)
            
            s_conf, s_idx = torch.max(s_probs, dim=-1)
            e_conf, e_idx = torch.max(e_probs, dim=-1)
            
            # Ensure indices are within bounds
            s_idx = min(s_idx.item(), len(SENTIMENT_LABELS) - 1)
            e_idx = min(e_idx.item(), len(EMOTION_LABELS) - 1)
            
        except Exception as e:
            st.error(f"Error during MTL prediction: {str(e)}")
            # Return neutral predictions as fallback
            return ("Neutral", 0.5, [0.33, 0.34, 0.33], enc,
                    "No Emotion", 0.5, [0.17, 0.17, 0.17, 0.17, 0.16, 0.16], enc)
    
    return (
        SENTIMENT_LABELS[s_idx], float(s_conf.item()), s_probs.cpu().tolist(), enc,
        EMOTION_LABELS[e_idx], float(e_conf.item()), e_probs.cpu().tolist(), enc
    )

def format_probs(labels, probs_list) -> Dict[str, float]:
    return {lbl: float(f"{p:.4f}") for lbl, p in zip(labels, probs_list)}


def tokens_and_embeds(model, tokenizer, enc):
    """Return tokens and an embedding function handle for Captum."""
    input_ids = enc["input_ids"]
    attention_mask = enc.get("attention_mask", None)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().tolist())

    def forward_embeds(embeds):
        outputs = model(inputs_embeds=embeds, attention_mask=attention_mask)
        return outputs.logits

    return tokens, attention_mask, input_ids, forward_embeds


def attribution_scores(model, tokenizer, enc, target_idx: int, real_ig: bool):
    """
    Return token-level attribution scores in [0,1].
    If real_ig=True and Captum available, use Integrated Gradients.
    Else use a one-pass saliency on input embeddings (norm of grad).
    """
    input_ids = enc["input_ids"]
    attention_mask = enc.get("attention_mask", None)
    token_count = input_ids.shape[1]
    
    # Handle different model architectures
    if hasattr(model, 'base_model'):
        embeddings_layer = model.base_model.get_input_embeddings()
    elif hasattr(model, 'roberta'):
        embeddings_layer = model.roberta.get_input_embeddings()
    elif hasattr(model, 'bert'):
        embeddings_layer = model.bert.get_input_embeddings()
    else:
        # Fallback: try to find embeddings layer
        embeddings_layer = None
        for module in model.modules():
            if hasattr(module, 'weight') and module.weight.shape[0] > 1000:  # Likely embedding layer
                embeddings_layer = module
                break
        
        if embeddings_layer is None:
            st.error("Could not find embeddings layer for attribution")
            return [0.0] * token_count

    if real_ig and USE_CAPTUM:
        # True Integrated Gradients
        tokens, attn, ids, forward = tokens_and_embeds(model, tokenizer, enc)
        baseline_ids = torch.full_like(ids, fill_value=tokenizer.pad_token_id or tokenizer.eos_token_id)
        baseline_embeds = embeddings_layer(baseline_ids).to(DEVICE)
        input_embeds = embeddings_layer(ids).to(DEVICE)

        ig = IntegratedGradients(lambda e: forward(e)[:, target_idx])
        attributions, _ = ig.attribute(inputs=input_embeds,
                                       baselines=baseline_embeds,
                                       additional_forward_args=None,
                                       n_steps=50, return_convergence_delta=True)
        # aggregate across hidden dim
        scores = attributions.norm(p=2, dim=-1).squeeze(0).detach().cpu().numpy()
    else:
        # Lightweight saliency: grad wrt input embeddings (one backward pass)
        for p in model.parameters():
            p.requires_grad_(False)
        input_embeds = embeddings_layer(input_ids).detach().clone().to(DEVICE).requires_grad_(True)
        outputs = model(inputs_embeds=input_embeds, attention_mask=attention_mask)
        logits = outputs.logits.squeeze(0)
        logits[target_idx].backward()
        grads = input_embeds.grad.detach()  # [1, seq, hid]
        scores = grads.norm(dim=-1).squeeze(0).cpu().numpy()

    # normalize to [0,1]
    if scores.max() > 0:
        scores = scores / (scores.max() + 1e-8)
    return scores.tolist()

from typing import List

def html_highlight(tokens: List[str], scores: List[float]) -> str:
    chunks = []
    skip = {"<s>", "</s>", "[CLS]", "[SEP]", "[PAD]"}
    for tok, s in zip(tokens, scores):
        if tok in skip:
            continue
        # strip common wordpiece prefixes
        if tok.startswith("##"):
            tok = tok[2:]
        if tok == "Ġ":
            continue
        opacity = min(max(float(s), 0.06), 1.0)  # clamp to [0.06, 1.0] so faint tokens are still visible
        chunks.append(
            f"<span style='background: rgba(255,165,0,{opacity}); padding:2px 4px; border-radius:4px; margin:2px'>{tok}</span>"
        )
    return "<div style='line-height:2; font-family: ui-sans-serif, system-ui, -apple-system'>" + " ".join(chunks) + "</div>"

def resolve_paths(mode: str, bk: str):
    if mode == "Single-task":
        sent_path = os.path.join("models", "sentiment", bk)
        emo_path  = os.path.join("models", "emotion", bk)
        return {"mode": "st", "sent": sent_path, "emo": emo_path}
    else:
        mtl_path = os.path.join("models", "mtl", bk)
        return {"mode": "mtl", "mtl": mtl_path}

def load_single_task(model_path):
    """Load a single task model from the given path"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.eval().to(DEVICE)
        return tokenizer, model
    except Exception as e:
        st.error(f"Failed to load model from {model_path}: {str(e)}")
        return None, None

def load_mtl_dir(mtl_path):
    """Load MTL model from the given path"""
    try:
        # Check if the path exists and contains required files
        if not os.path.exists(mtl_path):
            return None, None
            
        # Look for model files
        model_files = [f for f in os.listdir(mtl_path) if f.endswith('.pt') or f.endswith('.safetensors')]
        if not model_files:
            return None, None
            
        # Load tokenizer and model
        tok = AutoTokenizer.from_pretrained(mtl_path, use_fast=True)
        
        # For MTL, we need to create a custom model structure
        # Since we don't have the exact MTL model architecture, we'll use a fallback
        try:
            # Try to load as a regular model first
            mdl = AutoModelForSequenceClassification.from_pretrained(mtl_path)
        except:
            # If that fails, create a simple MTL model
            mdl = SimpleMTL(mtl_path)
            
        mdl.eval().to(DEVICE)
        return tok, mdl
    except Exception as e:
        st.error(f"Failed to load MTL model from {mtl_path}: {str(e)}")
        return None, None

# =========================
# UI
# =========================
st.title("PR Crisis Sentiment & Emotion – Streamlit Demo")
st.caption("Samsung Galaxy Note 7 crisis • Sentiment & Emotion with token-level attributions (XAI).")

with st.sidebar:
    st.header("Settings")

    # CHOOSE HERE (only place)
    mode_choice = st.radio("Mode:", ["Single-task", "Multi-task"], index=0)
    backbone_choice = st.radio("Backbone:", ["BERTweet", "DistilRoBERTa"], index=0)

    explain = st.checkbox("Show explanations (token attributions)", value=True)
    real_ig_toggle = st.checkbox(
        "Use Integrated Gradients (Captum)",
        value=False and USE_CAPTUM,
        help="Requires `captum`. If not installed, the app will fall back to a simple saliency."
    )

    st.markdown("---")
    st.write("**Notes**")
    st.write("- Keep class orders consistent with training.")
    st.write("- Captum IG is slower but more principled.")

bk = backbone_choice.lower()  # "bertweet" or "distilroberta"

# Reload models when backbone changes
if 'current_backbone' not in st.session_state:
    st.session_state.current_backbone = bk
if st.session_state.current_backbone != bk:
    st.session_state.current_backbone = bk
    sentiment_tok, sentiment_mdl, emotion_tok, emotion_mdl = load_models(bk)
    reload_mtl_for_backbone(bk)

choice = resolve_paths(mode_choice, bk)

tab1, tab2 = st.tabs(["Single Text", "Batch CSV"])
# Reload models when backbone changes
if 'current_backbone' not in st.session_state:
    st.session_state.current_backbone = bk

if st.session_state.current_backbone != bk:
    st.session_state.current_backbone = bk
    sentiment_tok, sentiment_mdl, emotion_tok, emotion_mdl = load_models(bk)
    # Also reload MTL for the new backbone
    reload_mtl_for_backbone(bk)

choice = resolve_paths(mode_choice, bk)

# ---------- Single Text ----------
with tab1:
    txt = st.text_area("Enter a Reddit post/comment", value="My Note 7 overheated again. I'm scared it might explode.", height=140)
    if st.button("Analyze", type="primary"):
        with st.spinner("Running inference..."):
            if choice["mode"] == "st":
                s_tok, s_mdl = load_single_task(choice["sent"])
                e_tok, e_mdl = load_single_task(choice["emo"])
                
                if s_tok is None or s_mdl is None or e_tok is None or e_mdl is None:
                    st.error("Failed to load one or more models")
                    st.stop()
                
                s_label, s_conf, s_probs, s_enc = predict_single(s_mdl, s_tok, txt, SENTIMENT_LABELS)
                e_label, e_conf, e_probs, e_enc = predict_single(e_mdl, e_tok, txt, EMOTION_LABELS)
                
                # Display results
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Sentiment Analysis")
                    st.write(f"**Label:** {s_label}")
                    st.write(f"**Confidence:** {s_conf:.3f}")
                    st.bar_chart(format_probs(SENTIMENT_LABELS, s_probs))
                
                with col2:
                    st.subheader("Emotion Analysis")
                    st.write(f"**Label:** {e_label}")
                    st.write(f"**Confidence:** {e_conf:.3f}")
                    st.bar_chart(format_probs(EMOTION_LABELS, e_probs))
                
                # Show explanations if requested
                if explain:
                    try:
                        st.markdown("### Token Attributions")
                        # Sentiment
                        s_tokens = s_tok.convert_ids_to_tokens(s_enc["input_ids"][0].cpu().tolist())
                        s_scores = attribution_scores(s_mdl_or_mtl, s_tok, s_enc, SENTIMENT_LABELS.index(s_label), real_ig_toggle and USE_CAPTUM, head="sent" if is_mtl else None)
                        st.markdown(html_highlight(s_tokens, s_scores), unsafe_allow_html=True)

                        # Emotion
                        e_tokens = e_tok.convert_ids_to_tokens(e_enc["input_ids"][0].cpu().tolist())
                        e_scores = attribution_scores(e_mdl_or_mtl, e_tok, e_enc, EMOTION_LABELS.index(e_label), real_ig_toggle and USE_CAPTUM, head="emo" if is_mtl else None)
                        st.markdown(html_highlight(e_tokens, e_scores), unsafe_allow_html=True)
                    except Exception as ex:
                        st.warning(f"Explanation rendering failed: {ex}")
            else:
                if not os.path.exists(choice["mtl"]):
                    st.error(f"MTL path not found: {choice['mtl']}")
                    st.stop()
                mtl_tok, mtl_mdl = load_mtl_dir(choice["mtl"])
                
                if mtl_tok is None or mtl_mdl is None:
                    st.error("Failed to load MTL model")
                    st.stop()
                
                (s_label, s_conf, s_probs, s_enc,
                 e_label, e_conf, e_probs, e_enc) = predict_mtl(mtl_mdl, mtl_tok, txt)
                
                # Display results
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Sentiment Analysis")
                    st.write(f"**Label:** {s_label}")
                    st.write(f"**Confidence:** {s_conf:.3f}")
                    st.bar_chart(format_probs(SENTIMENT_LABELS, s_probs))
                
                with col2:
                    st.subheader("Emotion Analysis")
                    st.write(f"**Label:** {e_label}")
                    st.write(f"**Confidence:** {e_conf:.3f}")
                    st.bar_chart(format_probs(EMOTION_LABELS, e_probs))
                
                # Show explanations if requested
                if explain:
                    st.subheader("Token Attributions")
                    st.write("**Sentiment:**")
                    s_scores = attribution_scores(mtl_mdl, mtl_tok, s_enc, 
                                                SENTIMENT_LABELS.index(s_label), real_ig_toggle)
                    s_tokens = mtl_tok.convert_ids_to_tokens(s_enc["input_ids"][0].cpu().tolist())
                    st.markdown(html_highlight(s_tokens, s_scores), unsafe_allow_html=True)
                    
                    st.write("**Emotion:**")
                    e_scores = attribution_scores(mtl_mdl, mtl_tok, e_enc, 
                                                EMOTION_LABELS.index(e_label), real_ig_toggle)
                    e_tokens = mtl_tok.convert_ids_to_tokens(e_enc["input_ids"][0].cpu().tolist())
                    st.markdown(html_highlight(e_tokens, e_scores), unsafe_allow_html=True)

# ---------- Batch ----------
with tab2:
    st.write("Upload a CSV with a **text** column.")
    up = st.file_uploader("CSV", type=["csv"])
    
    if up is not None:
        try:
            df = pd.read_csv(up)
            if "text" not in df.columns:
                st.error("CSV must contain a 'text' column")
            else:
                st.write(f"Loaded {len(df)} texts")
                if st.button("Analyze Batch"):
                    # Process batch analysis here
                    st.info("Batch analysis functionality to be implemented")
        except Exception as e:
            st.error(f"Error reading CSV: {str(e)}")

def html_highlight(tokens: List[str], scores: List[float]) -> str:
    chunks = []
    skip = set(["<s>", "</s>", "[CLS]", "[SEP]", "[PAD]"])
    for tok, s in zip(tokens, scores):
        if tok in skip:
            continue
        # strip common wordpiece prefixes
        if tok.startswith("##"):
            tok = tok[2:]
        if tok in ["Ġ"]:
            continue
        op = min(max(float(s), 0.06), 1.0)  # opacity
        chunks.append(
            f"<span style='background: rgba(255,165,0,{op}); padding:2px 4px; border-radius:4px; margin:2px'>{tok}</span>"
        )
    return "<div style='line-height:2; font-family: ui-sans-serif, system-ui, -apple-system'>" + " ".join(chunks) + "</div>"
