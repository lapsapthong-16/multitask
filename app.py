import os
import io
import torch
import pandas as pd
import numpy as np
import streamlit as st
import captum
from captum.attr import IntegratedGradients

from typing import List, Tuple, Dict
from torch.nn.functional import softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification

st.set_page_config(page_title="PR Crisis Sentiment & Emotion (XAI Demo)", layout="wide")

import torch.nn as nn
from transformers import AutoModel

MTL_DIR = "models/mtl" 

class SimpleMTL(nn.Module):
    def __init__(self, encoder_name_or_path, num_sent=3, num_emo=6):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name_or_path)
        hidden = self.encoder.config.hidden_size
        self.sent_head = nn.Linear(hidden, num_sent)
        self.emo_head  = nn.Linear(hidden, num_emo)

    def forward(self, **enc):
        out = self.encoder(**enc)
        cls = out.last_hidden_state[:, 0, :]
        return self.sent_head(cls), self.emo_head(cls)

@st.cache_resource(show_spinner=True)
def load_mtl():
    mtl_model_path = os.path.join(MTL_DIR, "mtl_model.pt")
    base_tok_path = os.path.join(MTL_DIR, "base_tok")
    
    if not os.path.exists(mtl_model_path) or not os.path.exists(base_tok_path):
        return None, None
    
    try:
        tok = AutoTokenizer.from_pretrained(base_tok_path, use_fast=True)
        mdl = SimpleMTL(base_tok_path)
        state = torch.load(mtl_model_path, map_location=DEVICE)
        mdl.load_state_dict(state, strict=True)
        mdl.eval().to(DEVICE)
        return tok, mdl
    except Exception as e:
        st.error(f"Failed to load MTL model: {str(e)}")
        return None, None

SENTIMENT_BASE = "models/sentiment"     
EMOTION_BASE   = "models/emotion"       

# match the exact class order used during training
SENTIMENT_LABELS = ["Negative", "Neutral", "Positive"]
EMOTION_LABELS   = ["Anger", "Fear", "Joy", "Sadness", "Surprise", "No Emotion"]

MAX_LEN = 128
DEVICE = "cpu" 

USE_CAPTUM = False
try:
    USE_CAPTUM = True
except Exception:
    USE_CAPTUM = False

# Check MTL availability
mtl_available = False
mtl_tok, mtl_mdl = None, None

# Check if MTL models exist for the current backbone
def check_mtl_availability(backbone):
    mtl_path = os.path.join(MTL_DIR, backbone)
    if not os.path.exists(mtl_path):
        return False
    
    model_files = [f for f in os.listdir(mtl_path) if f.endswith('.pt') or f.endswith('.safetensors') or f.endswith('.bin')]
    return len(model_files) > 0

if check_mtl_availability("bertweet"):
    mtl_tok, mtl_mdl = load_mtl()
    if mtl_tok is not None and mtl_mdl is not None:
        mtl_available = True

def reload_mtl_for_backbone(backbone):
    global mtl_tok, mtl_mdl, mtl_available
    if check_mtl_availability(backbone):
        mtl_tok, mtl_mdl = load_mtl_dir(os.path.join(MTL_DIR, backbone))
        mtl_available = mtl_tok is not None and mtl_mdl is not None
    else:
        mtl_tok, mtl_mdl = None, None
        mtl_available = False

@st.cache_resource(show_spinner=True)
def load_models(backbone="bertweet"):
    sentiment_path = os.path.join(SENTIMENT_BASE, backbone)
    emotion_path = os.path.join(EMOTION_BASE, backbone)
    
    # Load sentiment model
    sent_tok, sent_mdl = load_single_task(sentiment_path)
    if sent_tok is None or sent_mdl is None:
        return None, None, None, None
    
    # Load emotion model
    emo_tok, emo_mdl = load_single_task(emotion_path)
    if emo_tok is None or emo_mdl is None:
        return None, None, None, None

    return sent_tok, sent_mdl, emo_tok, emo_mdl

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


def attribution_scores(model, tokenizer, enc, target_idx: int, real_ig: bool, head=None):
    """
    Return token-level attribution scores in [0,1].
    If real_ig=True and Captum available, use Integrated Gradients.
    Else use a one-pass saliency on input embeddings (norm of grad).
    
    Args:
        head: For MTL models, specify 'sent' for sentiment or 'emo' for emotion
    """
    input_ids = enc["input_ids"]
    attention_mask = enc.get("attention_mask", None)
    token_count = input_ids.shape[1]
    
    # Handle different model architectures for embeddings
    if hasattr(model, 'encoder') and hasattr(model.encoder, 'embeddings'):
        # For SimpleMTL models
        embeddings_layer = model.encoder.embeddings.word_embeddings
    elif hasattr(model, 'base_model'):
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
        # True Integrated Gradients - needs more complex handling for MTL
        # For now, fall back to simple saliency for MTL models
        if hasattr(model, 'sent_head') and hasattr(model, 'emo_head'):
            real_ig = False
        else:
            tokens, attn, ids, forward = tokens_and_embeds(model, tokenizer, enc)
            baseline_ids = torch.full_like(ids, fill_value=tokenizer.pad_token_id or tokenizer.eos_token_id)
            baseline_embeds = embeddings_layer(baseline_ids).to(DEVICE)
            input_embeds = embeddings_layer(ids).to(DEVICE)

            ig = IntegratedGradients(lambda e: forward(e)[:, target_idx])
            attributions, _ = ig.attribute(inputs=input_embeds,
                                           baselines=baseline_embeds,
                                           additional_forward_args=None,
                                           n_steps=50, return_convergence_delta=True)
            scores = attributions.norm(p=2, dim=-1).squeeze(0).detach().cpu().numpy()
    
    if not real_ig or not USE_CAPTUM:
        # Lightweight saliency: grad wrt input embeddings (one backward pass)
        for p in model.parameters():
            p.requires_grad_(False)
        input_embeds = embeddings_layer(input_ids).detach().clone().to(DEVICE).requires_grad_(True)
        outputs = model(inputs_embeds=input_embeds, attention_mask=attention_mask)
        
        # Handle MTL models that return tuples vs standard models that return objects
        if hasattr(model, 'sent_head') and hasattr(model, 'emo_head'):
            # MTL model returns (sent_logits, emo_logits)
            sent_logits, emo_logits = outputs
            if head == "sent":
                logits = sent_logits.squeeze(0)
            elif head == "emo":
                logits = emo_logits.squeeze(0)
            else:
                # Default to sentiment if not specified
                logits = sent_logits.squeeze(0)
        else:
            # Standard model returns object with .logits attribute
            logits = outputs.logits.squeeze(0)
            
        # Ensure target_idx is within bounds
        if target_idx >= logits.shape[0]:
            st.warning(f"Target index {target_idx} is out of bounds for logits shape {logits.shape}. Using index 0.")
            target_idx = 0
            
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
        # Check if this is a custom BERTweet model
        config_path = os.path.join(model_path, "config.json")
        is_custom_bertweet = False
        if os.path.exists(config_path):
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)
                # Check if it's a custom BERTweet model that needs special handling
                if config.get("model_name") == "vinai/bertweet-base" and "num_classes" in config:
                    is_custom_bertweet = True
        
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        
        if is_custom_bertweet:
            # For custom BERTweet models, we need to load them differently
            # First try standard loading
            try:
                model = AutoModelForSequenceClassification.from_pretrained(model_path)
            except Exception:
                # If standard loading fails, create a model from vinai/bertweet-base and load the state dict
                from transformers import RobertaForSequenceClassification, RobertaConfig
                
                # Create config based on the saved config
                with open(config_path, 'r') as f:
                    saved_config = json.load(f)
                
                # Create a proper RobertaConfig
                model_config = RobertaConfig(
                    vocab_size=saved_config.get("vocab_size", 64002),
                    hidden_size=saved_config.get("hidden_size", 768),
                    num_hidden_layers=saved_config.get("num_hidden_layers", 12),
                    num_attention_heads=saved_config.get("num_attention_heads", 12),
                    intermediate_size=saved_config.get("intermediate_size", 3072),
                    max_position_embeddings=saved_config.get("max_position_embeddings", 130),
                    num_labels=saved_config.get("num_classes", 3),
                    pad_token_id=saved_config.get("pad_token_id", 1),
                    bos_token_id=saved_config.get("bos_token_id", 0),
                    eos_token_id=saved_config.get("eos_token_id", 2)
                )
                
                model = RobertaForSequenceClassification(model_config)
                
                # Load the state dict
                model_file = os.path.join(model_path, "pytorch_model.bin")
                if os.path.exists(model_file):
                    state_dict = torch.load(model_file, map_location=DEVICE)
                    model.load_state_dict(state_dict, strict=False)
        else:
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
        
        # Load tokenizer
        tok = AutoTokenizer.from_pretrained(mtl_path, use_fast=True)
        
        # Check what type of model we have based on the files present
        has_pytorch_model_bin = os.path.exists(os.path.join(mtl_path, "pytorch_model.bin"))
        has_custom_components = os.path.exists(os.path.join(mtl_path, "custom_components.pt"))
        has_safetensors = os.path.exists(os.path.join(mtl_path, "model.safetensors"))
        
        # Check if this is a custom model type that won't work with standard loading
        config_path = os.path.join(mtl_path, "config.json")
        is_custom_model_type = False
        if os.path.exists(config_path):
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)
                model_type = config.get("model_type", "")
                # Check for custom model types that Transformers doesn't recognize
                if model_type in ["BERTweetMultiTaskTransformer"]:
                    is_custom_model_type = True
        
        if has_pytorch_model_bin and not has_custom_components:
            # BERTweet case: Custom MTL model saved as pytorch_model.bin
            if is_custom_model_type:
                # Skip standard loading for known custom types
                mdl = SimpleMTL("vinai/bertweet-base")
                state_dict = torch.load(os.path.join(mtl_path, "pytorch_model.bin"), map_location=DEVICE)
                mdl.load_state_dict(state_dict, strict=False)
            else:
                try:
                    # Try standard loading for recognized types
                    mdl = AutoModelForSequenceClassification.from_pretrained(mtl_path)
                except Exception as e:
                    # Fallback to SimpleMTL
                    mdl = SimpleMTL("vinai/bertweet-base")
                    state_dict = torch.load(os.path.join(mtl_path, "pytorch_model.bin"), map_location=DEVICE)
                    mdl.load_state_dict(state_dict, strict=False)
                
        elif has_custom_components and has_safetensors:
            # DistilRoBERTa case: Base model + custom components
            try:
                # Load the base model first
                mdl = AutoModelForSequenceClassification.from_pretrained(mtl_path)
            except:
                # Fallback: create SimpleMTL and load components
                mdl = SimpleMTL("distilroberta-base")
                # Load custom components if they exist
                custom_components_path = os.path.join(mtl_path, "custom_components.pt")
                if os.path.exists(custom_components_path):
                    custom_state = torch.load(custom_components_path, map_location=DEVICE)
                    mdl.load_state_dict(custom_state, strict=False)
        else:
            # Fallback: try standard loading
            if is_custom_model_type:
                # Skip standard loading for custom types
                base_model = "vinai/bertweet-base" if "bertweet" in mtl_path.lower() else "distilroberta-base"
                mdl = SimpleMTL(base_model)
            else:
                try:
                    mdl = AutoModelForSequenceClassification.from_pretrained(mtl_path)
                except:
                    # Determine base model from path
                    base_model = "vinai/bertweet-base" if "bertweet" in mtl_path.lower() else "distilroberta-base"
                    mdl = SimpleMTL(base_model)
                
        mdl.eval().to(DEVICE)
        return tok, mdl
        
    except Exception as e:
        st.error(f"Failed to load MTL model from {mtl_path}: {str(e)}")
        return None, None

# Initialize with default backbone
sentiment_tok, sentiment_mdl, emotion_tok, emotion_mdl = load_models("bertweet")

# Check if models loaded successfully
if sentiment_tok is None or sentiment_mdl is None or emotion_tok is None or emotion_mdl is None:
    st.error("Failed to load initial models. Please check the model paths and try again.")
    st.stop()

STOPWORDS = {
    "the","a","an","to","of","and","or","but","if","in","on","at","for","with","as","by",
    "is","am","are","was","were","be","been","being","it","its","i","im","i'm","my","me",
    "you","your","he","she","they","we","our","us","this","that","these","those","again",
    ".",",","!","?","’","'","”","“","(",")","-","—","…"
}

def _aggregate_word_scores(tokens, scores):
    """
    Merge subword tokens into whole words and aggregate scores.
    Handles:
      - BERT WordPiece:        '##word'
      - RoBERTa/BERTweet:     'Ġword' (Ġ = leading space)
      - BPE suffix style:     'wor@@' + 'd'
    Aggregation: max over subpieces.
    """
    words, word_scores = [], []
    cur_word, cur_scores = "", []

    def flush():
        nonlocal cur_word, cur_scores
        if cur_word:
            # strip any lingering artifacts in the completed word
            w = cur_word.replace("@@", "")
            if w:
                words.append(w)
                word_scores.append(max(cur_scores) if cur_scores else 0.0)
        cur_word, cur_scores = "", []

    for tok, s in zip(tokens, scores):
        if tok in {"<s>", "</s>", "[CLS]", "[SEP]", "[PAD]"}:
            continue

        # BERT continuation: ##piece
        if tok.startswith("##"):
            piece = tok[2:]
            cur_word += piece
            cur_scores.append(float(s))
            continue

        # RoBERTa/BERTweet new word marker: Ġword
        if tok.startswith("Ġ"):
            # starting a new word -> flush previous
            flush()
            piece = tok[1:]  # drop Ġ
            # handle empty (rare)
            if piece:
                cur_word = piece
                cur_scores = [float(s)]
            continue

        # BPE suffix style: may end with '@@' meaning "continue with next token"
        if tok.endswith("@@"):
            # start or continue the current word
            cur_word += tok[:-2]  # drop @@
            cur_scores.append(float(s))
            continue

        # plain token (no markers)
        if cur_word:
            # if we're in the middle of a word, continue it
            cur_word += tok
            cur_scores.append(float(s))
        else:
            # start a new word
            cur_word = tok
            cur_scores = [float(s)]

    # flush tail
    flush()

    # remove punctuation-only or empty artifacts
    cleaned = []
    for w, sc in zip(words, word_scores):
        ww = w.strip()
        if not ww:
            continue
        cleaned.append((ww, sc))
    return cleaned

def top_k_contributors(tokens, scores, k=5, min_score=0.06, content_only=True):
    """
    Returns top-k (word, score) after aggregation, desc by score.
    content_only=True removes simple stopwords and 1-char tokens unless high score.
    """
    agg = _aggregate_word_scores(tokens, scores)
    out = []
    for w, sc in agg:
        lw = w.lower()
        if content_only and (lw in STOPWORDS or (len(lw) == 1 and sc < 0.15)):
            continue
        # final display cleanup (defensive)
        w_disp = w.replace("Ġ", "").replace("@@", "")
        out.append((w_disp, sc))
    out.sort(key=lambda x: x[1], reverse=True)
    return out[:k]

def rationale_sentence(label, top_words):
    """Human-friendly one-liner."""
    if not top_words:
        return f"The model predicted **{label}**."
    # prefer 3 terms if available
    words = ", ".join([w for w, _ in top_words[:3]])
    return f"The model predicted **{label}** mainly due to: *{words}*."

def drop_word_once(text, word):
    # remove one occurrence case-insensitively (simple heuristic)
    import re
    return re.sub(rf'\b{re.escape(word)}\b', '', text, count=1, flags=re.IGNORECASE).replace("  ", " ").strip()

def counterfactual_delta(predict_fn, tokenizer, model, text, label_list, chosen_label):
    """
    Re-run prediction after removing the top contributing word and report delta in confidence.
    predict_fn must match: (model, tokenizer, text, labels) -> (label, conf, probs, enc)
    """
    base_label, base_conf, _, _ = predict_fn(model, tokenizer, text, label_list)
    # If top word not supplied here, caller should compute and pass it in; we do a quick recompute instead
    # Caller supplies tokens/scores for chosen_label
    return base_label, base_conf

# =========================
# UI
# =========================
st.title("PR Crisis Sentiment & Emotion")
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
                
                # === Stakeholder-friendly explanations ===
                if explain:
                    st.markdown("### Why the model decided this (Stakeholder view)")

                    # Sentiment
                    s_tokens = s_tok.convert_ids_to_tokens(s_enc["input_ids"][0].cpu().tolist())
                    s_scores = attribution_scores(s_mdl, s_tok, s_enc, SENTIMENT_LABELS.index(s_label), real_ig_toggle and USE_CAPTUM)
                    s_top = top_k_contributors(s_tokens, s_scores, k=5)
                    st.write(rationale_sentence(s_label, s_top))
                    if s_top:
                        st.write("Top words:", "  ".join([f"`{w}` ({sc:.2f})" for w, sc in s_top[:5]]))
                        cf_text = drop_word_once(txt, s_top[0][0])
                        cf_label, cf_conf, _, _ = predict_single(s_mdl, s_tok, cf_text, SENTIMENT_LABELS)
                        st.caption(f"Counterfactual: removing `{s_top[0][0]}` → {cf_label} {cf_conf:.3f} (was {s_label} {s_conf:.3f})")

                    # Emotion
                    e_tokens = e_tok.convert_ids_to_tokens(e_enc["input_ids"][0].cpu().tolist())
                    e_scores = attribution_scores(e_mdl, e_tok, e_enc, EMOTION_LABELS.index(e_label), real_ig_toggle and USE_CAPTUM)
                    e_top = top_k_contributors(e_tokens, e_scores, k=5)
                    st.write(rationale_sentence(e_label, e_top))
                    if e_top:
                        st.write("Top words:", "  ".join([f"`{w}` ({sc:.2f})" for w, sc in e_top[:5]]))
                        cf_text2 = drop_word_once(txt, e_top[0][0])
                        cf_label2, cf_conf2, _, _ = predict_single(e_mdl, e_tok, cf_text2, EMOTION_LABELS)
                        st.caption(f"Counterfactual: removing `{e_top[0][0]}` → {cf_label2} {cf_conf2:.3f} (was {e_label} {e_conf:.3f})")

                    # Advanced: full heatmaps
                    with st.expander("Advanced: full token attributions", expanded=False):
                        st.write("**Sentiment heatmap**")
                        st.markdown(html_highlight(s_tokens, s_scores), unsafe_allow_html=True)
                        st.write("**Emotion heatmap**")
                        st.markdown(html_highlight(e_tokens, e_scores), unsafe_allow_html=True)

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
                
                # === Stakeholder-friendly explanations ===
                if explain:
                    st.markdown("### Why the model decided this (Stakeholder view)")

                    # Sentiment
                    s_scores = attribution_scores(mtl_mdl, mtl_tok, s_enc, SENTIMENT_LABELS.index(s_label), real_ig_toggle, head="sent")
                    s_tokens = mtl_tok.convert_ids_to_tokens(s_enc["input_ids"][0].cpu().tolist())
                    s_top = top_k_contributors(s_tokens, s_scores, k=5)
                    st.write(rationale_sentence(s_label, s_top))
                    if s_top:
                        st.write("Top words:", "  ".join([f"`{w}` ({sc:.2f})" for w, sc in s_top[:5]]))
                        cf_text = drop_word_once(txt, s_top[0][0])
                        cf_s_label, cf_s_conf, *_ = predict_mtl(mtl_mdl, mtl_tok, cf_text)[:2]
                        st.caption(f"Counterfactual: removing `{s_top[0][0]}` → {cf_s_label} {cf_s_conf:.3f} (was {s_label} {s_conf:.3f})")

                    # Emotion
                    e_scores = attribution_scores(mtl_mdl, mtl_tok, e_enc, EMOTION_LABELS.index(e_label), real_ig_toggle, head="emo")
                    e_tokens = mtl_tok.convert_ids_to_tokens(e_enc["input_ids"][0].cpu().tolist())
                    e_top = top_k_contributors(e_tokens, e_scores, k=5)
                    st.write(rationale_sentence(e_label, e_top))
                    if e_top:
                        st.write("Top words:", "  ".join([f"`{w}` ({sc:.2f})" for w, sc in e_top[:5]]))
                        cf_text2 = drop_word_once(txt, e_top[0][0])
                        _sL, _sC, _sp, _se, cf_e_label, cf_e_conf, *_ = predict_mtl(mtl_mdl, mtl_tok, cf_text2)
                        st.caption(f"Counterfactual: removing `{e_top[0][0]}` → {cf_e_label} {cf_e_conf:.3f} (was {e_label} {e_conf:.3f})")

                    # Advanced: full heatmaps
                    with st.expander("Advanced: full token attributions", expanded=False):
                        st.write("**Sentiment heatmap**")
                        st.markdown(html_highlight(s_tokens, s_scores), unsafe_allow_html=True)
                        st.write("**Emotion heatmap**")
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
