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

st.set_page_config(
    page_title="🧠 AI Crisis Sentiment & Emotion Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styles */
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-subtitle {
        font-size: 1.1rem;
        opacity: 0.9;
        font-weight: 400;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Result cards */
    .result-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 4px solid;
        margin: 1rem 0;
        transition: transform 0.2s ease;
    }
    
    .result-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.15);
    }
    
    .sentiment-positive { border-left-color: #28a745; }
    .sentiment-negative { border-left-color: #dc3545; }
    .sentiment-neutral { border-left-color: #6c757d; }
    
    .emotion-joy { border-left-color: #ffc107; }
    .emotion-anger { border-left-color: #dc3545; }
    .emotion-fear { border-left-color: #6f42c1; }
    .emotion-sadness { border-left-color: #17a2b8; }
    .emotion-surprise { border-left-color: #fd7e14; }
    .emotion-no-emotion { border-left-color: #6c757d; }
    
    /* Metrics styling */
    .metric-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin: 1rem 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #666;
        font-weight: 500;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #333;
    }
    
    .confidence-bar {
        height: 8px;
        background: #e9ecef;
        border-radius: 4px;
        margin: 0.5rem 0;
        overflow: hidden;
    }
    
    .confidence-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.5s ease;
    }
    
    /* Enhanced chips */
    .explanation-chips {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin: 1rem 0;
    }
    
    .chip {
        display: inline-flex;
        align-items: center;
        padding: 0.5rem 1rem;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border-radius: 25px;
        font-size: 0.9rem;
        font-weight: 500;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: transform 0.2s ease;
    }
    
    .chip:hover {
        transform: scale(1.05);
    }
    
    .chip-score {
        margin-left: 0.5rem;
        opacity: 0.8;
        font-weight: 400;
    }
    
    /* Sample texts */
    .sample-texts {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .sample-text {
        background: white;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 6px;
        cursor: pointer;
        transition: background 0.2s ease;
        border: 1px solid #e9ecef;
    }
    
    .sample-text:hover {
        background: #e3f2fd;
        border-color: #2196f3;
    }
    
    /* Loading animation */
    .loading-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        padding: 2rem;
    }
    
    .loading-spinner {
        width: 50px;
        height: 50px;
        border: 4px solid #f3f3f3;
        border-top: 4px solid #667eea;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Enhanced heatmap */
    .token-heatmap {
        line-height: 2.5;
        font-family: 'Inter', sans-serif;
        padding: 1rem;
        background: #f8f9fa;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .token {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        margin: 0.1rem;
        border-radius: 6px;
        transition: transform 0.2s ease;
        font-weight: 500;
    }
    
    .token:hover {
        transform: scale(1.1);
        z-index: 10;
        position: relative;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 1rem 2rem;
        border-radius: 10px 10px 0 0;
        font-weight: 600;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-weight: 600;
        border-radius: 8px;
        transition: transform 0.2s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

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

SENTIMENT_LABELS = ["Negative", "Neutral", "Positive"]
EMOTION_LABELS   = ["Anger", "Fear", "Joy", "Sadness", "Surprise", "No Emotion"]

SENTIMENT_EMOJIS = {"Negative": "😞", "Neutral": "😐", "Positive": "😊"}
EMOTION_EMOJIS = {
    "Anger": "😡", "Fear": "😨", "Joy": "😄", 
    "Sadness": "😢", "Surprise": "😲", "No Emotion": "😶"
}

MAX_LEN = 128
DEVICE = "cpu" 

USE_CAPTUM = False
try:
    USE_CAPTUM = True
except Exception:
    USE_CAPTUM = False

SAMPLE_TEXTS = [
    "My Note 7 overheated again. I'm scared it might explode.",
    "Samsung's response to the crisis was quick and professional.",
    "I'm disappointed with the recall process. It took forever to get a replacement.",
    "The new Galaxy S8 looks amazing! Can't wait to upgrade.",
    "I've lost all trust in Samsung after this incident.",
    "The battery issue is concerning but I still love my Samsung phone.",
]

mtl_available = False
mtl_tok, mtl_mdl = None, None

def check_mtl_availability(backbone):
    mtl_path = os.path.join(MTL_DIR, backbone)
    if not os.path.exists(mtl_path):
        return False
    
    model_files = [f for f in os.listdir(mtl_path) if f.endswith('.pt') or f.endswith('.safetensors') or f.endswith('.bin')]
    return len(model_files) > 0

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

def predict_single(model, tokenizer, text: str, labels: List[str], max_len=MAX_LEN):
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len)
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc)
        logits = out.logits
        
        if logits.shape[-1] == len(labels):
            probs = softmax(logits, dim=-1).squeeze(0)
        elif logits.shape[-1] == 2 and len(labels) == 3:
            binary_probs = softmax(logits, dim=-1).squeeze(0)
            neutral_prob = 0.1  
            neg_prob = binary_probs[0].item() * (1 - neutral_prob)
            pos_prob = binary_probs[1].item() * (1 - neutral_prob)
            probs = torch.tensor([neg_prob, neutral_prob, pos_prob])
        elif logits.shape[-1] == 2 and len(labels) == 6:
            binary_probs = softmax(logits, dim=-1).squeeze(0)
            # Map binary to emotions: assume [negative_emotion, positive_emotion]
            # Distribute across emotion categories
            neg_emotions = ["Anger", "Fear", "Sadness"]  # negative emotions
            pos_emotions = ["Joy", "Surprise"]  # positive emotions
            neutral_emotions = ["No Emotion"]  # neutral
            
            probs_list = []
            neg_prob = binary_probs[0].item() / len(neg_emotions)
            pos_prob = binary_probs[1].item() / len(pos_emotions)
            neutral_prob = 0.1
            
            for label in labels:
                if label in neg_emotions:
                    probs_list.append(neg_prob)
                elif label in pos_emotions:
                    probs_list.append(pos_prob)
                else:  # neutral
                    probs_list.append(neutral_prob)
            
            probs = torch.tensor(probs_list)
            probs = probs / probs.sum()
        else:
            if logits.shape[-1] < len(labels):
                padding = torch.full((logits.shape[0], len(labels) - logits.shape[-1]), -10.0).to(DEVICE)
                logits = torch.cat([logits, padding], dim=-1)
            else:
                # Truncate
                logits = logits[:, :len(labels)]
            probs = softmax(logits, dim=-1).squeeze(0)
        
        conf, idx = torch.max(probs, dim=-1)
        idx = min(idx.item(), len(labels) - 1)
        
    return labels[idx], float(conf.item()), probs.tolist(), enc

def predict_mtl(mtl_model, tokenizer, text: str):
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LEN)
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    
    with torch.no_grad():
        try:
            if hasattr(mtl_model, 'sent_head') and hasattr(mtl_model, 'emo_head'):
                s_logits, e_logits = mtl_model(**enc)
            else:
                outputs = mtl_model(**enc)
                logits = outputs.logits
                
                print(f"MTL fallback logits shape: {logits.shape}")
                
                if logits.shape[1] >= 9:  # 3 sentiment + 6 emotion
                    s_logits = logits[:, :3]
                    e_logits = logits[:, 3:9]
                elif logits.shape[1] >= 6:  # Just emotion classes
                    s_logits = logits[:, :3] if logits.shape[1] >= 3 else logits
                    e_logits = logits[:, :6]
                else:
                    s_logits = logits if logits.shape[1] >= 3 else torch.zeros(1, 3).to(DEVICE)
                    e_logits = logits if logits.shape[1] >= 6 else torch.zeros(1, 6).to(DEVICE)
            
            # Handle sentiment logits shape
            if s_logits.shape[-1] != len(SENTIMENT_LABELS):
                st.warning(f"⚠️ Sentiment logits shape mismatch: {s_logits.shape[-1]} vs {len(SENTIMENT_LABELS)}")
                if s_logits.shape[-1] < len(SENTIMENT_LABELS):
                    padding = torch.zeros(s_logits.shape[0], len(SENTIMENT_LABELS) - s_logits.shape[-1]).to(DEVICE)
                    s_logits = torch.cat([s_logits, padding], dim=-1)
                else:
                    s_logits = s_logits[:, :len(SENTIMENT_LABELS)]
            
            # Handle emotion logits shape
            if e_logits.shape[-1] != len(EMOTION_LABELS):
                st.warning(f"⚠️ Emotion logits shape mismatch: {e_logits.shape[-1]} vs {len(EMOTION_LABELS)}")
                if e_logits.shape[-1] < len(EMOTION_LABELS):
                    padding = torch.zeros(e_logits.shape[0], len(EMOTION_LABELS) - e_logits.shape[-1]).to(DEVICE)
                    e_logits = torch.cat([e_logits, padding], dim=-1)
                else:
                    e_logits = e_logits[:, :len(EMOTION_LABELS)]
            
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
    input_ids = enc["input_ids"]
    attention_mask = enc.get("attention_mask", None)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().tolist())

    def forward_embeds(embeds):
        outputs = model(inputs_embeds=embeds, attention_mask=attention_mask)
        return outputs.logits

    return tokens, attention_mask, input_ids, forward_embeds


def attribution_scores(model, tokenizer, enc, target_idx: int, real_ig: bool, head=None):
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
        for p in model.parameters():
            p.requires_grad_(False)
        input_embeds = embeddings_layer(input_ids).detach().clone().to(DEVICE).requires_grad_(True)
        outputs = model(inputs_embeds=input_embeds, attention_mask=attention_mask)
        
        if hasattr(model, 'sent_head') and hasattr(model, 'emo_head'):
            sent_logits, emo_logits = outputs
            if head == "sent":
                logits = sent_logits.squeeze(0)
            elif head == "emo":
                logits = emo_logits.squeeze(0)
            else:
                logits = sent_logits.squeeze(0)
        else:
            logits = outputs.logits.squeeze(0)
            
        if target_idx >= logits.shape[0]:
            target_idx = min(target_idx, logits.shape[0] - 1)
            
        logits[target_idx].backward()
        grads = input_embeds.grad.detach()  
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
        if tok.startswith("##"):
            tok = tok[2:]
        if tok == "Ġ":
            continue
        opacity = min(max(float(s), 0.06), 1.0)  
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
    try:
        config_path = os.path.join(model_path, "config.json")
        is_custom_bertweet = False
        if os.path.exists(config_path):
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)
                if config.get("model_name") == "vinai/bertweet-base" and "num_classes" in config:
                    is_custom_bertweet = True
        
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        
        if is_custom_bertweet:
            # For custom BERTweet models, we need to load them differently
            try:
                model = AutoModelForSequenceClassification.from_pretrained(model_path)
            except Exception:
                from transformers import RobertaForSequenceClassification, RobertaConfig
                
                with open(config_path, 'r') as f:
                    saved_config = json.load(f)
                
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
    try:
        if not os.path.exists(mtl_path):
            return None, None

        is_distil = "distilroberta" in mtl_path.lower()
        base_name = "distilroberta-base" if is_distil else "vinai/bertweet-base"

        # --- tokenizer with fallbacks (local only) ---
        tok = None
        try:
            tok = AutoTokenizer.from_pretrained(mtl_path, use_fast=True)
        except Exception:
            base_tok_dir = os.path.join(MTL_DIR, "base_tok")
            if os.path.isdir(base_tok_dir):
                try:
                    tok = AutoTokenizer.from_pretrained(base_tok_dir, use_fast=True)
                except Exception:
                    tok = None
            if tok is None:
                st.error(f"Could not load tokenizer from {mtl_path} or {base_tok_dir}. Please ensure tokenizer files are present locally.")
                return None, None

        has_pt_bin      = os.path.exists(os.path.join(mtl_path, "pytorch_model.bin"))
        has_safetensors = os.path.exists(os.path.join(mtl_path, "model.safetensors"))
        cfg_path        = os.path.join(mtl_path, "config.json")

        def _is_single_head_9label():
            try:
                import json
                with open(cfg_path, "r") as f:
                    cfg = json.load(f)
                return int(cfg.get("num_labels", -1)) == 9
            except Exception:
                return False

        # --- Case A: single 9-label head ---
        if (has_pt_bin or has_safetensors) and _is_single_head_9label():
            mdl = AutoModelForSequenceClassification.from_pretrained(mtl_path)
            mdl.eval().to(DEVICE)
            return tok, mdl

        # --- Case B: custom multi-head state_dict ---
        candidate_files = []
        for fname in ["custom_components.pt", "mtl_model.pt", "state_dict.pt", "pytorch_model.bin"]:
            fpath = os.path.join(mtl_path, fname)
            if os.path.exists(fpath):
                candidate_files.append(fpath)

        if candidate_files:
            try:
                mdl = SimpleMTL(mtl_path, num_sent=3, num_emo=6)
            except Exception:
                mdl = SimpleMTL(base_name, num_sent=3, num_emo=6)
                
            for f in candidate_files:
                try:
                    try:
                        sd = torch.load(f, map_location=DEVICE, weights_only=True)
                    except Exception:
                        # Fall back to weights_only=False for older checkpoint formats
                        sd = torch.load(f, map_location=DEVICE, weights_only=False)
                    
                    mdl.load_state_dict(sd, strict=False)  # tolerate partial/nested keys
                    mdl.eval().to(DEVICE)
                    return tok, mdl
                except Exception as e:
                    continue

        if has_pt_bin or has_safetensors:
            try:
                # For MTL models, we need 9 labels (3 sentiment + 6 emotion)
                mdl = AutoModelForSequenceClassification.from_pretrained(mtl_path)
                
                # If the model doesn't have 9 labels, create one with the right number
                if mdl.config.num_labels != 9:
                    from transformers import AutoConfig
                    config = AutoConfig.from_pretrained(mtl_path)
                    config.num_labels = 9
                    mdl = AutoModelForSequenceClassification.from_pretrained(
                        mtl_path, config=config, ignore_mismatched_sizes=True
                    )
                
                mdl.eval().to(DEVICE)
                return tok, mdl
            except Exception as e:
                st.warning(f"Could not load as HF classifier: {str(e)}")
                pass

        st.error(f"Could not load a valid MTL checkpoint from: {mtl_path}")
        return None, None

    except Exception as e:
        st.error(f"Failed to load MTL model from {mtl_path}: {str(e)}")
        return None, None

# Initialize with default backbone
sentiment_tok, sentiment_mdl, emotion_tok, emotion_mdl = load_models("bertweet")

def ensure_mtl(backbone: str):
    tok, mdl = load_mtl_dir(os.path.join(MTL_DIR, backbone))
    ok = tok is not None and mdl is not None
    return ok, tok, mdl

sentiment_tok, sentiment_mdl, emotion_tok, emotion_mdl = load_models("bertweet")

mtl_available, mtl_tok, mtl_mdl = ensure_mtl("bertweet")

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
    words, word_scores = [], []
    cur_word, cur_scores = "", []
    wp_open = False   # WordPiece continuation (##)
    bpe_open = False  # BPE continuation (@@)

    def flush():
        nonlocal cur_word, cur_scores, wp_open, bpe_open
        if cur_word:
            w = cur_word.replace("@@", "")
            if w:
                words.append(w)
                word_scores.append(max(cur_scores) if cur_scores else 0.0)
        cur_word, cur_scores = "", []
        wp_open = False
        bpe_open = False

    for tok, s in zip(tokens, scores):
        if tok in {"<s>", "</s>", "[CLS]", "[SEP]", "[PAD]"}:
            continue
        s = float(s)

        # Case 1: WordPiece continuation (##piece)
        if tok.startswith("##"):
            piece = tok[2:]
            if not cur_word:
                cur_word = piece
            else:
                cur_word += piece
            cur_scores.append(s)
            wp_open = True
            bpe_open = False
            continue

        # Case 2: RoBERTa/BERTweet new word marker (Ġword)
        if tok.startswith("Ġ"):
            flush()
            piece = tok[1:]
            if piece:
                cur_word = piece
                cur_scores = [s]
            continue

        # Case 3: BPE suffix (wor@@)
        if tok.endswith("@@"):
            piece = tok[:-2]
            if not cur_word:
                cur_word = piece
            else:
                cur_word += piece
            cur_scores.append(s)
            bpe_open = True
            wp_open = False
            continue

        # Case 4: plain token
        if wp_open or bpe_open:
            # we were in a continuation -> this plain token finishes the word
            cur_word += tok
            cur_scores.append(s)
            flush()
        else:
            # start a brand-new word
            flush()
            cur_word = tok
            cur_scores = [s]

    flush()

    # final cleanup: remove empties
    cleaned = [(w.strip(), sc) for (w, sc) in zip(words, word_scores) if w.strip()]
    return cleaned


def top_k_contributors(tokens, scores, k=3, min_score=0.07, content_only=True):

    agg = _aggregate_word_scores(tokens, scores)
    out = []
    for w, sc in agg:
        lw = w.lower()
        if content_only and (lw in STOPWORDS or (len(lw) == 1 and sc < 0.2)):
            continue
        out.append((w, sc))
    out.sort(key=lambda x: x[1], reverse=True)
    return out[:k]


def rationale_sentence(label, top_words):
    if not top_words:
        return f"The model predicted **{label}**."
    terms = ", ".join([w for w, _ in top_words[:3]])
    return f"The model predicted **{label}** mainly due to: *{terms}*."


def render_chips(pairs):
    """Nice inline chips for top words."""
    if not pairs:
        return ""
    chips = []
    for w, sc in pairs:
        chips.append(
            f"<span style='display:inline-block; margin:2px; padding:2px 8px; "
            f"border-radius:999px; background:#183c3c; border:1px solid #256b6b; "
            f"font-size:0.9rem;'>{w} <span style='opacity:.7;'>({sc:.2f})</span></span>"
        )
    return "<div>" + " ".join(chips) + "</div>"

def drop_word_once(text, word):
    # remove one occurrence case-insensitively (simple heuristic)
    import re
    return re.sub(rf'\b{re.escape(word)}\b', '', text, count=1, flags=re.IGNORECASE).replace("  ", " ").strip()

def counterfactual_delta(predict_fn, tokenizer, model, text, label_list, chosen_label):
    base_label, base_conf, _, _ = predict_fn(model, tokenizer, text, label_list)
    return base_label, base_conf

def create_result_card(label, confidence, probs, label_type="sentiment"):
    emoji = SENTIMENT_EMOJIS.get(label, "🤔") if label_type == "sentiment" else EMOTION_EMOJIS.get(label, "🤔")
    
    if label_type == "sentiment":
        card_class = f"sentiment-{label.lower()}"
    else:
        card_class = f"emotion-{label.lower().replace(' ', '-')}"
    
    conf_percentage = confidence * 100
    conf_color = "#28a745" if confidence > 0.7 else "#ffc107" if confidence > 0.4 else "#dc3545"
    
    card_html = f"""
    <div class="result-card {card_class}">
        <div class="metric-container">
            <div>
                <div class="metric-label">{label_type.title()} Analysis</div>
                <div class="metric-value">{emoji} {label}</div>
            </div>
            <div style="text-align: right;">
                <div class="metric-label">Confidence</div>
                <div class="metric-value" style="color: {conf_color};">{conf_percentage:.1f}%</div>
            </div>
        </div>
        <div class="confidence-bar">
            <div class="confidence-fill" style="width: {conf_percentage}%; background: {conf_color};"></div>
        </div>
    </div>
    """
    
    return card_html.strip()

def create_explanation_chips(top_words):
    if not top_words:
        return ""
    parts = ['<div class="explanation-chips">']
    for word, score in top_words:
        parts.append(f'<div class="chip">{word}<span class="chip-score">{score:.2f}</span></div>')
    parts.append('</div>')
    return "".join(parts)

def enhanced_html_highlight(tokens: List[str], scores: List[float]) -> str:
    chunks = []
    skip = {"<s>", "</s>", "[CLS]", "[SEP]", "[PAD]"}
    
    # Filter out special tokens and get valid scores for better color mapping
    valid_scores = []
    valid_tokens = []
    
    for tok, s in zip(tokens, scores):
        if tok in skip:
            continue
        # Clean token
        clean_tok = tok
        if tok.startswith("##"):
            clean_tok = tok[2:]
        if clean_tok == "Ġ" or clean_tok == "":
            continue
        valid_scores.append(s)
        valid_tokens.append(clean_tok)
    
    if not valid_scores:
        return '<div class="token-heatmap">No valid tokens to display</div>'
    
    # Calculate score statistics for better color distribution
    min_score = min(valid_scores)
    max_score = max(valid_scores)
    score_range = max_score - min_score if max_score > min_score else 1.0
    
    # Rebuild with proper coloring
    token_idx = 0
    for tok, s in zip(tokens, scores):
        if tok in skip:
            continue
        
        # Clean token
        if tok.startswith("##"):
            tok = tok[2:]
        if tok == "Ġ" or tok == "":
            continue
            
        # Normalize score relative to the actual distribution
        normalized_score = (s - min_score) / score_range if score_range > 0 else 0.5
        
        # Create a smooth color gradient: low importance (light blue) → high importance (red)
        if normalized_score < 0.33:
            # Low importance: Light blue
            intensity = normalized_score * 3  # Scale to 0-1
            r, g, b = int(220 - 70 * intensity), int(235 - 60 * intensity), 255
            opacity = 0.4 + 0.3 * intensity  # 0.4 to 0.7
        elif normalized_score < 0.67:
            # Medium importance: Blue to orange
            intensity = (normalized_score - 0.33) * 3  # Scale to 0-1
            r, g, b = int(150 + 105 * intensity), int(175 + 80 * intensity), int(255 - 155 * intensity)
            opacity = 0.6 + 0.2 * intensity  # 0.6 to 0.8
        else:
            # High importance: Orange to red
            intensity = (normalized_score - 0.67) * 3  # Scale to 0-1
            r, g, b = 255, int(255 - 100 * intensity), int(100 - 100 * intensity)
            opacity = 0.7 + 0.3 * intensity  # 0.7 to 1.0
        
        color = f"rgba({r}, {g}, {b}, {opacity})"
        
        # Add a subtle border for better definition
        border_opacity = min(opacity + 0.2, 1.0)
        border_color = f"rgba({max(r-30, 0)}, {max(g-30, 0)}, {max(b-30, 0)}, {border_opacity})"
        
        chunks.append(f'<span class="token" style="background: {color}; border: 1px solid {border_color};" title="Score: {s:.3f}">{tok}</span>')
        token_idx += 1
    
    return f'<div class="token-heatmap">{"".join(chunks)}</div>'

def create_importance_legend() -> str:
    return """
    <div style="margin: 10px 0; padding: 15px; background: #f8f9fa; border-radius: 8px; border: 1px solid #e9ecef;">
        <div style="font-weight: 600; margin-bottom: 10px; color: #495057;">🎨 Token Importance Legend:</div>
        <div style="display: flex; align-items: center; gap: 20px; flex-wrap: wrap;">
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="background: rgba(220, 235, 255, 0.6); padding: 4px 8px; border-radius: 4px; border: 1px solid rgba(150, 205, 255, 0.8); font-size: 12px;">Low</span>
                <span style="color: #6c757d; font-size: 14px;">Less Important</span>
            </div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="background: rgba(255, 175, 100, 0.7); padding: 4px 8px; border-radius: 4px; border: 1px solid rgba(225, 145, 70, 0.9); font-size: 12px;">Med</span>
                <span style="color: #6c757d; font-size: 14px;">Moderately Important</span>
            </div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="background: rgba(255, 87, 87, 0.9); padding: 4px 8px; border-radius: 4px; border: 1px solid rgba(225, 57, 57, 1.0); font-size: 12px;">High</span>
                <span style="color: #6c757d; font-size: 14px;">Most Important</span>
            </div>
        </div>
    </div>
    """

def display_colored_probability_chart(labels, probs_list, chart_type="sentiment", title="Probability Distribution"):
    try:
        import plotly.express as px
        import pandas as pd
        
        # Debug information
        print(f"Labels: {labels} (length: {len(labels)})")
        print(f"Probs: {probs_list} (length: {len(probs_list)})")
        
        # Ensure labels and probs_list have the same length
        if len(labels) != len(probs_list):
            st.error(f"❌ Data length mismatch in {chart_type} chart:")
            st.write(f"• Expected {len(labels)} probabilities for labels: {labels}")
            st.write(f"• Got {len(probs_list)} probabilities: {probs_list}")
            
            # Try to fix by truncating or padding
            min_len = min(len(labels), len(probs_list))
            if min_len > 0:
                st.info(f"🔧 Using first {min_len} items for chart")
                labels = labels[:min_len]
                probs_list = probs_list[:min_len]
            else:
                st.error("Cannot create chart with no valid data")
                return
        
        # Define colors for different categories
        if chart_type == "sentiment":
            color_map = {
                "Positive": "#28a745",  # Green
                "Neutral": "#6c757d",   # Gray
                "Negative": "#dc3545"   # Red
            }
        else:  # emotion
            color_map = {
                "Joy": "#ffc107",       # Yellow/Gold
                "Anger": "#dc3545",     # Red
                "Fear": "#6f42c1",      # Purple
                "Sadness": "#007bff",   # Blue
                "Surprise": "#fd7e14",  # Orange
                "No Emotion": "#6c757d" # Gray
            }
        
        # Create DataFrame
        df = pd.DataFrame({
            "Label": labels,
            "Probability": [float(p) for p in probs_list]
        })
        
        # Create colored bar chart
        fig = px.bar(
            df, 
            x="Label", 
            y="Probability",
            color="Label",
            color_discrete_map=color_map,
            title=title
        )
        
        # Customize the chart
        fig.update_layout(
            showlegend=False,
            height=350,
            xaxis_title="Categories",
            yaxis_title="Probability",
            title_x=0.5,
            font=dict(size=12),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=50, b=50, l=50, r=50)
        )
        
        fig.update_traces(
            texttemplate='%{y:.3f}',
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Probability: %{y:.3f}<extra></extra>'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except ImportError:
        # Fallback to regular bar chart if plotly is not available
        st.bar_chart(format_probs(labels, probs_list))
        st.info("💡 Install plotly for colored probability charts: `pip install plotly`")
    except Exception as e:
        # General error handling
        st.error(f"❌ Error creating {chart_type} probability chart: {str(e)}")
        st.write(f"Debug info:")
        st.write(f"• Labels ({len(labels)}): {labels}")
        st.write(f"• Probabilities ({len(probs_list)}): {probs_list}")
        
        # Try fallback
        try:
            min_len = min(len(labels), len(probs_list))
            if min_len > 0:
                st.info("🔄 Attempting fallback chart...")
                st.bar_chart(format_probs(labels[:min_len], probs_list[:min_len]))
            else:
                st.error("Cannot create fallback chart - no valid data")
        except Exception as fallback_error:
            st.error(f"Fallback chart also failed: {str(fallback_error)}")

def show_sample_texts():
    st.markdown("### 💡 Try these sample texts:")
    
    cols = st.columns(2)
    for i, sample in enumerate(SAMPLE_TEXTS):
        with cols[i % 2]:
            if st.button(f"📝 {sample[:50]}...", key=f"sample_{i}", help=sample):
                st.session_state.sample_text = sample

# Header
st.markdown("""
<div class="main-header">
    <div class="main-title">🧠 AI Crisis Sentiment Analyzer</div>
    <div class="main-subtitle">
        Sentiment & emotion analysis with explainable AI for crisis communication
    </div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    
    st.markdown("### 🤖 Model Settings")
    mode_choice = st.radio(
        "Analysis Mode:",
        ["Single-task", "Multi-task"],
        index=0,
        help="Single-task: Separate models for sentiment and emotion\nMulti-task: One model for both tasks"
    )
    
    backbone_choice = st.radio(
        "AI Backbone:",
        ["BERTweet", "DistilRoBERTa"],
        index=0,
        help="BERTweet: Optimized for social media text\nDistilRoBERTa: Faster, general-purpose model"
    )

    st.markdown("### 🔍 Explanation Settings")
    explain = st.checkbox("Show AI explanations", value=True, help="Display token-level attributions")

    st.markdown("---")
    
    st.markdown("### 📊 Model Status")
    if sentiment_tok and sentiment_mdl:
        st.success("✅ Models loaded successfully")
    else:
        st.error("❌ Model loading failed")
    
    # Quick stats
    st.markdown("### 📈 Quick Stats")
    st.metric("Supported Languages", "English")
    st.metric("Processing Speed", "< 10s")

bk = backbone_choice.lower().replace('bertweet', 'bertweet').replace('distilroberta', 'distilroberta')

if 'mtl_bootstrapped' not in st.session_state:
    st.session_state.mtl_bootstrapped = True
    mtl_available, mtl_tok, mtl_mdl = ensure_mtl(bk)

if 'current_backbone' not in st.session_state:
    st.session_state.current_backbone = bk

if st.session_state.current_backbone != bk:
    st.session_state.current_backbone = bk
    sentiment_tok, sentiment_mdl, emotion_tok, emotion_mdl = load_models(bk)
    mtl_available, mtl_tok, mtl_mdl = ensure_mtl(bk)

choice = resolve_paths(mode_choice, bk)

tab1, tab2 = st.tabs(["🔍 Single Analysis", "ℹ️ About"])

with tab1:
    show_sample_texts()
    
    st.markdown("### 📝 Enter your text:")
    
    if 'sample_text' not in st.session_state:
        st.session_state.sample_text = "My Note 7 overheated again. I'm scared it might explode."
    
    txt = st.text_area(
        "Text to analyze:",
        value=st.session_state.sample_text,
        height=120,
        placeholder="Enter a social media post, comment, or any text to analyze...",
        help="Try entering text related to product reviews, customer feedback, or social media posts"
    )
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_button = st.button("🚀 Analyze Text", type="primary", use_container_width=True)
    
    if analyze_button and txt.strip():
        with st.spinner("🤖 AI is analyzing your text..."):
            if choice["mode"] == "st":
                # ... (existing single-task logic with enhanced display) ...
                s_tok, s_mdl = load_single_task(choice["sent"])
                e_tok, e_mdl = load_single_task(choice["emo"])
                
                if s_tok is None or s_mdl is None or e_tok is None or e_mdl is None:
                    st.error("❌ Failed to load models")
                    st.stop()
                
                s_label, s_conf, s_probs, s_enc = predict_single(s_mdl, s_tok, txt, SENTIMENT_LABELS)
                e_label, e_conf, e_probs, e_enc = predict_single(e_mdl, e_tok, txt, EMOTION_LABELS)
                
                # Enhanced results display
                st.markdown("## 📊 Analysis Results")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(create_result_card(s_label, s_conf, s_probs, "sentiment"), unsafe_allow_html=True)
                    
                with col2:
                    st.markdown(create_result_card(e_label, e_conf, e_probs, "emotion"), unsafe_allow_html=True)
                
                # Enhanced explanations
                if explain:
                    st.markdown("## 🔍 AI Explanation")
                    st.markdown("*Understanding why the AI made these predictions!*")
                    
                    # Sentiment explanation
                    s_tokens = s_tok.convert_ids_to_tokens(s_enc["input_ids"][0].cpu().tolist())
                    s_scores = attribution_scores(s_mdl, s_tok, s_enc, SENTIMENT_LABELS.index(s_label), False)
                    s_top = top_k_contributors(s_tokens, s_scores, k=3)
                    
                    st.markdown(f"### {SENTIMENT_EMOJIS[s_label]} Sentiment: {s_label}")
                    st.write(rationale_sentence(s_label, s_top))
                    if s_top:
                        st.markdown(create_explanation_chips(s_top), unsafe_allow_html=True)
                        
                        # Counterfactual analysis
                        with st.expander("🔄 Counterfactual Analysis", expanded=False):
                            cf_text = drop_word_once(txt, s_top[0][0])
                            cf_label, cf_conf, _, _ = predict_single(s_mdl, s_tok, cf_text, SENTIMENT_LABELS)
                            st.info(f"If we remove **'{s_top[0][0]}'**: {SENTIMENT_EMOJIS[cf_label]} {cf_label} ({cf_conf:.1%} confidence)")
                    
                    # Emotion explanation
                    e_tokens = e_tok.convert_ids_to_tokens(e_enc["input_ids"][0].cpu().tolist())
                    e_scores = attribution_scores(e_mdl, e_tok, e_enc, EMOTION_LABELS.index(e_label), False)
                    e_top = top_k_contributors(e_tokens, e_scores, k=3)
                    
                    st.markdown(f"### {EMOTION_EMOJIS[e_label]} Emotion: {e_label}")
                    st.write(rationale_sentence(e_label, e_top))
                    if e_top:
                        st.markdown(create_explanation_chips(e_top), unsafe_allow_html=True)
                        
                        # Counterfactual analysis
                        with st.expander("🔄 Counterfactual Analysis", expanded=False):
                            cf_text2 = drop_word_once(txt, e_top[0][0])
                            cf_label2, cf_conf2, _, _ = predict_single(e_mdl, e_tok, cf_text2, EMOTION_LABELS)
                            st.info(f"If we remove **'{e_top[0][0]}'**: {EMOTION_EMOJIS[cf_label2]} {cf_label2} ({cf_conf2:.1%} confidence)")
                    
                    # Advanced visualizations
                    with st.expander("🎨 Advanced Token Visualization", expanded=False):
                        st.markdown("**Sentiment Token Importance**")
                        st.markdown(enhanced_html_highlight(s_tokens, s_scores), unsafe_allow_html=True)
                        st.markdown(create_importance_legend(), unsafe_allow_html=True)
                        
                        st.markdown("**Emotion Token Importance**")
                        st.markdown(enhanced_html_highlight(e_tokens, e_scores), unsafe_allow_html=True)
                        st.markdown(create_importance_legend(), unsafe_allow_html=True)
                        
                        # Probability distributions
                        st.markdown("**Detailed Probability Distributions**")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("*Sentiment Distribution*")
                            display_colored_probability_chart(SENTIMENT_LABELS, s_probs, "sentiment", "Sentiment Probabilities")
                        with col2:
                            st.markdown("*Emotion Distribution*")
                            display_colored_probability_chart(EMOTION_LABELS, e_probs, "emotion", "Emotion Probabilities")

            else:
                # --- Multi-task path ---
                mtl_available, mtl_tok, mtl_mdl = ensure_mtl(bk)
                if not mtl_available:
                    st.error(f"❌ MTL model not found for backbone '{bk}'. Expected at: {os.path.join(MTL_DIR, bk)}")
                    st.stop()

                s_label, s_conf, s_probs, s_enc, e_label, e_conf, e_probs, e_enc = predict_mtl(
                    mtl_mdl, mtl_tok, txt
                )

                st.markdown("## 📊 Analysis Results")
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(create_result_card(s_label, s_conf, s_probs, "sentiment"), unsafe_allow_html=True)
                with c2:
                    st.markdown(create_result_card(e_label, e_conf, e_probs, "emotion"), unsafe_allow_html=True)

                if explain:
                    st.markdown("## 🔍 AI Explanation")
                    st.markdown("*Understanding why the AI made these predictions!*")

                    # Sentiment explanation
                    s_tokens = mtl_tok.convert_ids_to_tokens(s_enc["input_ids"][0].cpu().tolist())
                    s_scores = attribution_scores(mtl_mdl, mtl_tok, s_enc,
                                                  SENTIMENT_LABELS.index(s_label),
                                                  False, head="sent")
                    s_top = top_k_contributors(s_tokens, s_scores, k=3)
                    st.markdown(f"### {SENTIMENT_EMOJIS[s_label]} Sentiment: {s_label}")
                    st.write(rationale_sentence(s_label, s_top))
                    if s_top:
                        st.markdown(create_explanation_chips(s_top), unsafe_allow_html=True)
                        with st.expander("🔄 Counterfactual Analysis", expanded=False):
                            cf_text = drop_word_once(txt, s_top[0][0])
                            cf_s_label, cf_s_conf, _, _, _, _, _, _ = predict_mtl(mtl_mdl, mtl_tok, cf_text)
                            st.info(f"If we remove **'{s_top[0][0]}'**: {SENTIMENT_EMOJIS.get(cf_s_label,'🤔')} {cf_s_label} ({cf_s_conf:.1%} confidence)")

                    # Emotion explanation
                    e_tokens = mtl_tok.convert_ids_to_tokens(e_enc["input_ids"][0].cpu().tolist())
                    e_scores = attribution_scores(mtl_mdl, mtl_tok, e_enc,
                                                  EMOTION_LABELS.index(e_label),
                                                  False, head="emo")
                    e_top = top_k_contributors(e_tokens, e_scores, k=3)
                    st.markdown(f"### {EMOTION_EMOJIS[e_label]} Emotion: {e_label}")
                    st.write(rationale_sentence(e_label, e_top))
                    if e_top:
                        st.markdown(create_explanation_chips(e_top), unsafe_allow_html=True)
                        with st.expander("🔄 Counterfactual Analysis", expanded=False):
                            cf_text2 = drop_word_once(txt, e_top[0][0])
                            _, _, _, _, cf_e_label, cf_e_conf, _, _ = predict_mtl(mtl_mdl, mtl_tok, cf_text2)
                            st.info(f"If we remove **'{e_top[0][0]}'**: {EMOTION_EMOJIS.get(cf_e_label,'🤔')} {cf_e_label} ({cf_e_conf:.1%} confidence)")

                    with st.expander("🎨 Advanced Token Visualization", expanded=False):
                        st.markdown("**Sentiment Token Importance**")
                        st.markdown(enhanced_html_highlight(s_tokens, s_scores), unsafe_allow_html=True)
                        st.markdown(create_importance_legend(), unsafe_allow_html=True)
                        
                        st.markdown("**Emotion Token Importance**")
                        st.markdown(enhanced_html_highlight(e_tokens, e_scores), unsafe_allow_html=True)
                        st.markdown(create_importance_legend(), unsafe_allow_html=True)

                        st.markdown("**Detailed Probability Distributions**")
                        cc1, cc2 = st.columns(2)
                        with cc1:
                            st.markdown("*Sentiment Distribution*")
                            display_colored_probability_chart(SENTIMENT_LABELS, s_probs, "sentiment", "Sentiment Probabilities")
                        with cc2:
                            st.markdown("*Emotion Distribution*")
                            display_colored_probability_chart(EMOTION_LABELS, e_probs, "emotion", "Emotion Probabilities")

    elif analyze_button:
        st.warning("⚠️ Please enter some text to analyze!")

# ---------- About Tab ----------
with tab2:
    st.markdown("## ℹ️ About This Application")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 Purpose
        This application provides advanced sentiment and emotion analysis specifically designed for crisis communication scenarios, with a focus on the Samsung Galaxy Note 7 crisis case study.
        
        ### 🤖 AI Models
        - **BERTweet**: Specialized for social media text
        - **DistilRoBERTa**: Fast and efficient general-purpose model
        - **Multi-task Learning**: Joint sentiment and emotion prediction
        
        ### 🔍 Explainable AI
        - Token-level attribution scores
        - Counterfactual analysis
        - Integrated Gradients support
        """)
    
    with col2:
        st.markdown("""
        ### 📊 Supported Analysis
        
        **Sentiment Categories:**
        - 😊 Positive
        - 😐 Neutral  
        - 😞 Negative
        
        **Emotion Categories:**
        - 😄 Joy
        - 😡 Anger
        - 😨 Fear
        - 😢 Sadness
        - 😲 Surprise
        - 😶 No Emotion
        """)
    
    st.markdown("---")
    st.markdown("### 🔧 Technical Details")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Supported Tokens", "512", "max length")
    with col2:
        st.metric("Processing Speed", "<10s", "per text")
