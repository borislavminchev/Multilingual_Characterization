"""
Streamlit Demo — SemEval-2025 Task 10 ST1: Entity Framing

Interactive UI for hierarchical entity role classification.
Loads trained coarse + fine classifiers and demonstrates the pipeline.

Usage:
    cd src2 && streamlit run demo.py
"""

import os
import re
import json
import time
import numpy as np
import torch
import streamlit as st
import matplotlib.pyplot as plt
from transformers import AutoModel, AutoTokenizer

from config import (
    MODEL_NAME, MAX_LENGTH,
    COARSE_CHECKPOINT_DIR, FINE_CHECKPOINT_DIR,
    FINE_THRESHOLD, FINE_GAP_RATIO, FINE_MIN_LABELS, FINE_MAX_LABELS,
    USE_SOFT_CONDITIONING, CARDINALITY_WEIGHT, TARGET_CARDINALITY,
    NUM_COARSE_LABELS, NUM_FINE_LABELS,
    COARSE_HEAD_TYPE, COARSE_LOSS_TYPE
)
from data_utils import ENTITY_START_TOKEN, ENTITY_END_TOKEN
from datasets import (
    coarse_label2id, coarse_id2label, fine_label2id, fine_id2label
)
from hierarchical_model import CoarseRoleClassifier, SoftConditionedFineClassifier, FineRoleClassifier
from inference import load_fine_classifier, soft_prediction, smart_prediction
from taxonomy_manager import TaxonomyManager
from saliency import (
    compute_occlusion_saliency,
    compute_span_occlusion_saliency,
    compute_gradient_x_embedding_saliency,
    aggregate_to_words,
    render_saliency_html,
    plot_saliency_bar,
    select_top_entries,
)

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

COARSE_LABELS = ["Protagonist", "Antagonist", "Innocent"]
COARSE_COLORS = {"Protagonist": "#2196F3", "Antagonist": "#F44336", "Innocent": "#4CAF50"}

FINE_LABELS_BY_COARSE = {
    "Protagonist": ["Guardian", "Martyr", "Peacemaker", "Rebel", "Underdog", "Virtuous"],
    "Antagonist": ["Instigator", "Conspirator", "Tyrant", "Foreign Adversary", "Traitor",
                    "Spy", "Saboteur", "Corrupt", "Incompetent", "Terrorist", "Deceiver", "Bigot"],
    "Innocent": ["Forgotten", "Exploited", "Victim", "Scapegoat"],
}

COARSE_FOR_FINE = {}
for _c, _fines in FINE_LABELS_BY_COARSE.items():
    for _f in _fines:
        COARSE_FOR_FINE[_f] = _c

EXAMPLES = [
    {
        "name": 'EN - Democrats (Antagonist: Conspirator + Tyrant)',
        "text": 'The very compound that the Democrats are targeting – CO2 – is actually the solution to preserving croplands, grasslands, forests and water supplies for growing populations.',
        "mention": 'Democrats',
    },
    {
        "name": 'EN - the West (Antagonist: Instigator + Foreign Adversary)',
        "text": 'Messed up your proxy war in Ukraine? Well, just blame China. So while the West has been pouring massive military hardware into the conflict to the extent of emptying their own weapons warehouses – and still not winning – it’s Beijing that’s supposed to be the “decisive enabler” of the war. Seriously.',
        "mention": 'the West',
    },
    {
        "name": 'BG - Володимир Зеленски (Protagonist: Guardian)',
        "text": 'Президентът на Украйна Володимир Зеленски призова за "пълна защита на украинското небе" след масираната руска въздушна атака през нощта, предаде Укринформ.',
        "mention": 'Володимир Зеленски',
    },
    {
        "name": 'BG - Путин (Antagonist: Tyrant)',
        "text": '"Обща цел за всички руснаци би трябвало да бъде освобождаването на Русия от безумния диктатор Путин и неговия режим, а не борбата със санкциите. Санкциите трябва само да бъдат засилвани докато Русия продължава въоръжената си агресия." – написа той в социалните мрежи.',
        "mention": 'Путин',
    },
    {
        "name": 'BG - Луганск (Innocent: Victim)',
        "text": 'Западът отгледа "терористична гадина", която унищожава всичко. Така официалният представител на МВнР на Русия Мария Захарова коментира пред журналисти "Новина" Николай Иванов днес удари Въоръжените сили на Украйна (ВСУ) по Луганск.',
        "mention": 'Луганск',
    },
    {
        "name": 'HI - प्रधानमंत्री / PM (Protagonist: Guardian + Peacemaker + Virtuous)',
        "text": 'प्रधानमंत्री नरेंद्र मोदी ने भी समित के दौरान राष्ट्रपति ज़ेलेंस्की के साथ एक बैठक की, जिसमें उन्होंने यूक्रेन युद्ध और रूसी आक्रामकता के बारे में बातचीत की।',
        "mention": 'प्रधानमंत्री',
    },
]


# ─────────────────────────────────────────────
# Model Loading
# ─────────────────────────────────────────────

def find_last_checkpoint(checkpoint_dir):
    """Find the checkpoint with the highest step number."""
    if not os.path.isdir(checkpoint_dir):
        return None
    checkpoints = []
    for name in os.listdir(checkpoint_dir):
        m = re.match(r'checkpoint-(\d+)$', name)
        if m and os.path.isdir(os.path.join(checkpoint_dir, name)):
            checkpoints.append((int(m.group(1)), name))
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda x: x[0])
    return os.path.join(checkpoint_dir, checkpoints[-1][1])


def load_model_weights(model, checkpoint_dir):
    """Load weights from pytorch_model.bin or model.safetensors."""
    ckpt = find_last_checkpoint(checkpoint_dir)
    if ckpt is None:
        return False

    bin_path = os.path.join(ckpt, 'pytorch_model.bin')
    st_path = os.path.join(ckpt, 'model.safetensors')

    if os.path.exists(bin_path):
        state_dict = torch.load(bin_path, map_location='cpu')
        # Use strict=False to ignore loss_fn weights that shouldn't be loaded
        model.load_state_dict(state_dict, strict=False)
        return True
    elif os.path.exists(st_path):
        from safetensors.torch import load_file
        state_dict = load_file(st_path)
        # Use strict=False to ignore loss_fn weights that shouldn't be loaded
        model.load_state_dict(state_dict, strict=False)
        return True
    return False


@st.cache_resource
def load_models():
    """Load tokenizer, coarse classifier, and fine classifier."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.add_special_tokens({
        'additional_special_tokens': [ENTITY_START_TOKEN, ENTITY_END_TOKEN]
    })

    # Coarse classifier — move base model to device BEFORE passing to constructor,
    # because TaxonomyManager runs inference during __init__
    coarse_base = AutoModel.from_pretrained(MODEL_NAME)
    coarse_base.resize_token_embeddings(len(tokenizer))
    coarse_base.to(device)
    coarse_model = CoarseRoleClassifier(
        base_model=coarse_base,
        tokenizer=tokenizer,
        device=device,
        head_type=COARSE_HEAD_TYPE,
        loss_type=COARSE_LOSS_TYPE,
    )
    coarse_loaded = load_model_weights(coarse_model, COARSE_CHECKPOINT_DIR)
    coarse_model.to(device)
    coarse_model.eval()

    # Fine classifier
    fine_checkpoint_dir = FINE_CHECKPOINT_DIR + ("_soft" if USE_SOFT_CONDITIONING else "_hard")

    fine_base = AutoModel.from_pretrained(MODEL_NAME)
    fine_base.resize_token_embeddings(len(tokenizer))
    fine_base.to(device)

    if USE_SOFT_CONDITIONING:
        fine_model = SoftConditionedFineClassifier(
            base_model=fine_base,
            tokenizer=tokenizer,
            device=device,
            threshold=FINE_THRESHOLD,
            num_coarse=NUM_COARSE_LABELS,
            num_fine=NUM_FINE_LABELS,
            cardinality_weight=CARDINALITY_WEIGHT,
            target_cardinality=TARGET_CARDINALITY,
        )
    else:
        fine_model = FineRoleClassifier(
            base_model=fine_base,
            tokenizer=tokenizer,
            device=device,
            threshold=FINE_THRESHOLD,
        )

    fine_loaded = load_model_weights(fine_model, fine_checkpoint_dir)
    fine_model.to(device)
    fine_model.eval()

    # Taxonomy manager (for hard masking fallback)
    taxonomy_mgr = TaxonomyManager(coarse_base, tokenizer, device)

    return tokenizer, coarse_model, fine_model, taxonomy_mgr, device, coarse_loaded, fine_loaded


# ─────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────

def predict(text, mention, start, end, tokenizer, coarse_model, fine_model, taxonomy_mgr, device):
    """Run hierarchical coarse → fine prediction."""
    # Mark entity in text
    marked_text = (
        f"{text[:start]} {ENTITY_START_TOKEN} {text[start:end]} {ENTITY_END_TOKEN} {text[end:]}"
    )

    # Tokenize
    encoding = tokenizer(
        marked_text, truncation=True, padding='max_length',
        max_length=MAX_LENGTH, return_tensors='pt',
        return_offsets_mapping=True,
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        # Coarse prediction
        coarse_out = coarse_model(input_ids, attention_mask)
        coarse_logits = coarse_out.logits[0]
        coarse_probs = torch.softmax(coarse_logits, dim=-1).cpu().numpy()
        coarse_id = int(coarse_probs.argmax())
        coarse_label = coarse_id2label[coarse_id]

        # Fine prediction
        coarse_probs_tensor = torch.tensor(
            coarse_probs, dtype=torch.float32
        ).unsqueeze(0).to(device)
        if USE_SOFT_CONDITIONING:
            fine_out = fine_model(
                input_ids, attention_mask,
                coarse_probs=coarse_probs_tensor,
                coarse_labels=torch.tensor([coarse_id]).to(device),
            )
            fine_logits = fine_out.logits[0]
            fine_probs = torch.sigmoid(fine_logits).cpu().numpy()
            predictions = soft_prediction(
                fine_probs,
                threshold=FINE_THRESHOLD, gap_ratio=FINE_GAP_RATIO,
                min_labels=FINE_MIN_LABELS, max_labels=FINE_MAX_LABELS,
            )
        else:
            fine_out = fine_model(
                input_ids, attention_mask,
                coarse_labels=torch.tensor([coarse_id]).to(device),
            )
            fine_logits = fine_out.logits[0]
            fine_probs = torch.sigmoid(fine_logits).cpu().numpy()
            predictions = smart_prediction(
                fine_probs, coarse_id, taxonomy_mgr,
                threshold=FINE_THRESHOLD, gap_ratio=FINE_GAP_RATIO,
                min_labels=FINE_MIN_LABELS, max_labels=FINE_MAX_LABELS,
            )

    selected_fine = [fine_id2label[i] for i in range(NUM_FINE_LABELS) if predictions[i] == 1]

    return {
        'coarse_label': coarse_label,
        'coarse_probs': {coarse_id2label[i]: float(coarse_probs[i]) for i in range(NUM_COARSE_LABELS)},
        'fine_labels': selected_fine,
        'fine_probs': {fine_id2label[i]: float(fine_probs[i]) for i in range(NUM_FINE_LABELS)},
        'predictions': predictions,
        # --- Saliency support: raw tensors + encoding for downstream analysis ---
        'input_ids': input_ids.detach(),
        'attention_mask': attention_mask.detach(),
        'encoding': encoding,
        'marked_text': marked_text,
        'coarse_probs_tensor': coarse_probs_tensor.detach(),
        'target_coarse_id': coarse_id,
        'target_fine_ids': [i for i in range(NUM_FINE_LABELS) if predictions[i] == 1],
    }


# ─────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────

def plot_coarse_probs(probs):
    """Horizontal bar chart for coarse probabilities."""
    fig, ax = plt.subplots(figsize=(6, 2))
    labels = list(probs.keys())
    values = list(probs.values())
    colors = [COARSE_COLORS[l] for l in labels]
    bars = ax.barh(labels, values, color=colors, edgecolor='white', height=0.6)
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f'{val:.1%}', va='center', fontsize=11, fontweight='bold')
    ax.set_xlim(0, 1.15)
    ax.set_xlabel('Probability')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    return fig


def plot_fine_probs(fine_probs, predictions, threshold):
    """Horizontal bar chart for fine-grained probabilities."""
    # Order by coarse group
    ordered_labels = []
    for c in COARSE_LABELS:
        ordered_labels.extend(FINE_LABELS_BY_COARSE[c])

    fig, ax = plt.subplots(figsize=(6, 6))
    y_pos = np.arange(len(ordered_labels))
    values = [fine_probs.get(l, 0) for l in ordered_labels]
    selected = [l in [fine_id2label[i] for i, p in enumerate(predictions) if p == 1] for l in ordered_labels]
    colors = []
    for i, l in enumerate(ordered_labels):
        c = COARSE_FOR_FINE[l]
        base_color = COARSE_COLORS[c]
        colors.append(base_color if selected[i] else '#E0E0E0')

    bars = ax.barh(y_pos, values, color=colors, edgecolor='white', height=0.7)

    for i, (bar, val, sel) in enumerate(zip(bars, values, selected)):
        if val > 0.02:
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                    f'{val:.2f}', va='center', fontsize=9,
                    fontweight='bold' if sel else 'normal',
                    color='black' if sel else 'gray')

    # Threshold line
    ax.axvline(x=threshold, color='red', linestyle='--', linewidth=1, alpha=0.6, label=f'Threshold={threshold}')

    # Group separators
    prot_end = len(FINE_LABELS_BY_COARSE["Protagonist"]) - 0.5
    ant_end = prot_end + len(FINE_LABELS_BY_COARSE["Antagonist"])
    for pos in [prot_end, ant_end]:
        ax.axhline(y=pos, color='black', linewidth=1)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(ordered_labels, fontsize=9)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel('Probability (sigmoid)')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    ax.legend(fontsize=9, loc='lower right')
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# Streamlit App
# ─────────────────────────────────────────────

def main():
    st.set_page_config(page_title="Entity Framing Demo", layout="wide")

    st.title("SemEval-2025 Task 10 ST1 -- Entity Framing")
    st.caption("Hierarchical multi-label classification of entity roles in news articles")

    # Sidebar: taxonomy reference
    with st.sidebar:
        st.header("Label Taxonomy")
        for coarse in COARSE_LABELS:
            color = COARSE_COLORS[coarse]
            st.markdown(f"**:{color[1:]}[{coarse}]**" if False else f"**{coarse}**")
            for fine in FINE_LABELS_BY_COARSE[coarse]:
                st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;- {fine}")
            st.markdown("---")

        st.header("Configuration")
        st.text(f"Model: {MODEL_NAME}")
        st.text(f"Mode: {'Soft Conditioning' if USE_SOFT_CONDITIONING else 'Hard Masking'}")
        st.text(f"Threshold: {FINE_THRESHOLD}")
        st.text(f"Gap ratio: {FINE_GAP_RATIO}")
        st.text(f"Min/Max labels: {FINE_MIN_LABELS}/{FINE_MAX_LABELS}")

    # Load models
    with st.spinner("Loading models (this may take a minute on first run)..."):
        tokenizer, coarse_model, fine_model, taxonomy_mgr, device, coarse_ok, fine_ok = load_models()

    if not coarse_ok or not fine_ok:
        st.warning(
            "Some model checkpoints not found. Predictions will use random weights.\n\n"
            f"- Coarse: {'Loaded' if coarse_ok else 'NOT FOUND at ' + COARSE_CHECKPOINT_DIR}\n"
            f"- Fine: {'Loaded' if fine_ok else 'NOT FOUND'}\n\n"
            "Please ensure checkpoint directories contain trained model files."
        )

    st.markdown(f"**Device:** `{device}` &nbsp;|&nbsp; "
                f"**Coarse model:** {'Ready' if coarse_ok else 'Missing'} &nbsp;|&nbsp; "
                f"**Fine model:** {'Ready' if fine_ok else 'Missing'}")

    st.markdown("---")

    # Example selector
    example_names = ["Custom input"] + [e["name"] for e in EXAMPLES]
    selected_example = st.selectbox("Load example:", example_names)

    if selected_example != "Custom input":
        ex = EXAMPLES[example_names.index(selected_example) - 1]
        default_text = ex["text"]
        default_mention = ex["mention"]
    else:
        default_text = ""
        default_mention = ""

    # Input
    col_input, col_entity = st.columns([3, 1])
    with col_input:
        text = st.text_area("Article text:", value=default_text, height=180)
    with col_entity:
        mention = st.text_input("Entity mention:", value=default_mention)
        auto_detect = st.checkbox("Auto-detect position", value=True)

        if auto_detect and mention and mention in text:
            start = text.index(mention)
            end = start + len(mention)
            st.text(f"Position: [{start}, {end})")
        else:
            c1, c2 = st.columns(2)
            with c1:
                start = st.number_input("Start:", min_value=0, value=0)
            with c2:
                end = st.number_input("End:", min_value=0, value=0)

    # Show highlighted text
    if text and mention and mention in text:
        s = text.index(mention)
        highlighted = (
            text[:s]
            + f"**:red[\\[{mention}\\]]**"
            + text[s + len(mention):]
        )
        st.markdown(highlighted)

    # Classify button
    if st.button("Classify", type="primary", disabled=not (text and mention and end > start)):
        with st.spinner("Running inference..."):
            result = predict(text, mention, start, end,
                             tokenizer, coarse_model, fine_model, taxonomy_mgr, device)
        # Persist across reruns triggered by widgets in the saliency section.
        st.session_state["last_result"] = result
        st.session_state["last_text"] = text
        # New classification → drop any cached saliency from a previous example
        # (otherwise stale word offsets get rendered over the new text).
        st.session_state["saliency_cache"] = {}

    # Everything below (results + saliency) is driven from session_state so
    # widgets in the saliency section don't wipe the results block on rerun.
    if "last_result" in st.session_state:
        result = st.session_state["last_result"]

        # Results
        st.markdown("---")
        st.subheader("Results")

        # Coarse
        col_coarse, col_fine = st.columns([1, 2])

        with col_coarse:
            st.markdown("### Coarse Role")
            coarse_label = result['coarse_label']
            coarse_conf = result['coarse_probs'][coarse_label]
            st.markdown(
                f"<h2 style='color:{COARSE_COLORS[coarse_label]}'>{coarse_label} ({coarse_conf:.1%})</h2>",
                unsafe_allow_html=True,
            )
            fig_coarse = plot_coarse_probs(result['coarse_probs'])
            st.pyplot(fig_coarse)
            plt.close(fig_coarse)

        with col_fine:
            st.markdown("### Fine-Grained Roles")
            # Selected labels as chips
            for fl in result['fine_labels']:
                prob = result['fine_probs'][fl]
                color = COARSE_COLORS[COARSE_FOR_FINE[fl]]
                st.markdown(
                    f"<span style='background-color:{color};color:white;padding:4px 12px;"
                    f"border-radius:16px;margin:2px;display:inline-block;font-weight:bold'>"
                    f"{fl} ({prob:.2f})</span>",
                    unsafe_allow_html=True,
                )

            fig_fine = plot_fine_probs(result['fine_probs'], result['predictions'], FINE_THRESHOLD)
            st.pyplot(fig_fine)
            plt.close(fig_fine)

        # Raw probabilities expander
        with st.expander("Show raw probabilities"):
            st.json({
                'coarse_probabilities': result['coarse_probs'],
                'fine_probabilities': result['fine_probs'],
                'selected_fine_labels': result['fine_labels'],
            })

        # ── Saliency Analysis ────────────────────────────────────────────
        _render_saliency_section(
            result=result,
            coarse_model=coarse_model,
            fine_model=fine_model,
            tokenizer=tokenizer,
            device=device,
        )


def _render_saliency_section(result, coarse_model, fine_model, tokenizer, device):
    """Renders the Saliency Analysis expander (occlusion + gradient×embedding)."""
    with st.expander("🔬 Saliency Analysis", expanded=False):
        st.caption(
            "For each word in the context we measure how much the predicted "
            "class probability drops if that word were removed. "
            "Green = supports the predicted class, red = opposes it."
        )

        # Method selector: occlusion (word) / phrase occlusion / gradient×embedding
        method = st.radio(
            "Method:",
            options=["Occlusion (word)", "Occlusion (phrase)", "Gradient × Embedding"],
            horizontal=True,
            help=(
                "Occlusion (word, recommended): mask each word individually. "
                "Occlusion (phrase): mask whole phrases at once — a stronger, "
                "less noisy signal (phrase size is chosen automatically). "
                "Gradient×Embedding: local linear approximation; faster, noisier."
            ),
            key="saliency_method",
        )
        # Phrase occlusion always uses non-overlapping blocks (one phrase per word).
        phrase_mode = 'block' if method == "Occlusion (phrase)" else None
        if method == "Gradient × Embedding":
            st.caption(
                "ℹ️ Gradient×Embedding is an approximate method and may give "
                "noisier results than Occlusion. When in doubt, compare with Occlusion."
            )
        # Phrase mode shows fewer, coarser bars, so default to Top-3 there.
        is_phrase = method == "Occlusion (phrase)"
        top_k = st.slider(
            "Top-K phrases:" if is_phrase else "Top-K words:",
            min_value=1,
            max_value=25,
            value=3 if is_phrase else 10,
            key="saliency_topk_phrase" if is_phrase else "saliency_topk",
        )

        # Short hash of the marked text so cache keys never collide across
        # different examples that happen to share a coarse/fine class id.
        text_key = str(abs(hash(result['marked_text'])) % (10 ** 8))
        method_key = method + (f"-{phrase_mode}" if phrase_mode else "")

        tab_coarse, tab_fine = st.tabs(
            [f"Coarse: {result['coarse_label']}",
             f"Fine ({len(result['fine_labels'])})"]
        )

        with tab_coarse:
            _render_saliency_for_target(
                result=result,
                model=coarse_model,
                tokenizer=tokenizer,
                device=device,
                task='coarse',
                target_class=result['target_coarse_id'],
                target_label=result['coarse_label'],
                method=method,
                phrase_mode=phrase_mode,
                top_k=top_k,
                cache_key=f"{text_key}-coarse-{result['target_coarse_id']}-{method_key}",
            )


        with tab_fine:
            fine_ids = result['target_fine_ids']
            if not fine_ids:
                st.info("No fine-grained roles predicted.")
            else:
                fine_names = [fine_id2label[i] for i in fine_ids]
                selected = st.selectbox(
                    "Fine-grained role:",
                    options=fine_names,
                    key="saliency_fine_selector",
                )
                selected_id = fine_label2id[selected]
                _render_saliency_for_target(
                    result=result,
                    model=fine_model,
                    tokenizer=tokenizer,
                    device=device,
                    task='fine',
                    target_class=selected_id,
                    target_label=selected,
                    method=method,
                    phrase_mode=phrase_mode,
                    top_k=top_k,
                    cache_key=f"{text_key}-fine-{selected_id}-{method_key}",
                )


def _render_saliency_for_target(result, model, tokenizer, device, task,
                                target_class, target_label, method, top_k, cache_key,
                                phrase_mode=None):
    """Compute and render saliency for one (task, target_class) combination."""
    # Cache the raw token-level saliency in session_state so switching top_k or
    # bouncing between tabs doesn't recompute unnecessarily.
    cache = st.session_state.setdefault("saliency_cache", {})
    if cache_key not in cache:
        method_used = method
        agg_strategy = 'sum'
        with st.spinner(f"Computing saliency ({method_used})…"):
            t0 = time.perf_counter()
            token_sal = None
            span_info = None

            if method == "Gradient × Embedding":
                token_sal = compute_gradient_x_embedding_saliency(
                    input_ids=result['input_ids'],
                    attention_mask=result['attention_mask'],
                    model=model,
                    target_class=target_class,
                    tokenizer=tokenizer,
                    task=task,
                    coarse_probs=result['coarse_probs_tensor'] if task == 'fine' else None,
                    device=device,
                )
                if token_sal is None:
                    st.warning(
                        "Gradient method does not work with the current model — falling back to occlusion."
                    )
                    method_used = "Occlusion (fallback)"

            elif method == "Occlusion (phrase)":
                token_sal, _, span_info = compute_span_occlusion_saliency(
                    input_ids=result['input_ids'],
                    attention_mask=result['attention_mask'],
                    model=model,
                    target_class=target_class,
                    tokenizer=tokenizer,
                    encoding=result['encoding'],
                    marked_text=result['marked_text'],
                    task=task,
                    coarse_probs=result['coarse_probs_tensor'] if task == 'fine' else None,
                    device=device,
                    mode=phrase_mode or 'block',
                    max_phrases=10,
                )
                # Every subword token in a word already carries the same phrase
                # delta, so aggregate by 'max' to avoid multiplying by the
                # number of subwords.
                agg_strategy = 'max'

            if token_sal is None:
                token_sal, _ = compute_occlusion_saliency(
                    input_ids=result['input_ids'],
                    attention_mask=result['attention_mask'],
                    model=model,
                    target_class=target_class,
                    tokenizer=tokenizer,
                    task=task,
                    coarse_probs=result['coarse_probs_tensor'] if task == 'fine' else None,
                    device=device,
                )

            words = aggregate_to_words(
                token_saliency=token_sal,
                encoding=result['encoding'],
                marked_text=result['marked_text'],
                strategy=agg_strategy,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000
            cache[cache_key] = {
                'words': words,
                'elapsed_ms': elapsed_ms,
                'method_used': method_used,
                'nonzero': int(np.count_nonzero(token_sal)),
                'span_info': span_info,
            }

    entry = cache[cache_key]
    words = entry['words']
    span_note = ""
    if entry.get('span_info'):
        si = entry['span_info']
        span_note = (
            f" | phrase = {si['span_size']} "
            f"{'word' if si['span_size'] == 1 else 'words'}, "
            f"{si['num_phrases']} phrases"
        )
    st.caption(
        f"Computed in {entry['elapsed_ms']:.0f} ms over "
        f"{len(words)} words ({entry['nonzero']} significant tokens, "
        f"method: {entry['method_used']}){span_note}."
    )

    if not words:
        st.info("No context found to analyse.")
        return

    col_html, col_bar = st.columns([2, 1])

    # Determine whether to group into phrases, and compute the exact set of
    # entries shown on the bar chart so we can highlight only those in the text.
    group_phrases = (
        entry.get('span_info') is not None
        and phrase_mode == 'block'
        and entry['span_info'].get('span_size', 1) > 1
    )
    top_entries = select_top_entries(
        words, top_k=top_k, group_phrases=group_phrases,
        marked_text=result['marked_text'],
    )
    highlight_spans = [(e['start'], e['end']) for e in top_entries]

    with col_html:
        st.markdown("#### Highlighted context")
        html_str = render_saliency_html(
            marked_text=result['marked_text'],
            words=words,
            target_label=target_label,
            method_name=entry['method_used'].lower(),
            highlight_spans=highlight_spans,
        )
        st.markdown(html_str, unsafe_allow_html=True)

    with col_bar:
        chart_title = "Top influencing phrases" if group_phrases else f"Top-{top_k} influencing words"
        st.markdown(f"#### {chart_title}")
        fig_sal = plot_saliency_bar(words, target_label, top_k=top_k,
                                    group_phrases=group_phrases,
                                    marked_text=result['marked_text'])
        st.pyplot(fig_sal)
        plt.close(fig_sal)


if __name__ == "__main__":
    main()
