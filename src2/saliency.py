"""
Saliency Analysis for Entity Role Classification

Provides two attribution methods that work on already-trained E19 models
without any retraining or architecture modifications:

  1. Occlusion (primary): mask each context token with <mask>, measure the
     drop in the target class probability. Model-agnostic; batched.

  2. Gradient x Embedding (optional): grad of target logit w.r.t. input
     embeddings, dotted with the embeddings themselves. Uses a forward hook
     on base_model.embeddings.word_embeddings — no forward() modification.

Public API:
  compute_occlusion_saliency(...) -> (np.ndarray[T], np.ndarray[T])
  compute_gradient_x_embedding_saliency(...) -> np.ndarray[T] | None
  aggregate_to_words(...)        -> list[dict]
  render_saliency_html(...)      -> str
  plot_saliency_bar(...)         -> matplotlib.figure.Figure

Skip positions (never masked / not attributed):
  CLS, SEP, PAD, [ENTITY], [/ENTITY], and any token between the two markers.
  The entity span itself is trivial to mask (destroys the input) so we
  exclude it. entity_span_pooling in classifiers.py locates the markers
  in input_ids, so preserving them is required.
"""

import html

import numpy as np
import torch
import matplotlib.pyplot as plt

from data_utils import ENTITY_START_TOKEN, ENTITY_END_TOKEN


# Diverging colors for signed saliency (design system: green supports, red opposes).
POS_COLOR = "#2E7D32"   # green
NEG_COLOR = "#C62828"   # red
NEUTRAL_BG = "#EEEEEE"  # entity chip background

# Words whose |s_norm| falls below this threshold are rendered without a
# colored background (they're visual noise).
DISPLAY_ALPHA_CUTOFF = 0.05
MAX_ALPHA = 0.75


# =============================================================================
# INTERNAL HELPERS
# =============================================================================

def _resolve_special_ids(tokenizer):
    """Collect token IDs that must never be occluded and locate the entity markers."""
    special = set()
    for tok_id in (tokenizer.cls_token_id, tokenizer.sep_token_id,
                   tokenizer.pad_token_id, tokenizer.bos_token_id,
                   tokenizer.eos_token_id):
        if tok_id is not None:
            special.add(int(tok_id))

    ent_start_id = tokenizer.convert_tokens_to_ids(ENTITY_START_TOKEN)
    ent_end_id = tokenizer.convert_tokens_to_ids(ENTITY_END_TOKEN)
    return special, int(ent_start_id), int(ent_end_id)


def _find_entity_span(input_ids_1d, ent_start_id, ent_end_id):
    """Return (start_marker_pos, end_marker_pos) or (None, None) if not found.

    Mirrors classifiers.py::entity_span_pooling — first occurrence only.
    """
    starts = (input_ids_1d == ent_start_id).nonzero(as_tuple=True)[0]
    ends = (input_ids_1d == ent_end_id).nonzero(as_tuple=True)[0]
    if len(starts) == 0 or len(ends) == 0:
        return None, None
    return int(starts[0].item()), int(ends[0].item())


def _valid_positions(input_ids_1d, attention_mask_1d, tokenizer):
    """Positions that are safe to occlude for saliency analysis."""
    special, ent_start_id, ent_end_id = _resolve_special_ids(tokenizer)
    start_pos, end_pos = _find_entity_span(input_ids_1d, ent_start_id, ent_end_id)

    positions = []
    seq_len = input_ids_1d.size(0)
    for i in range(seq_len):
        if attention_mask_1d[i].item() != 1:
            continue
        tok_id = int(input_ids_1d[i].item())
        if tok_id in special or tok_id == ent_start_id or tok_id == ent_end_id:
            continue
        # Skip the entity span itself (tokens strictly between the markers).
        if start_pos is not None and end_pos is not None:
            if start_pos <= i <= end_pos:
                continue
        positions.append(i)
    return positions


def _forward_prob(model, input_ids, attention_mask, target_class, task,
                  coarse_probs=None):
    """Single-example forward → probability of the target class.

    task='coarse'  → softmax over 3 classes
    task='fine'    → sigmoid over 22 classes (multi-label), select target_class
    """
    kwargs = {}
    if task == 'fine' and coarse_probs is not None:
        kwargs['coarse_probs'] = coarse_probs
    out = model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
    logits = out.logits
    if task == 'coarse':
        probs = torch.softmax(logits, dim=-1)
    else:
        probs = torch.sigmoid(logits)
    return probs[:, target_class]  # shape (B,)


# =============================================================================
# OCCLUSION SALIENCY
# =============================================================================

def compute_occlusion_saliency(
    input_ids,
    attention_mask,
    model,
    target_class,
    tokenizer,
    task='coarse',
    coarse_probs=None,
    device='cuda',
    batch_size=32,
):
    """Batched occlusion saliency.

    For each maskable position i, replace input_ids[0, i] with <mask> and
    measure delta = p_original(target) - p_masked(target). Positive delta
    means the token supports the target class.

    Args:
        input_ids: (1, T) LongTensor
        attention_mask: (1, T) LongTensor
        model: CoarseRoleClassifier or SoftConditionedFineClassifier (eval mode)
        target_class: int index of the target class
        tokenizer: HF fast tokenizer (must be .is_fast)
        task: 'coarse' or 'fine'
        coarse_probs: (1, 3) FloatTensor — required for task='fine' with
                      SoftConditionedFineClassifier
        device: device to run on
        batch_size: micro-batch size for perturbed forward passes

    Returns:
        saliency: np.ndarray shape (T,) — deltas; 0.0 at skipped positions
        valid_mask: np.ndarray shape (T,) bool — True at probed positions
    """
    if not getattr(tokenizer, 'is_fast', False):
        raise RuntimeError("Fast tokenizer required (needs word_ids/offset_mapping).")

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    if coarse_probs is not None:
        coarse_probs = coarse_probs.to(device)

    T = input_ids.size(1)
    saliency = np.zeros(T, dtype=np.float32)
    valid_mask = np.zeros(T, dtype=bool)

    model.eval()
    with torch.no_grad():
        # Positions we will occlude.
        positions = _valid_positions(input_ids[0], attention_mask[0], tokenizer)
        if len(positions) == 0:
            return saliency, valid_mask

        # Original probability.
        p_orig = _forward_prob(
            model, input_ids, attention_mask, target_class,
            task=task, coarse_probs=coarse_probs
        ).item()

        # Build batched perturbations. Each row is a copy of input_ids with
        # exactly one token replaced by <mask> at positions[row_idx].
        mask_id = tokenizer.mask_token_id
        n = len(positions)
        deltas = np.zeros(n, dtype=np.float32)

        pos_tensor = torch.tensor(positions, dtype=torch.long, device=device)
        # Repeat coarse_probs across the perturbation batch if needed.
        cp_repeated = (
            coarse_probs.expand(min(batch_size, n), -1) if coarse_probs is not None else None
        )

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            chunk = end - start
            # (chunk, T)
            perturbed = input_ids.expand(chunk, -1).clone()
            am_rep = attention_mask.expand(chunk, -1).contiguous()
            row_ids = torch.arange(chunk, device=device)
            col_ids = pos_tensor[start:end]
            perturbed[row_ids, col_ids] = mask_id

            cp_chunk = None
            if coarse_probs is not None:
                cp_chunk = coarse_probs.expand(chunk, -1).contiguous()

            probs = _forward_prob(
                model, perturbed, am_rep, target_class,
                task=task, coarse_probs=cp_chunk,
            )
            deltas[start:end] = (p_orig - probs.detach().cpu().numpy()).astype(np.float32)

        for local_idx, pos in enumerate(positions):
            saliency[pos] = deltas[local_idx]
            valid_mask[pos] = True

    return saliency, valid_mask


# =============================================================================
# GRADIENT x EMBEDDING SALIENCY
# =============================================================================

def compute_gradient_x_embedding_saliency(
    input_ids,
    attention_mask,
    model,
    target_class,
    tokenizer,
    task='coarse',
    coarse_probs=None,
    device='cuda',
):
    """Gradient x Embedding saliency via forward hook.

    Hooks `base_model.embeddings.word_embeddings` (nn.Embedding) to grab the
    embedding output tensor. We then run a normal forward, backprop the target
    logit, and take element-wise product with the embeddings summed over the
    hidden dimension.

    Returns None if the hook path fails (e.g. embeddings are not a plain
    tensor) — callers should fall back to occlusion.
    """
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    if coarse_probs is not None:
        coarse_probs = coarse_probs.to(device)

    # Find the word_embeddings module.
    try:
        embed_module = model.base_model.get_input_embeddings()
    except AttributeError:
        return None

    cache = {}

    def _hook(module, inputs, output):
        # output shape: (B, T, H). retain_grad so we can access .grad later.
        if not isinstance(output, torch.Tensor):
            return
        output.retain_grad()
        cache['embed'] = output

    handle = embed_module.register_forward_hook(_hook)

    was_training = model.training
    model.eval()
    # Ensure we can compute gradients on the embedding output even though the
    # embedding module itself may be frozen (requires_grad=False on weights is
    # fine; retain_grad on activations is what matters).
    try:
        with torch.enable_grad():
            kwargs = {}
            if task == 'fine' and coarse_probs is not None:
                kwargs['coarse_probs'] = coarse_probs
            model.zero_grad(set_to_none=True)
            out = model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
            logits = out.logits
            # Target the PROBABILITY (not the raw logit) so the attribution is
            # directly comparable to occlusion (which measures Δp). For coarse
            # use softmax; for fine use sigmoid. The model here is not confident
            # enough to saturate the nonlinearity, so gradients don't vanish.
            if task == 'coarse':
                target_scalar = torch.softmax(logits, dim=-1)[0, target_class]
            else:
                target_scalar = torch.sigmoid(logits)[0, target_class]
            target_scalar.backward()

            emb = cache.get('embed', None)
            if emb is None or emb.grad is None:
                return None

            # (1, T, H) → (T,). Positive = the token's embedding pushes the
            # target probability up (supports the class), matching occlusion's
            # sign convention (positive delta = supports).
            saliency = (emb.grad * emb).sum(dim=-1).detach().cpu().numpy()[0]
    except Exception:
        return None
    finally:
        handle.remove()
        if was_training:
            model.train()

    # Zero out positions we don't want to display (specials + entity span).
    invalid = np.ones_like(saliency, dtype=bool)
    for pos in _valid_positions(input_ids[0], attention_mask[0], tokenizer):
        invalid[pos] = False
    saliency[invalid] = 0.0

    return saliency.astype(np.float32)


# =============================================================================
# SUBWORD -> WORD AGGREGATION
# =============================================================================

def aggregate_to_words(token_saliency, encoding, marked_text, strategy='sum'):
    """Group subword token saliency into whole-word saliency using word_ids().

    Args:
        token_saliency: np.ndarray shape (T,)
        encoding: HF BatchEncoding with is_fast=True
        marked_text: the string that was tokenized (with [ENTITY] / [/ENTITY])
        strategy: 'sum' | 'max' | 'mean'

    Returns:
        List of dicts sorted by start:
            {'word': str, 'start': int, 'end': int, 'saliency': float,
             'is_entity_marker': bool}
    """
    word_ids = encoding.word_ids(batch_index=0)
    offsets = encoding['offset_mapping']
    # offsets may be a torch.Tensor (1, T, 2), a list of lists of tuples, or a
    # plain list of (start, end) pairs. Normalize to a plain list-of-pairs.
    if hasattr(offsets, 'tolist'):
        offsets = offsets.tolist()
    if len(offsets) > 0 and len(offsets) > 0 and isinstance(offsets[0], (list, tuple)) \
            and len(offsets[0]) > 0 and isinstance(offsets[0][0], (list, tuple)):
        # Nested one level (batch dim) — take first row.
        offsets = offsets[0]

    words = {}
    for tok_idx, wid in enumerate(word_ids):
        if wid is None:
            continue
        span = offsets[tok_idx]
        s, e = int(span[0]), int(span[1])
        if s == 0 and e == 0:
            continue  # special token
        if wid not in words:
            words[wid] = {
                'start': s,
                'end': e,
                'saliency_values': [float(token_saliency[tok_idx])],
            }
        else:
            words[wid]['start'] = min(words[wid]['start'], s)
            words[wid]['end'] = max(words[wid]['end'], e)
            words[wid]['saliency_values'].append(float(token_saliency[tok_idx]))

    out = []
    for wid, w in words.items():
        s, e = w['start'], w['end']
        text = marked_text[s:e]
        # Skip if the "word" is exactly an entity marker.
        if text.strip() in (ENTITY_START_TOKEN, ENTITY_END_TOKEN):
            continue
        vals = w['saliency_values']
        if strategy == 'max':
            sal = max(vals, key=abs)
        elif strategy == 'mean':
            sal = float(np.mean(vals))
        else:  # sum
            sal = float(np.sum(vals))
        out.append({
            'word': text,
            'start': s,
            'end': e,
            'saliency': sal,
        })
    out.sort(key=lambda x: x['start'])
    return out


# =============================================================================
# HTML RENDERING
# =============================================================================

def _rgba(hex_color, alpha):
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    return f"rgba({r},{g},{b},{alpha:.3f})"


def render_saliency_html(marked_text, words, target_label, method_name="occlusion"):
    """Render marked_text with per-word background colors reflecting saliency.

    Positive saliency (supports the target class) -> green tint.
    Negative saliency (opposes the target class)  -> red tint.
    Entity markers are stripped from display; the entity span is shown as a
    neutral rounded chip with bold underline.

    Args:
        marked_text: the string with [ENTITY] / [/ENTITY] markers
        words: output of aggregate_to_words
        target_label: display name of the target class (e.g. "Antagonist")
        method_name: 'occlusion' or 'gradient×embedding' (for the legend)

    Returns:
        HTML string suitable for st.markdown(..., unsafe_allow_html=True)
    """
    if not words:
        return "<em>No context to analyse.</em>"

    max_abs = max(abs(w['saliency']) for w in words)
    if max_abs < 1e-9:
        max_abs = 1e-9

    # Find entity marker character positions in marked_text.
    ent_open = marked_text.find(ENTITY_START_TOKEN)
    ent_close = marked_text.find(ENTITY_END_TOKEN)

    # Emit text between and around word spans, replacing markers with a chip.
    pieces = []
    cursor = 0
    words_sorted = sorted(words, key=lambda w: w['start'])

    def emit_plain(segment):
        """Emit a plain text run, replacing entity markers with a chip."""
        if not segment:
            return
        # If markers fall inside this segment, split around them.
        # (Simple case: at most one open + one close in the whole text.)
        pieces.append(html.escape(segment))

    for w in words_sorted:
        # Emit text before this word.
        if w['start'] > cursor:
            emit_plain(marked_text[cursor:w['start']])
        # Emit the word itself, colored.
        s_norm = w['saliency'] / max_abs
        alpha = min(MAX_ALPHA, abs(s_norm))
        word_html = html.escape(w['word'])
        if abs(s_norm) < DISPLAY_ALPHA_CUTOFF:
            pieces.append(word_html)
        else:
            color = POS_COLOR if s_norm > 0 else NEG_COLOR
            bg = _rgba(color, alpha)
            tooltip = f"Δp = {w['saliency']:+.3f}"
            pieces.append(
                f'<span style="background:{bg};border-radius:3px;'
                f'padding:1px 3px;" title="{tooltip}">{word_html}</span>'
            )
        cursor = w['end']

    # Emit trailing text.
    if cursor < len(marked_text):
        emit_plain(marked_text[cursor:])

    joined = ''.join(pieces)

    # Replace entity markers with a chip in the joined string. We escaped the
    # text runs but not our own tags, so the markers still appear literally.
    ent_open_esc = html.escape(ENTITY_START_TOKEN)
    ent_close_esc = html.escape(ENTITY_END_TOKEN)
    chip_open = (
        f'<span style="background:{NEUTRAL_BG};color:#111;border-radius:4px;'
        f'padding:1px 4px;font-weight:600;text-decoration:underline;">'
    )
    chip_close = '</span>'
    # Whitespace around the entity chip: the marker is typically surrounded by
    # spaces in marked_text. We turn "[ENTITY] X [/ENTITY]" into a single chip.
    joined = joined.replace(f'{ent_open_esc} ', chip_open)
    joined = joined.replace(f' {ent_close_esc}', chip_close)
    # Fallback if the space pattern didn't match exactly:
    joined = joined.replace(ent_open_esc, chip_open)
    joined = joined.replace(ent_close_esc, chip_close)

    # Legend + body wrapper.
    legend = (
        f'<div style="font-size:0.85em;color:#555;margin-bottom:8px;">'
        f'<span style="background:{_rgba(POS_COLOR, 0.55)};padding:2px 6px;'
        f'border-radius:3px;">supports {html.escape(target_label)}</span> '
        f'&nbsp;&nbsp;'
        f'<span style="background:{_rgba(NEG_COLOR, 0.55)};padding:2px 6px;'
        f'border-radius:3px;">opposes {html.escape(target_label)}</span> '
        f'&nbsp;&nbsp;<em>method: {html.escape(method_name)}</em>'
        f'</div>'
    )
    body = (
        f'<div style="white-space:pre-wrap;line-height:2.0;font-size:1.02rem;'
        f'font-family:inherit;padding:8px;background:#FAFAFA;'
        f'border-radius:4px;border:1px solid #EEE;">{joined}</div>'
    )
    return legend + body


# =============================================================================
# BAR CHART
# =============================================================================

def _fmt_delta(v, scale_exp):
    """Format a delta value adaptively based on the magnitude scale of the data.

    scale_exp is floor(log10(max_abs)). For tiny values (e.g. 1e-4) we show
    enough significant digits instead of rounding everything to 0.000.
    """
    if v == 0:
        return "0"
    # Number of decimals: keep ~2 significant digits relative to the scale.
    decimals = max(1, min(6, 1 - scale_exp))
    return f"{v:+.{decimals}f}"


def plot_saliency_bar(words, target_label, top_k=10):
    """Horizontal bar chart of the top-K words by |saliency|.

    Sorted by signed saliency descending so positive contributors appear at the
    top and opposing words at the bottom. Bar color reflects sign.
    """
    if not words:
        fig, ax = plt.subplots(figsize=(4.5, 2.5))
        ax.text(0.5, 0.5, "No context", ha='center', va='center')
        ax.axis('off')
        return fig

    ranked = sorted(words, key=lambda w: abs(w['saliency']), reverse=True)[:top_k]
    ranked.sort(key=lambda w: w['saliency'], reverse=True)

    labels = [w['word'] for w in ranked]
    values = [w['saliency'] for w in ranked]
    colors = [POS_COLOR if v > 0 else NEG_COLOR for v in values]

    max_abs = max((abs(v) for v in values), default=0.0)
    # Magnitude scale used both for label formatting and axis tick density.
    if max_abs > 0:
        scale_exp = int(np.floor(np.log10(max_abs)))
    else:
        scale_exp = 0

    k = len(ranked)
    fig, ax = plt.subplots(figsize=(5.8, max(2.5, 0.45 * k)))
    y_pos = np.arange(k)
    bars = ax.barh(y_pos, values, color=colors, edgecolor='white',
                   height=0.7, linewidth=1.5)

    label_offset = max_abs * 0.03 if max_abs > 0 else 0.01
    for bar, v in zip(bars, values):
        x = bar.get_width()
        ha = 'left' if x >= 0 else 'right'
        ax.text(
            x + (label_offset if x >= 0 else -label_offset),
            bar.get_y() + bar.get_height() / 2,
            _fmt_delta(v, scale_exp),
            va='center', ha=ha, fontsize=9,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.axvline(0, color='#888', linewidth=0.8)
    ax.set_xlabel(f'Δ p({target_label}) when word is masked', fontsize=9)
    ax.grid(axis='x', alpha=0.25, linewidth=0.5)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    ax.spines['left'].set_color('#CCC')
    ax.spines['bottom'].set_color('#CCC')

    # Limit x-axis ticks to at most 5 so labels don't overlap, and format them
    # in scientific notation when the values are tiny.
    from matplotlib.ticker import MaxNLocator, FuncFormatter
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, prune='both'))
    if 0 < max_abs < 1e-2:
        ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
        ax.xaxis.get_offset_text().set_fontsize(8)
    ax.tick_params(axis='x', labelsize=8)

    # Symmetric-ish x padding so annotations don't clip.
    xmin = min(values + [0.0])
    xmax = max(values + [0.0])
    span = (xmax - xmin) if xmax > xmin else max(max_abs, 1e-6)
    pad = span * 0.28
    ax.set_xlim(xmin - pad, xmax + pad)

    fig.tight_layout()
    return fig
