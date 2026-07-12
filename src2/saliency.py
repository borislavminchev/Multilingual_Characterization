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

from data_utils import ENTITY_START_TOKEN, ENTITY_END_TOKEN, split_sentences_with_offsets


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
                  coarse_probs=None, score_mode='prob'):
    """Single-example forward → score of the target class.

    task='coarse'  → softmax over 3 classes
    task='fine'    → sigmoid over 22 classes (multi-label), select target_class

    score_mode:
        'prob'   → the raw probability of the target class (default; used for
                   coarse, whose 3-way softmax is not saturated ~0.4).
        'logit'  → the raw target logit (used for fine). Chosen over the fine
                   probability because:
                     * The sigmoid probability saturates near 1.0 for confident
                       fine predictions, so masking barely moves it (the "all
                       red / noisy sign" problem). The logit is unbounded and
                       stays sensitive.
                     * With soft conditioning, fine_logit = entity_projection(e)
                       + hierarchy_prior, and the prior is constant while we hold
                       coarse_probs fixed. In the occlusion DELTA the constant
                       prior cancels exactly, so the score reflects only the
                       entity-driven, target-specific contribution.
                   A plain single logit is used rather than a margin over the
                   other classes: the max-over-others reference shifts between
                   the clean and masked runs, which biases the sign (masking a
                   word that supports a RIVAL role would otherwise show up as
                   "against" the target).
    """
    kwargs = {}
    if task == 'fine' and coarse_probs is not None:
        kwargs['coarse_probs'] = coarse_probs
    out = model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
    logits = out.logits

    if score_mode == 'logit':
        return logits[:, target_class]  # (B,) — unbounded, prior cancels in delta

    # score_mode == 'prob'
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

        # For confident fine (multi-label) predictions the raw probability is
        # saturated near 1.0, so masking barely moves it and the sign becomes
        # noise. Score the target LOGIT instead — unbounded (no saturation) and,
        # because the hierarchy prior is constant, the prior cancels in the
        # occlusion delta, leaving only the entity-driven contribution.
        score_mode = 'logit' if task == 'fine' else 'prob'

        # Original score.
        p_orig = _forward_prob(
            model, input_ids, attention_mask, target_class,
            task=task, coarse_probs=coarse_probs, score_mode=score_mode
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
                task=task, coarse_probs=cp_chunk, score_mode=score_mode,
            )
            deltas[start:end] = (p_orig - probs.detach().cpu().numpy()).astype(np.float32)

        # Baseline correction — COARSE ONLY. Inserting <mask> tokens shifts the
        # softmax probability by a roughly constant amount regardless of which
        # word is masked (the model loses information and drifts), which would
        # tint almost every word the same color. Subtracting the median makes the
        # sign reflect each word's RELATIVE influence.
        #   Not applied for fine: fine uses raw target LOGITS, where the constant
        #   hierarchy prior already cancels in the delta, so there is no baseline
        #   drift to remove — and centering there flips genuinely supporting
        #   phrases to red once most phrases sit above the median.
        if n > 0 and task != 'fine':
            deltas = deltas - float(np.median(deltas))

        for local_idx, pos in enumerate(positions):
            saliency[pos] = deltas[local_idx]
            valid_mask[pos] = True

    return saliency, valid_mask


# =============================================================================
# SPAN (PHRASE) OCCLUSION SALIENCY
# =============================================================================

def _word_token_groups(input_ids_1d, attention_mask_1d, encoding, tokenizer):
    """Group maskable token positions by word.

    Returns an ordered list of (word_id, [token_positions]) for words that are
    fully maskable (not special tokens, not the entity span). Order follows the
    left-to-right appearance of words in the text.
    """
    valid = set(_valid_positions(input_ids_1d, attention_mask_1d, tokenizer))
    word_ids = encoding.word_ids(batch_index=0)

    groups = {}
    order = []
    for tok_idx, wid in enumerate(word_ids):
        if wid is None or tok_idx not in valid:
            continue
        if wid not in groups:
            groups[wid] = []
            order.append(wid)
        groups[wid].append(tok_idx)
    return [(wid, groups[wid]) for wid in order]


def _auto_span_size(num_words):
    """Choose a phrase length automatically based on how much context there is.

    Short contexts → smaller phrases (so we still get several distinct spans);
    longer contexts → larger phrases (so the signal is strong and the number of
    forward passes stays bounded).
    """
    if num_words <= 6:
        return 1
    if num_words <= 15:
        return 2
    if num_words <= 30:
        return 3
    return 4


# Sentence-ending punctuation across the task's five languages (Latin/Cyrillic
# use ./!/?; Devanagari uses the danda । and double danda ॥). Used only as a
# fallback if nltk sentence splitting is unavailable.
_SENTENCE_TERMINATORS = '.!?…।॥'


def _sentence_ids_for_words(word_groups, encoding, marked_text):
    """Assign each word (by index in word_groups) a sentence id.

    Sentences are detected with the project's nltk-based splitter
    (`split_sentences_with_offsets`), which returns char spans. Each word is
    mapped to the sentence whose char span contains the word's start offset, so
    a phrase never crosses a sentence boundary. Punctuation such as the final
    "." correctly belongs to the sentence it ends.

    Returns a list of ints, one per word_group, e.g. [0,0,0,1,1,2,...].
    """
    offsets = encoding['offset_mapping']
    if hasattr(offsets, 'tolist'):
        offsets = offsets.tolist()
    if len(offsets) > 0 and isinstance(offsets[0], (list, tuple)) \
            and len(offsets[0]) > 0 and isinstance(offsets[0][0], (list, tuple)):
        offsets = offsets[0]

    # Character span of each word: min start / max end over its subword tokens.
    word_char_spans = []
    for _wid, tok_positions in word_groups:
        starts = [int(offsets[p][0]) for p in tok_positions]
        ends = [int(offsets[p][1]) for p in tok_positions]
        word_char_spans.append((min(starts), max(ends)))

    # Replace entity markers with equal-length blanks so nltk sees clean text
    # while character offsets stay aligned with `marked_text`.
    clean = marked_text
    for marker in (ENTITY_START_TOKEN, ENTITY_END_TOKEN):
        clean = clean.replace(marker, ' ' * len(marker))

    # nltk sentence spans (start inclusive, end exclusive-ish per the helper).
    try:
        sent_spans = split_sentences_with_offsets(clean)
    except Exception:
        sent_spans = []

    if not sent_spans:
        # Fallback: gap-terminator heuristic.
        sent_ids = []
        sid = 0
        prev_end = None
        for (start, end) in word_char_spans:
            if prev_end is not None:
                if any(t in marked_text[prev_end:start] for t in _SENTENCE_TERMINATORS):
                    sid += 1
            sent_ids.append(sid)
            prev_end = end
        return sent_ids

    # Map each word to the sentence whose [start, end) contains the word start.
    sent_ids = []
    for (w_start, _w_end) in word_char_spans:
        assigned = 0
        for si, (s_start, s_end, _txt) in enumerate(sent_spans):
            if s_start <= w_start < s_end:
                assigned = si
                break
        else:
            # Word start falls outside all sentence spans (e.g. in whitespace
            # that was a marker); attach to the nearest preceding sentence.
            assigned = sent_ids[-1] if sent_ids else 0
        sent_ids.append(assigned)
    return sent_ids


def compute_span_occlusion_saliency(
    input_ids,
    attention_mask,
    model,
    target_class,
    tokenizer,
    encoding,
    marked_text,
    task='coarse',
    coarse_probs=None,
    device='cuda',
    mode='block',        # 'block' (non-overlapping) | 'sliding' (overlapping)
    span_size=None,      # None → auto
    max_phrases=10,      # cap total phrases (grow span_size until it fits)
    batch_size=32,
):
    """Phrase-level occlusion saliency.

    Instead of masking one token at a time, we mask a whole phrase (a span of
    consecutive words) at once and measure the drop in the target probability.
    Masking a phrase removes more context jointly, so the signal is stronger and
    less noisy than single-token occlusion. A phrase of size 1 reduces to
    word-level occlusion. Phrases never cross sentence boundaries (a phrase is
    always contained within a single sentence).

    mode:
        'block'   — non-overlapping consecutive phrases; each word belongs to
                    exactly one phrase. Every word in a phrase gets that phrase's
                    delta (so words in the same phrase share a value).
        'sliding' — overlapping windows; each word's saliency is the mean of the
                    deltas of all windows containing it (smoother attribution).

    span_size: number of words per phrase; None picks it automatically from the
        context length via _auto_span_size.
    marked_text: the string that was tokenized (used to detect sentence
        boundaries so phrases stay within one sentence).

    Returns:
        saliency: np.ndarray (T,) token-level deltas (each token inherits its
                  word's phrase value), 0.0 at skipped positions
        valid_mask: np.ndarray (T,) bool
        info: dict with {'span_size', 'num_phrases', 'num_sentences'}
    """
    if not getattr(tokenizer, 'is_fast', False):
        raise RuntimeError("Fast tokenizer required.")

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    if coarse_probs is not None:
        coarse_probs = coarse_probs.to(device)

    T = input_ids.size(1)
    saliency = np.zeros(T, dtype=np.float32)
    valid_mask = np.zeros(T, dtype=bool)

    word_groups = _word_token_groups(input_ids[0], attention_mask[0], encoding, tokenizer)
    num_words = len(word_groups)
    info = {'span_size': 0, 'num_phrases': 0}
    if num_words == 0:
        return saliency, valid_mask, info

    if span_size is None:
        span_size = _auto_span_size(num_words)
    span_size = max(1, span_size)
    info['span_size'] = span_size

    # Assign each word to a sentence so phrases never cross sentence boundaries.
    sent_ids = _sentence_ids_for_words(word_groups, encoding, marked_text)
    # Group word indices by sentence, preserving left-to-right order.
    sentences = []
    for wi, sid in enumerate(sent_ids):
        if not sentences or sentences[-1][0] != sid:
            sentences.append((sid, [wi]))
        else:
            sentences[-1][1].append(wi)
    info['num_sentences'] = len(sentences)

    # Build the list of phrases as index ranges into word_groups, restricted to
    # within each sentence. Enforce a cap on the total number of phrases by
    # growing the phrase size until the count fits (fewer, larger phrases keep
    # the display readable and the forward passes bounded).
    def _build_phrases(sz):
        out = []
        for _sid, wi_list in sentences:
            m = len(wi_list)
            if mode == 'sliding':
                if m <= sz:
                    out.append(list(wi_list))
                else:
                    for s in range(0, m - sz + 1):
                        out.append(list(wi_list[s:s + sz]))
            else:  # block
                for s in range(0, m, sz):
                    out.append(list(wi_list[s:s + sz]))
        return out

    phrases = _build_phrases(span_size)
    while len(phrases) > max_phrases:
        span_size += 1
        info['span_size'] = span_size
        phrases = _build_phrases(span_size)
    info['num_phrases'] = len(phrases)

    model.eval()
    with torch.no_grad():
        # Same rationale as word-level occlusion: use the target logit for
        # confident fine predictions so the signal doesn't vanish at saturation
        # (and the constant hierarchy prior cancels in the delta).
        score_mode = 'logit' if task == 'fine' else 'prob'

        p_orig = _forward_prob(
            model, input_ids, attention_mask, target_class,
            task=task, coarse_probs=coarse_probs, score_mode=score_mode,
        ).item()

        mask_id = tokenizer.mask_token_id
        n = len(phrases)
        phrase_deltas = np.zeros(n, dtype=np.float32)

        # Precompute the token positions each phrase masks.
        phrase_token_positions = []
        for wi_list in phrases:
            toks = []
            for wi in wi_list:
                toks.extend(word_groups[wi][1])
            phrase_token_positions.append(toks)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            chunk = end - start
            perturbed = input_ids.expand(chunk, -1).clone()
            am_rep = attention_mask.expand(chunk, -1).contiguous()
            for row, phrase_idx in enumerate(range(start, end)):
                for pos in phrase_token_positions[phrase_idx]:
                    perturbed[row, pos] = mask_id

            cp_chunk = None
            if coarse_probs is not None:
                cp_chunk = coarse_probs.expand(chunk, -1).contiguous()

            probs = _forward_prob(
                model, perturbed, am_rep, target_class,
                task=task, coarse_probs=cp_chunk, score_mode=score_mode,
            )
            phrase_deltas[start:end] = (
                p_orig - probs.detach().cpu().numpy()
            ).astype(np.float32)

    # Baseline correction — COARSE ONLY (see compute_occlusion_saliency for the
    # full rationale). Fine uses raw logits where the constant prior cancels in
    # the delta, so centering there would wrongly flip supporting phrases to red.
    if n > 0 and task != 'fine':
        phrase_deltas = phrase_deltas - float(np.median(phrase_deltas))

    # Distribute phrase deltas back to token positions.
    if mode == 'sliding':
        # Average over all phrases covering each word.
        word_sum = np.zeros(num_words, dtype=np.float64)
        word_cnt = np.zeros(num_words, dtype=np.float64)
        for phrase_idx, wi_list in enumerate(phrases):
            for wi in wi_list:
                word_sum[wi] += phrase_deltas[phrase_idx]
                word_cnt[wi] += 1
        word_vals = np.where(word_cnt > 0, word_sum / np.maximum(word_cnt, 1), 0.0)
        for wi, (_, toks) in enumerate(word_groups):
            for pos in toks:
                saliency[pos] = word_vals[wi]
                valid_mask[pos] = True
    else:  # block — each word gets its phrase's delta
        for phrase_idx, wi_list in enumerate(phrases):
            for wi in wi_list:
                for pos in word_groups[wi][1]:
                    saliency[pos] = phrase_deltas[phrase_idx]
                    valid_mask[pos] = True

    return saliency, valid_mask, info


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


def render_saliency_html(marked_text, words, target_label, method_name="occlusion",
                         highlight_spans=None):
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
        highlight_spans: optional list of (start, end) char ranges. When given,
            ONLY words whose character span overlaps one of these ranges are
            tinted — used to color exactly the entries shown on the bar chart.
            Everything else is rendered as plain (dark) text.

    Returns:
        HTML string suitable for st.markdown(..., unsafe_allow_html=True)
    """
    if not words:
        return "<em>No context to analyse.</em>"

    max_abs = max(abs(w['saliency']) for w in words)
    if max_abs < 1e-9:
        max_abs = 1e-9

    # Total influence budget for expressing each word as a share (%).
    total_abs = sum(abs(w['saliency']) for w in words)
    if total_abs <= 0:
        total_abs = 1e-12

    def _on_diagram(w):
        """True if this word should be tinted (overlaps an allowed span)."""
        if highlight_spans is None:
            return True
        for (hs, he) in highlight_spans:
            if w['start'] < he and w['end'] > hs:  # overlap
                return True
        return False

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
        word_html = html.escape(w['word'])
        share = w['saliency'] / total_abs * 100.0
        direction = "supports" if w['saliency'] > 0 else "opposes"
        # Rich tooltip: the word/phrase text, its signed share, and direction.
        tip = html.escape(
            f"“{w['word']}” — {share:+.1f}% ({direction} {target_label})"
        )
        # Inline hover handlers survive Streamlit's HTML sanitizer (a <style>
        # block would be stripped). On hover we add a visible outline + shadow.
        hover_in = (
            "this.style.outline='2px solid #333';"
            "this.style.outlineOffset='1px';"
            "this.style.filter='brightness(0.92)';"
        )
        hover_out = (
            "this.style.outline='none';"
            "this.style.filter='none';"
        )
        hover_attrs = (
            f'onmouseover="{hover_in}" onmouseout="{hover_out}" '
            f'title="{tip}"'
        )

        if abs(s_norm) < DISPLAY_ALPHA_CUTOFF or not _on_diagram(w):
            # Below cutoff OR not on the diagram: no background, but keep an
            # explicit dark color so the word stays readable on the light body
            # background (Streamlit dark theme would otherwise render it white).
            # Still interactive: hovering shows the tooltip + outline.
            pieces.append(
                f'<span style="color:#1a1a1a;cursor:default;border-radius:3px;" '
                f'{hover_attrs}>{word_html}</span>'
            )
        else:
            color = POS_COLOR if s_norm > 0 else NEG_COLOR
            # Floor the alpha so even weak-but-significant words get a clearly
            # visible tint; scale the rest up to MAX_ALPHA.
            alpha = min(MAX_ALPHA, 0.30 + 0.55 * abs(s_norm))
            bg = _rgba(color, alpha)
            pieces.append(
                f'<span style="background:{bg};color:#111;border-radius:3px;'
                f'padding:1px 3px;font-weight:600;cursor:default;" '
                f'{hover_attrs}>{word_html}</span>'
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
        f'<span style="background:{_rgba(POS_COLOR, 0.75)};color:#fff;'
        f'padding:2px 6px;border-radius:3px;font-weight:600;">'
        f'supports {html.escape(target_label)}</span> '
        f'&nbsp;&nbsp;'
        f'<span style="background:{_rgba(NEG_COLOR, 0.75)};color:#fff;'
        f'padding:2px 6px;border-radius:3px;font-weight:600;">'
        f'opposes {html.escape(target_label)}</span> '
        f'&nbsp;&nbsp;<em>method: {html.escape(method_name)}</em>'
        f'</div>'
    )
    body = (
        f'<div style="white-space:pre-wrap;line-height:2.1;font-size:1.05rem;'
        f'font-family:inherit;padding:12px;background:#FFFFFF;color:#1a1a1a;'
        f'border-radius:6px;border:1px solid #DDD;">{joined}</div>'
    )
    return legend + body


# =============================================================================
# BAR CHART
# =============================================================================

def _fmt_share(v, total_abs):
    """Format a saliency value as a signed share (%) of the total influence.

    share = v / sum(|all saliencies|) * 100. The sum of |shares| over all words
    equals 100%, so "+12.3%" reads as "this word carries 12.3% of the total
    influence, in the direction of supporting the class".
    """
    if v == 0 or total_abs <= 0:
        return "0%"
    return f"{(v / total_abs) * 100.0:+.1f}%"


def _sentence_id_at_char(char_pos, sent_spans):
    """Return the index of the sentence span whose [start, end) contains char_pos."""
    for si, (s_start, s_end, _txt) in enumerate(sent_spans):
        if s_start <= char_pos < s_end:
            return si
    return -1


def _group_consecutive_phrases(words, tol=1e-9, marked_text=None):
    """Merge consecutive words that share the same saliency into one phrase.

    Used for block-phrase occlusion, where every word in a masked phrase gets
    the identical delta. Consecutive same-value words are joined into a single
    entry whose 'word' is the phrase text. Words are assumed sorted by 'start'.

    If marked_text is given, sentences are detected with the nltk-based splitter
    and a merge is blocked whenever two words fall in different sentences, so a
    displayed phrase never spans two sentences (even if both sentences' phrases
    happened to get the same delta). This is robust to punctuation being its own
    token (e.g. the "." ending a sentence).

    Returns a new list of {'word', 'start', 'end', 'saliency'} dicts.
    """
    if not words:
        return []

    sent_spans = None
    if marked_text is not None:
        clean = marked_text
        for marker in (ENTITY_START_TOKEN, ENTITY_END_TOKEN):
            clean = clean.replace(marker, ' ' * len(marker))
        try:
            sent_spans = split_sentences_with_offsets(clean)
        except Exception:
            sent_spans = None

    ordered = sorted(words, key=lambda w: w['start'])
    merged = []
    cur = None
    cur_sid = None
    for w in ordered:
        w_sid = _sentence_id_at_char(w['start'], sent_spans) if sent_spans else None
        crosses_sentence = (
            cur is not None and sent_spans and cur_sid is not None
            and w_sid != cur_sid
        )
        if (cur is not None
                and abs(w['saliency'] - cur['saliency']) <= tol
                and not crosses_sentence):
            # Extend the current phrase.
            cur['words'].append(w['word'])
            cur['end'] = w['end']
        else:
            if cur is not None:
                merged.append(cur)
            cur = {
                'words': [w['word']],
                'start': w['start'],
                'end': w['end'],
                'saliency': w['saliency'],
            }
            cur_sid = w_sid
    if cur is not None:
        merged.append(cur)
    # Flatten 'words' list into a display string.
    out = []
    for m in merged:
        out.append({
            'word': ' '.join(m['words']),
            'start': m['start'],
            'end': m['end'],
            'saliency': m['saliency'],
        })
    return out


def select_top_entries(words, top_k=10, group_phrases=False, marked_text=None):
    """Return the top-K entries shown on the bar chart (highest |saliency|).

    Encapsulates the ranking used by both the chart and the highlighter so they
    always agree on which words/phrases are "on the diagram". When group_phrases
    is True, consecutive same-value words are merged into phrases first.

    Returns a list of {'word', 'start', 'end', 'saliency'} dicts, sorted by
    signed saliency descending (same order the chart draws them).
    """
    if not words:
        return []
    entries = _group_consecutive_phrases(words, marked_text=marked_text) if group_phrases else words
    ranked = sorted(entries, key=lambda w: abs(w['saliency']), reverse=True)[:top_k]
    ranked.sort(key=lambda w: w['saliency'], reverse=True)
    return ranked


def plot_saliency_bar(words, target_label, top_k=10, group_phrases=False, marked_text=None):
    """Horizontal bar chart of the top-K words (or phrases) by share of influence.

    Each entry's saliency is expressed as a signed share (%) of the total
    absolute influence across ALL words, so the magnitudes are interpretable
    (shares of |all| sum to 100%). Sorted by signed share descending so
    supporting entries appear at the top and opposing ones at the bottom.

    group_phrases: when True, consecutive words sharing the same saliency value
        (as produced by block-phrase occlusion) are merged into a single bar
        labelled with the whole phrase.
    """
    if not words:
        fig, ax = plt.subplots(figsize=(4.5, 2.5))
        ax.text(0.5, 0.5, "No context", ha='center', va='center')
        ax.axis('off')
        return fig

    # Total influence budget over all words (not just top-k) so shares are
    # meaningful relative to the whole context. Computed BEFORE any phrase
    # grouping so the denominator counts each word once.
    total_abs = sum(abs(w['saliency']) for w in words)
    if total_abs <= 0:
        total_abs = 1e-12

    ranked = select_top_entries(words, top_k=top_k, group_phrases=group_phrases,
                                marked_text=marked_text)

    labels = [w['word'] for w in ranked]
    shares = [w['saliency'] / total_abs * 100.0 for w in ranked]  # signed %
    colors = [POS_COLOR if s > 0 else NEG_COLOR for s in shares]

    # Truncate very long phrase labels so the y-axis stays readable.
    labels = [(lbl if len(lbl) <= 32 else lbl[:29] + '…') for lbl in labels]

    max_abs_share = max((abs(s) for s in shares), default=0.0)

    k = len(ranked)
    fig, ax = plt.subplots(figsize=(6.2, max(2.6, 0.5 * k)))
    y_pos = np.arange(k)
    bars = ax.barh(y_pos, shares, color=colors, edgecolor='white',
                   height=0.7, linewidth=1.5)

    label_offset = max_abs_share * 0.03 if max_abs_share > 0 else 0.5
    for bar, s in zip(bars, shares):
        x = bar.get_width()
        ha = 'left' if x >= 0 else 'right'
        ax.text(
            x + (label_offset if x >= 0 else -label_offset),
            bar.get_y() + bar.get_height() / 2,
            f'{s:+.1f}%',
            va='center', ha=ha, fontsize=9,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.axvline(0, color='#888', linewidth=0.8)
    ax.set_xlabel(f'Share of total influence on {target_label} (%)', fontsize=9)
    ax.grid(axis='x', alpha=0.25, linewidth=0.5)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    ax.spines['left'].set_color('#CCC')
    ax.spines['bottom'].set_color('#CCC')

    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, prune='both'))
    ax.tick_params(axis='x', labelsize=8)

    # Padding so the percentage annotations don't clip.
    xmin = min(shares + [0.0])
    xmax = max(shares + [0.0])
    span = (xmax - xmin) if xmax > xmin else max(max_abs_share, 1.0)
    ax.set_xlim(xmin - span * 0.30, xmax + span * 0.30)

    fig.tight_layout()
    return fig
