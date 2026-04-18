import re
from collections import Counter
import nltk
from typing import Dict, Any

def tokenize(text):
    return nltk.word_tokenize(text)

def compute_answer_coverage(sent_tokens, answer_tokens):
    if not sent_tokens or not answer_tokens:
        return 0.0
    return len(set(sent_tokens) & set(answer_tokens)) / len(set(answer_tokens))

def compute_pos_ratio_features(text):
    tokens = tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    counts = Counter(tag for word, tag in pos_tags)
    total = sum(counts.values())
    ratios = {tag: count / total for tag, count in counts.items()}
    return ratios

def compute_syntactic_complexity(text):
    # Simple proxy: number of clauses (count of 'and', 'but', 'or', ',')
    return text.count(',') + text.lower().count(' and ') + text.lower().count(' but ') + text.lower().count(' or ')

def compute_readability(text):
    # Flesch Reading Ease (simple version)
    words = tokenize(text)
    num_words = len(words)
    num_sentences = max(1, text.count('.') + text.count('!') + text.count('?'))
    num_syllables = sum(len(re.findall(r'[aeiouy]+', word.lower())) for word in words)
    if num_words == 0:
        return 0.0
    return 206.835 - 1.015 * (num_words / num_sentences) - 84.6 * (num_syllables / num_words)

def compute_discourse_markers(text):
    markers = ['however', 'therefore', 'because', 'although', 'meanwhile', 'moreover', 'thus', 'instead', 'finally', 'first', 'second']
    found = [m for m in markers if m in text.lower()]
    return '|'.join(found)

def compute_pronoun_usage(text):
    pronouns = ['he', 'she', 'it', 'they', 'we', 'i', 'you', 'him', 'her', 'them', 'us', 'me']
    tokens = tokenize(text.lower())
    found = [p for p in tokens if p in pronouns]
    return '|'.join(found)

def compute_temporal_markers(text):
    temporal = ['before', 'after', 'when', 'while', 'during', 'since', 'until', 'now', 'then', 'later', 'soon', 'recently', 'eventually']
    found = [t for t in temporal if t in text.lower()]
    return '|'.join(found)

def compute_punctuation_patterns(text):
    return {
        'exclamation_count': text.count('!'),
        'question_count': text.count('?'),
        'comma_count': text.count(','),
        'semicolon_count': text.count(';'),
        'colon_count': text.count(':'),
        'ellipsis_count': text.count('...'),
    }


def compute_concreteness(text):
    # Dummy: count nouns as proxy for concreteness
    tokens = tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    noun_count = sum(1 for word, tag in pos_tags if tag.startswith('NN'))
    return {'noun_count': noun_count, 'total': len(tokens), 'ratio': noun_count / len(tokens) if tokens else 0.0}

def extract_all_features(sent_text: str, answer_text: str = None) -> Dict[str, Any]:
    import numpy as np
    sent_tokens = tokenize(sent_text)

    # --- Basic word-level features ---
    words = [w for w in sent_tokens if w.isalpha()]
    word_lengths = [len(w) for w in words] if words else [0]
    avg_word_length = float(np.mean(word_lengths))
    sentence_length_words = len(words)
    type_token_ratio = len(set(words)) / len(words) if words else 0.0

    # --- POS tagging and entity features ---
    tagged = nltk.pos_tag(words)
    pos_counts = Counter(tag for _, tag in tagged)
    total_pos = sum(pos_counts.values()) or 1
    ne_count = sum(1 for _, tag in tagged if tag in {"NNP", "NNPS"})
    named_entity_density = ne_count / len(words) if words else 0.0

    # --- Marker ratios ---
    CAUSAL_MARKERS = {"because", "since", "as", "therefore", "thus", "hence", "so"}
    CONTRAST_MARKERS = {"but", "however", "although", "though", "yet", "whereas", "nevertheless"}
    tokens_lower = [w.lower() for w in words]
    total_tokens = len(tokens_lower) or 1
    causal_marker_ratio = sum(w in CAUSAL_MARKERS for w in tokens_lower) / total_tokens
    contrast_marker_ratio = sum(w in CONTRAST_MARKERS for w in tokens_lower) / total_tokens

    # --- Main feature dictionary ---
    features = {
        'pos_ratio_features': compute_pos_ratio_features(sent_text),
        'syntactic_complexity_score': compute_syntactic_complexity(sent_text),
        'readability_score': compute_readability(sent_text),
        'discourse_marker_features': compute_discourse_markers(sent_text),
        'pronoun_usage_features': compute_pronoun_usage(sent_text),
        'temporal_marker_features': compute_temporal_markers(sent_text),
        'concreteness_features': compute_concreteness(sent_text),
        'avg_word_length': avg_word_length,
        'sentence_length_words': sentence_length_words,
        'type_token_ratio': type_token_ratio,
        'causal_marker_ratio': causal_marker_ratio,
        'contrast_marker_ratio': contrast_marker_ratio,
        'named_entity_density': named_entity_density,
    }

    # --- Punctuation features ---
    features.update(compute_punctuation_patterns(sent_text))

    return features
