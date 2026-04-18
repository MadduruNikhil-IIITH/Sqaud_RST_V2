from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional



@dataclass(frozen=True)
class ProcessedSentenceRecord:
    avg_word_length: float
    sentence_length_words: int
    type_token_ratio: float
    causal_marker_ratio: float
    contrast_marker_ratio: float
    named_entity_density: float
    # Stable identifiers and alignment fields
    sent_id: str
    para_id: str
    sent_idx: int
    sent_text: Optional[str] = None
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    gold_salient: Optional[int] = None
    rst_relation: Optional[str] = None
    rst_nuclearity: Optional[str] = None
    rst_tree_depth: Optional[float] = None
    span_importance_score: Optional[float] = None
    sentence_position_ratio: Optional[float] = None
    named_entity_count: Optional[float] = None
    prev_next_cohesion_score: Optional[float] = None
    paragraph_discourse_continuity_score: Optional[float] = None
    cue_word_flags: Optional[str] = None
    content_word_density: Optional[float] = None
    sentence_length_tokens: Optional[float] = None
    lexical_density: Optional[float] = None
    syntactic_complexity_score: Optional[float] = None
    readability_score: Optional[float] = None
    discourse_marker_features: Optional[str] = None
    pronoun_usage_features: Optional[str] = None
    temporal_marker_features: Optional[str] = None
    pos_ratio_NN: Optional[float] = None
    pos_ratio_NNP: Optional[float] = None
    pos_ratio_NNS: Optional[float] = None
    pos_ratio_VB: Optional[float] = None
    pos_ratio_VBD: Optional[float] = None
    pos_ratio_VBG: Optional[float] = None
    pos_ratio_VBN: Optional[float] = None
    pos_ratio_VBP: Optional[float] = None
    pos_ratio_VBZ: Optional[float] = None
    pos_ratio_JJ: Optional[float] = None
    pos_ratio_RB: Optional[float] = None
    punctuation_pattern_comma_count: Optional[int] = None
    punctuation_pattern_semicolon_count: Optional[int] = None
    concreteness_noun_count: Optional[int] = None
    concreteness_total: Optional[int] = None
    prev_sent_label: Optional[int] = None
    concreteness_ratio: Optional[float] = None
    feature_salience_score: Optional[float] = None
    feature_salience_label: Optional[int] = None
    feature_salience_rank: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GoldSalienceRecord:
    sent_id: str
    para_id: str
    sent_idx: int
    gold_salient: int
    gold_rank_optional: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
