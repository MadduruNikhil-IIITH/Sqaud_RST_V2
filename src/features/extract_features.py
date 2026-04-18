from __future__ import annotations
import re


import unicodedata

def _clean_text(text: str) -> str:
    # Normalize Unicode, remove control/non-printable characters, and strip
    text = unicodedata.normalize("NFKC", text)
    # Remove non-printable/control characters except common whitespace
    text = ''.join(ch for ch in text if ch.isprintable() or ch in '\t\n\r')
    return text.strip()

import csv
import json
import math
import statistics
import subprocess
import sys
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path
from typing import Any

from src.common.io_utils import write_csv

from src.common.models import ProcessedSentenceRecord
from src.features.feature_utils import extract_all_features


STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "if",
    "then",
    "else",
    "of",
    "to",
    "in",
    "on",
    "for",
    "at",
    "by",
    "with",
    "from",
    "as",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "this",
    "that",
    "these",
    "those",
}

CUE_WORDS = {
    "however",
    "therefore",
    "because",
    "although",
    "meanwhile",
    "moreover",
    "thus",
    "instead",
    "finally",
    "first",
    "second",
}


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _tokenize_words(text: str) -> list[str]:
    out: list[str] = []
    current: list[str] = []
    for ch in text:
        if ch.isalnum() or ch in {"'", "-"}:
            current.append(ch)
        elif current:
            out.append("".join(current))
            current = []
    if current:
        out.append("".join(current))
    return out


def _safe_text_for_stims(text: str) -> str:
    return text.replace("*", "").strip()


def _rst_sentence_map(rst_artifacts_path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    mapping: dict[tuple[str, int], dict[str, Any]] = {}
    if not rst_artifacts_path.exists():
        return mapping

    for artifact in _read_jsonl_rows(rst_artifacts_path):
        para_id = artifact.get("para_id")
        links = artifact.get("sentence_to_discourse_links", [])
        for link in links:
            link_obj = json.loads(link) if isinstance(link, str) else link
            sent_idx = int(link_obj.get("sent_idx", -1))
            if para_id is None or sent_idx < 0:
                continue
            mapping[(para_id, sent_idx)] = {
                "relation": link_obj.get("relation"),
                "nuclearity": link_obj.get("nuclearity"),
                "depth": float(link_obj.get("depth", 0.0)),
            }
    return mapping


def _gold_map(gold_table_path: Path) -> dict[str, int]:
    mapping: dict[str, int] = {}
    if not gold_table_path.exists():
        return mapping
    for row in _read_csv_rows(gold_table_path):
        mapping[row["sent_id"]] = int(row.get("gold_salient", 0) or 0)
    return mapping


def _span_importance_from_nuclearity(nuclearity: str | None) -> float | None:
    if not nuclearity:
        return None
    nuc = nuclearity.upper()
    if nuc == "NN":
        return 1.0
    if "N" in nuc and "S" in nuc:
        return 0.75
    if nuc == "S":
        return 0.5
    return None


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def remove_ipa_and_symbols(text: str) -> str:
    # Remove IPA and other non-ASCII symbols (keep basic Latin, numbers, punctuation)
    return re.sub(r"[^\x00-\x7F]+", "", text)

def _build_psychformers_stimuli(
    sentence_rows: list[dict[str, str]],
    stimuli_dir: Path,
) -> dict[str, Path]:
    stimuli_dir.mkdir(parents=True, exist_ok=True)

    by_para: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in sentence_rows:
        by_para[row["para_id"]].append(row)
    for para_rows in by_para.values():
        para_rows.sort(key=lambda r: int(r["sent_idx"]))

    sentence_stims = stimuli_dir / "sentence_level.stims"
    sentence_meta = stimuli_dir / "sentence_level_meta.csv"
    word_stims = stimuli_dir / "word_level.stims"
    word_meta = stimuli_dir / "word_level_meta.csv"

    sent_lines: list[str] = []
    sent_meta_rows: list[dict[str, Any]] = []
    word_lines: list[str] = []
    word_meta_rows: list[dict[str, Any]] = []

    for para_id, para_rows in by_para.items():
        for i, row in enumerate(para_rows):
            prev_txt = _safe_text_for_stims(para_rows[i - 1]["sent_text"]) if i > 0 else ""
            cur_txt = _safe_text_for_stims(row["sent_text"])
            next_txt = _safe_text_for_stims(para_rows[i + 1]["sent_text"]) if i + 1 < len(para_rows) else ""

            # Remove IPA and non-ASCII symbols from all context
            prev_txt = remove_ipa_and_symbols(prev_txt)
            cur_txt = remove_ipa_and_symbols(cur_txt)
            next_txt = remove_ipa_and_symbols(next_txt)

            sent_line = f"{prev_txt} *{cur_txt}* {next_txt}".strip()
            sent_lines.append(sent_line)
            sent_meta_rows.append(
                {
                    "line_idx": len(sent_lines) - 1,
                    "sent_id": row["sent_id"],
                    "para_id": para_id,
                    "sent_idx": int(row["sent_idx"]),
                    "target_type": "sentence",
                }
            )

            words = _tokenize_words(cur_txt)
            for w_idx, word in enumerate(words):
                if not word:
                    continue
                # Split cur_txt into tokens, insert asterisks only around the target word at w_idx
                tokens = _tokenize_words(cur_txt)
                marked_tokens = [f"*{t}*" if i == w_idx else t for i, t in enumerate(tokens)]
                # Reconstruct the sentence, preserving original spacing as much as possible
                # This will join tokens with a single space, which is safe for PsychFormers
                marked = ' '.join(marked_tokens)
                word_line = f"{prev_txt} {marked} {next_txt}".strip()
                word_lines.append(word_line)
                word_meta_rows.append(
                    {
                        "line_idx": len(word_lines) - 1,
                        "sent_id": row["sent_id"],
                        "para_id": para_id,
                        "sent_idx": int(row["sent_idx"]),
                        "target_type": "word",
                        "target_word": word,
                        "target_word_idx": w_idx,
                    }
                )

    sentence_stims.write_text("\n".join(sent_lines) + "\n", encoding="utf-8")
    word_stims.write_text("\n".join(word_lines) + "\n", encoding="utf-8")

    write_csv(
        sentence_meta,
        sent_meta_rows,
        ["line_idx", "sent_id", "para_id", "sent_idx", "target_type"],
    )
    write_csv(
        word_meta,
        word_meta_rows,
        ["line_idx", "sent_id", "para_id", "sent_idx", "target_type", "target_word", "target_word_idx"],
    )

    return {
        "sentence_stims": sentence_stims,
        "sentence_meta": sentence_meta,
        "word_stims": word_stims,
        "word_meta": word_meta,
    }


def _run_psychformers(
    psychformers_dir: Path,
    stimulus_file: Path,
    output_dir: Path,
    model_name: str,
    decoder: str,
    include_following_context: bool,
    use_cpu: bool,
) -> Path:
    psychformers_dir = psychformers_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    script = (psychformers_dir / "psychformers.py").resolve()
    if not script.exists():
        raise FileNotFoundError(f"PsychFormers script not found: {script}")

    cmd = [
        sys.executable,
        str(script),
        "-i",
        str(stimulus_file),
        "-o",
        str(output_dir),
        "-m",
        model_name,
        "-t",
        "surprisal",
        "-d",
        decoder,
    ]
    if include_following_context:
        cmd.append("-f")
    if use_cpu:
        cmd.append("-cpu")

    subprocess.run(cmd, check=True)

    pattern = f"{stimulus_file.stem}.surprisal.*.{decoder}.output"
    candidates = sorted(output_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No PsychFormers output found matching {pattern} in {output_dir}")
    return candidates[0]


def _read_surprisal_output(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
    except UnicodeDecodeError:
        with path.open("r", encoding="cp1252") as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
    # Clean all text fields in all rows
    def clean_row(row):
        return {k: _clean_text(v) if isinstance(v, str) else v for k, v in row.items()}
    return [clean_row(row) for row in rows]


def _attach_surprisal_features(
    rows: list[dict[str, Any]],
    sentence_meta: Path,
    word_meta: Path,
    sentence_output: Path,
    word_output: Path,
) -> None:
    sent_meta_rows = _read_csv_rows(sentence_meta)
    word_meta_rows = _read_csv_rows(word_meta)
    sent_scores = _read_surprisal_output(sentence_output)
    word_scores = _read_surprisal_output(word_output)

    if len(sent_meta_rows) != len(sent_scores):
        print("[DEBUG] sent_meta_rows length:", len(sent_meta_rows))
        print("[DEBUG] sent_scores length:", len(sent_scores))
        print("[DEBUG] sent_meta_rows sample:", sent_meta_rows[:3])
        print("[DEBUG] sent_scores sample:", sent_scores[:3])
        raise ValueError("Sentence-level PsychFormers output length does not match sentence stimulus metadata")
    if len(word_meta_rows) != len(word_scores):
        print("[DEBUG] word_meta_rows length:", len(word_meta_rows))
        print("[DEBUG] word_scores length:", len(word_scores))
        print("[DEBUG] word_meta_rows sample:", word_meta_rows[:3])
        print("[DEBUG] word_scores sample:", word_scores[:3])
        raise ValueError("Word-level PsychFormers output length does not match word stimulus metadata")

    sent_score_by_id: dict[str, tuple[float, float]] = {}
    for meta, score in zip(sent_meta_rows, sent_scores):
        sent_id = meta["sent_id"]
        total = float(score.get("Surprisal") or score.get("surprisal") or 0.0)
        n_tok = float(score.get("NumTokens") or score.get("numtokens") or 0.0)
        sent_score_by_id[sent_id] = (total, n_tok)

    word_score_by_id: dict[str, list[float]] = defaultdict(list)
    for meta, score in zip(word_meta_rows, word_scores):
        sent_id = meta["sent_id"]
        word_score_by_id[sent_id].append(float(score.get("Surprisal") or score.get("surprisal") or 0.0))

    for row in rows:
        sent_id = row["sent_id"]
        total, n_tok = sent_score_by_id.get(sent_id, (0.0, 0.0))
        ws = word_score_by_id.get(sent_id, [])

        row["surprisal_sentence_total"] = total
        row["surprisal_sentence_per_token"] = (total / n_tok) if n_tok else None
        row["surprisal_word_mean"] = statistics.mean(ws) if ws else None
        row["surprisal_word_max"] = max(ws) if ws else None
        row["surprisal_word_std"] = statistics.pstdev(ws) if len(ws) > 1 else 0.0 if ws else None


def extract_features(
    sentence_table_path: Path,
    gold_table_path: Path,
    rst_artifacts_path: Path,
    output_csv_path: Path,
    psychformers_dir: Path | None = None,
    psychformers_output_dir: Path | None = None,
    psychformers_model: str = "gpt2",
    psychformers_decoder: str = "masked",
    psychformers_following_context: bool = True,
    psychformers_use_cpu: bool = False,
    run_psychformers: bool = False,
) -> list[dict[str, Any]]:
    print(f"[DEBUG] Reading sentence table from: {sentence_table_path}")
    sentence_rows = _read_csv_rows(sentence_table_path)
    print(f"[DEBUG] Loaded {len(sentence_rows)} sentence rows")
    gold_by_sent = _gold_map(gold_table_path)
    print(f"[DEBUG] Loaded gold labels for {len(gold_by_sent)} sentences")
    rst_by_sentence = _rst_sentence_map(rst_artifacts_path)
    print(f"[DEBUG] Loaded RST artifacts for {len(rst_by_sentence)} sentences")

    by_para: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in sentence_rows:
        by_para[row["para_id"]].append(row)
    for para_rows in by_para.values():
        para_rows.sort(key=lambda r: int(r["sent_idx"]))
    print(f"[DEBUG] Grouped into {len(by_para)} paragraphs")

    features: list[dict[str, Any]] = []
    for para_id, para_rows in tqdm(by_para.items(), desc='Extracting features (paragraphs)'):
        para_count = len(para_rows)
        for i, row in tqdm(list(enumerate(para_rows)), desc=f'Para {para_id} sents', leave=False):
            sent_id = row["sent_id"]
            sent_text = _clean_text(row["sent_text"])
            sent_idx = int(row["sent_idx"])

            tokens = _tokenize_words(sent_text)
            token_set = {t.lower() for t in tokens}
            content_tokens = [t for t in tokens if t.lower() not in STOPWORDS]
            cue_hits = sorted({w for w in _tokenize_words(sent_text.lower()) if w in CUE_WORDS})
            ne_count = sum(1 for t in tokens[1:] if t[:1].isupper())

            prev_set: set[str] = set()
            next_set: set[str] = set()
            if i > 0:
                prev_set = {t.lower() for t in _tokenize_words(_clean_text(para_rows[i - 1]["sent_text"]))}
            if i + 1 < para_count:
                next_set = {t.lower() for t in _tokenize_words(_clean_text(para_rows[i + 1]["sent_text"]))}

            cohesion_scores: list[float] = []
            if prev_set:
                cohesion_scores.append(_jaccard(token_set, prev_set))
            if next_set:
                cohesion_scores.append(_jaccard(token_set, next_set))
            cohesion = float(statistics.mean(cohesion_scores)) if cohesion_scores else 0.0

            rst = rst_by_sentence.get((para_id, sent_idx), {})
            relation = rst.get("relation")
            nuclearity = rst.get("nuclearity")
            depth = rst.get("depth")

            continuity_votes = 0
            continuity_total = 0
            if i > 0:
                prev_rst = rst_by_sentence.get((para_id, int(para_rows[i - 1]["sent_idx"])), {})
                if prev_rst.get("relation"):
                    continuity_total += 1
                    continuity_votes += int(prev_rst.get("relation") == relation)
            if i + 1 < para_count:
                next_rst = rst_by_sentence.get((para_id, int(para_rows[i + 1]["sent_idx"])), {})
                if next_rst.get("relation"):
                    continuity_total += 1
                    continuity_votes += int(next_rst.get("relation") == relation)
            continuity = (continuity_votes / continuity_total) if continuity_total else 0.0

            # Extract engineered features
            eng_feats = extract_all_features(sent_text)
            # Map engineered features to ProcessedSentenceRecord fields
            # Assign prev_sent_label as gold_salient of previous sentence in paragraph
            if i > 0:
                prev_sent_id = para_rows[i - 1]["sent_id"]
                prev_sent_label = gold_by_sent.get(prev_sent_id, None)
            else:
                prev_sent_label = None

            record = ProcessedSentenceRecord(
                sent_id=sent_id,
                para_id=para_id,
                sent_idx=sent_idx,
                sent_text=sent_text,
                start_char=int(row.get("start_char", 0) or 0),
                end_char=int(row.get("end_char", 0) or 0),
                gold_salient=gold_by_sent.get(sent_id, 0),
                rst_relation=relation,
                rst_nuclearity=nuclearity,
                rst_tree_depth=float(depth) if depth is not None else None,
                span_importance_score=_span_importance_from_nuclearity(nuclearity),
                sentence_position_ratio=(sent_idx / (para_count - 1)) if para_count > 1 else 0.0,
                named_entity_count=float(ne_count),
                prev_next_cohesion_score=cohesion,
                paragraph_discourse_continuity_score=float(continuity),
                cue_word_flags="|".join(cue_hits),
                content_word_density=(len(content_tokens) / len(tokens)) if tokens else 0.0,
                sentence_length_tokens=float(len(tokens)),
                lexical_density=(len(content_tokens) / len(tokens)) if tokens else 0.0,
                syntactic_complexity_score=eng_feats.get("syntactic_complexity_score"),
                readability_score=eng_feats.get("readability_score"),
                discourse_marker_features=eng_feats.get("discourse_marker_features"),
                pronoun_usage_features=eng_feats.get("pronoun_usage_features"),
                temporal_marker_features=eng_feats.get("temporal_marker_features"),
                pos_ratio_NN=eng_feats.get("pos_ratio_features", {}).get("NN"),
                pos_ratio_NNP=eng_feats.get("pos_ratio_features", {}).get("NNP"),
                pos_ratio_NNS=eng_feats.get("pos_ratio_features", {}).get("NNS"),
                pos_ratio_VB=eng_feats.get("pos_ratio_features", {}).get("VB"),
                pos_ratio_VBD=eng_feats.get("pos_ratio_features", {}).get("VBD"),
                pos_ratio_VBG=eng_feats.get("pos_ratio_features", {}).get("VBG"),
                pos_ratio_VBN=eng_feats.get("pos_ratio_features", {}).get("VBN"),
                pos_ratio_VBP=eng_feats.get("pos_ratio_features", {}).get("VBP"),
                pos_ratio_VBZ=eng_feats.get("pos_ratio_features", {}).get("VBZ"),
                pos_ratio_JJ=eng_feats.get("pos_ratio_features", {}).get("JJ"),
                pos_ratio_RB=eng_feats.get("pos_ratio_features", {}).get("RB"),
                punctuation_pattern_comma_count=eng_feats.get("comma_count"),
                punctuation_pattern_semicolon_count=eng_feats.get("semicolon_count"),
                # punctuation_pattern_colon_count removed (always zero in current data)
                concreteness_noun_count=eng_feats.get("concreteness_features", {}).get("noun_count"),
                concreteness_total=eng_feats.get("concreteness_features", {}).get("total"),
                prev_sent_label=prev_sent_label,
                concreteness_ratio=eng_feats.get("concreteness_features", {}).get("ratio"),
                avg_word_length=eng_feats.get("avg_word_length"),
                sentence_length_words=eng_feats.get("sentence_length_words"),
                type_token_ratio=eng_feats.get("type_token_ratio"),
                causal_marker_ratio=eng_feats.get("causal_marker_ratio"),
                contrast_marker_ratio=eng_feats.get("contrast_marker_ratio"),
                named_entity_density=eng_feats.get("named_entity_density"),
            )
            features.append(record.to_dict())
    print(f"[DEBUG] Extracted {len(features)} feature rows")

    if psychformers_output_dir is None:
        psychformers_output_dir = Path("data/interim/psychformers/output")


    # For inference, do not use seed or sample prefix in output/stims
    stims_dir = psychformers_output_dir.parent / "stims"
    stims_dir.mkdir(parents=True, exist_ok=True)
    print(f"[DEBUG] Building PsychFormers stimuli in: {stims_dir}")
    stimuli_paths = _build_psychformers_stimuli(sentence_rows, stims_dir)
    sentence_out = None
    word_out = None

    if run_psychformers:
        if psychformers_dir is None:
            raise ValueError("run_psychformers=True requires psychformers_dir")
        sentence_out = _run_psychformers(
            psychformers_dir=psychformers_dir,
            stimulus_file=stimuli_paths["sentence_stims"],
            output_dir=psychformers_output_dir,
            model_name=psychformers_model,
            decoder=psychformers_decoder,
            include_following_context=psychformers_following_context,
            use_cpu=psychformers_use_cpu,
        )
        word_out = _run_psychformers(
            psychformers_dir=psychformers_dir,
            stimulus_file=stimuli_paths["word_stims"],
            output_dir=psychformers_output_dir,
            model_name=psychformers_model,
            decoder=psychformers_decoder,
            include_following_context=psychformers_following_context,
            use_cpu=psychformers_use_cpu,
        )
    else:
        # Only match output files with the default stims name
        sentence_candidates = sorted(psychformers_output_dir.glob("sentence_level.surprisal.*.output"))
        word_candidates = sorted(psychformers_output_dir.glob("word_level.surprisal.*.output"))
        if sentence_candidates and word_candidates:
            sentence_out = sentence_candidates[-1]
            word_out = word_candidates[-1]
        else:
            raise FileNotFoundError(f"No matching PsychFormers output found in {psychformers_output_dir}")

    for row in features:
        row["surprisal_sentence_total"] = None
        row["surprisal_sentence_per_token"] = None
        row["surprisal_word_mean"] = None
        row["surprisal_word_max"] = None
        row["surprisal_word_std"] = None
        # Remove unwanted fields if present
        for col in ["feature_salience_score", "feature_salience_label", "feature_salience_rank"]:
            if col in row:
                del row[col]
    print(f"[DEBUG] Set surprisal fields to None for all features and removed unwanted salience columns")

    if sentence_out and word_out:
        print(f"[DEBUG] Attaching PsychFormers surprisal features")
        _attach_surprisal_features(
            rows=features,
            sentence_meta=stimuli_paths["sentence_meta"],
            word_meta=stimuli_paths["word_meta"],
            sentence_output=sentence_out,
            word_output=word_out,
        )

    # Remove specified columns from output (for fieldnames)
    remove_cols = [
        "question_ids", "answer_text", "answer_start_char", "answer_end_char",
        "lexical_overlap_with_answer", "answer_coverage_ratio", "next_sent_label",
        "feature_salience_score", "feature_salience_label", "feature_salience_rank",
        "punctuation_pattern_colon_count",  # Ensure this is removed from output if present
    ]
    if features:
        fieldnames = [f for f in features[0].keys() if f not in remove_cols]
    else:
        fieldnames = []
    print(f"[DEBUG] Writing {len(features)} rows to {output_csv_path} with columns: {fieldnames}")
    write_csv(output_csv_path, features, fieldnames)
    print(f"[DEBUG] Feature CSV written: {output_csv_path}")
    return features
