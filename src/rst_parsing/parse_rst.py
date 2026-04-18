"""
RST (Rhetorical Structure Theory) parsing module using isanlp_rst.

Parses paragraphs into discourse trees using tchewik/isanlp_rst parser,
extracts sentence-level discourse features, and renders tree visualizations.

API Reference: https://github.com/tchewik/isanlp_rst
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Try to import isanlp RST parser from correct location
try:
    from isanlp_rst.parser import Parser
    ISANLP_AVAILABLE = True
except ImportError:
    ISANLP_AVAILABLE = False
    logger.warning("isanlp_rst not available - RST parsing will use placeholder mode")

from src.common.io_utils import load_json, write_csv, write_jsonl
from src.common.text_utils import locate_sentences_with_offsets, naive_sentence_split


# ============================================================================
# RST PARSER CLASS
# ============================================================================


class RSTParser:
    """Wrapper for isanlp_rst Parser with sentence-level feature extraction."""

    def __init__(self, model_version: str = "gumrrg", cuda_device: int = 0):
        """
        Initialize RST parser.

        Args:
            model_version: Model version ('gumrrg', 'rstdt', 'rstreebank', 'unirst')
            cuda_device: GPU device (0 or higher), -1 for CPU
        """
        if not ISANLP_AVAILABLE:
            raise ImportError(
                "isanlp_rst not installed. "
                "Install: pip install git+https://github.com/iinemo/isanlp.git && pip install isanlp_rst"
            )

        try:
            self.parser = Parser(
                hf_model_name="tchewik/isanlp_rst_v3",
                hf_model_version=model_version,
                cuda_device=cuda_device,
            )
            logger.info(f"RST parser initialized with model={model_version}, cuda_device={cuda_device}")
        except Exception as e:
            logger.error(f"Failed to initialize RST parser: {e}")
            raise

    def parse(self, text: str) -> dict[str, Any]:
        """
        Parse text into RST tree.

        Args:
            text: Paragraph text to parse

        Returns:
            dict with rst_tree (DiscourseUnit), and status
        """
        try:
            # Parse text - returns dict with 'rst' key containing list of DiscourseUnits
            result = self.parser(text)

            # Extract root tree (usually first element)
            rst_tree = result.get("rst", [])
            if not rst_tree:
                return {
                    "status": "error",
                    "rst_tree": None,
                    "error": "No RST tree returned",
                }

            root_tree = rst_tree[0] if isinstance(rst_tree, list) else rst_tree

            return {
                "status": "success",
                "rst_tree": root_tree,
                "raw_result": result,
                "error": None,
            }
        except Exception as e:
            logger.error(f"RST parsing failed: {e}")
            return {
                "status": "error",
                "rst_tree": None,
                "raw_result": None,
                "error": str(e),
            }


# ============================================================================
# RST FEATURE EXTRACTION HELPERS
# ============================================================================


def discourse_unit_to_dict(unit: Any) -> dict[str, Any]:
    """
    Convert DiscourseUnit to serializable dict.

    Args:
        unit: DiscourseUnit from isanlp_rst

    Returns:
        dict representation of the tree node
    """
    if unit is None:
        return {}

    left = getattr(unit, "left", None)
    right = getattr(unit, "right", None)

    return {
        "id": getattr(unit, "id", None),
        "relation": getattr(unit, "relation", None),
        "nuclearity": getattr(unit, "nuclearity", None),
        "start": getattr(unit, "start", None),
        "end": getattr(unit, "end", None),
        "entropy": getattr(unit, "entropy", None),
        "left": discourse_unit_to_dict(left) if left is not None else None,
        "right": discourse_unit_to_dict(right) if right is not None else None,
    }


def _assign_sentence_links(
    node: Any,
    sentence_offsets: list[tuple[int, int]],
    sentence_links: list[dict[str, Any]],
    depth: int,
    parent_nuclearity: str = None,
) -> None:
    if node is None:
        return

    node_start = getattr(node, "start", None)
    node_end = getattr(node, "end", None)
    relation = getattr(node, "relation", None)
    nuclearity = getattr(node, "nuclearity", None)

    # For elementary units, assign parent's relation and nuclearity
    is_elementary = relation == "elementary"
    effective_nuclearity = parent_nuclearity[1] if is_elementary and parent_nuclearity else nuclearity
    effective_relation = parent_nuclearity[0] if is_elementary and parent_nuclearity else relation

    if node_start is not None and node_end is not None:
        for sent_idx, (sent_start, sent_end) in enumerate(sentence_offsets):
            if node_start < sent_end and node_end > sent_start:
                current_depth = sentence_links[sent_idx].get("depth", 0)
                if depth >= current_depth:
                    sentence_links[sent_idx] = {
                        "sent_idx": sent_idx,
                        "relation": effective_relation,
                        "nuclearity": effective_nuclearity,
                        "depth": depth,
                    }

    # Pass (relation, nuclearity) as parent info
    parent_info = (relation, nuclearity)
    _assign_sentence_links(getattr(node, "left", None), sentence_offsets, sentence_links, depth + 1, parent_info)
    _assign_sentence_links(getattr(node, "right", None), sentence_offsets, sentence_links, depth + 1, parent_info)


def extract_sentence_discourse_links(rst_tree: Any, paragraph_text: str) -> list[str]:
    sentences = naive_sentence_split(paragraph_text)
    if not sentences:
        return []

    sentence_offsets = locate_sentences_with_offsets(paragraph_text, sentences)
    sentence_links: list[dict[str, Any]] = [
        {"sent_idx": idx, "relation": None, "nuclearity": None, "depth": 0}
        for idx in range(len(sentences))
    ]

    _assign_sentence_links(rst_tree, sentence_offsets, sentence_links, depth=1)
    return [json.dumps(link) for link in sentence_links]


# ============================================================================
# PIPELINE FUNCTIONS
# ============================================================================

# Try to initialize RST parser globally
_RST_PARSER: RSTParser | None = None
if ISANLP_AVAILABLE:
    try:
        _RST_PARSER = RSTParser(model_version="gumrrg", cuda_device=0)
    except Exception as e:
        logger.warning(f"Failed to initialize RST parser: {e}. Using placeholder mode.")


def parse_rst_placeholder(sample_manifest_path: Path, output_jsonl: Path, rs3_root: Path = None) -> list[dict[str, Any]]:
    """
    Parse RST structures from paragraphs.

    Uses real isanlp_rst parser if available, otherwise generates placeholder data.

    Args:
        sample_manifest_path: Path to cleaned sample manifest
        output_jsonl: Output path for RST artifacts
        rs3_root: Optional output directory for RS3 files. If None, uses env var or default.

    Returns:
        List of RST artifact records
    """
    from tqdm import tqdm
    manifest = load_json(sample_manifest_path)
    import os
    rows: list[dict[str, Any]] = []
    # Allow caller to specify rs3_root, else use env var, else default
    if rs3_root is None:
        rs3_root = Path("data/artifacts/rst_rs3")
    rs3_root.mkdir(parents=True, exist_ok=True)

    paragraphs = manifest["paragraphs"]
    for paragraph in tqdm(paragraphs, desc="Parsing RST trees", total=len(paragraphs)):
        para_id = paragraph["para_id"]
        context = paragraph.get("context", "")

        if _RST_PARSER and context:
            try:
                # Parse with real RST parser
                parse_result = _RST_PARSER.parse(context)

                if parse_result["status"] == "success":
                    rst_tree = parse_result.get("rst_tree")
                    tree_dict = discourse_unit_to_dict(rst_tree)
                    sentence_links = extract_sentence_discourse_links(rst_tree, context)
                    rs3_path = rs3_root / f"{para_id}.rs3"

                    try:
                        rst_tree.to_rs3(str(rs3_path))
                        rs3_path_str = str(rs3_path)
                    except Exception as rs3_error:
                        logger.warning(f"RS3 export failed for {para_id}: {rs3_error}")
                        rs3_path_str = ""

                    rows.append({
                        "para_id": para_id,
                        "rst_tree_json": json.dumps(tree_dict),
                        "sentence_to_discourse_links": sentence_links,
                        "parse_status": "success",
                        "rs3_path": rs3_path_str,
                    })
                else:
                    rows.append({
                        "para_id": para_id,
                        "rst_tree_json": None,
                        "sentence_to_discourse_links": [],
                        "parse_status": f"error: {parse_result.get('error', 'unknown')}",
                        "rs3_path": "",
                    })
            except Exception as e:
                logger.error(f"RST parsing failed for {para_id}: {e}")
                rows.append({
                    "para_id": para_id,
                    "rst_tree_json": None,
                    "sentence_to_discourse_links": [],
                    "parse_status": f"error: {str(e)}",
                    "rs3_path": "",
                })
        else:
            # Placeholder mode
            rows.append({
                "para_id": para_id,
                "rst_tree_json": None,
                "sentence_to_discourse_links": [],
                "parse_status": "pending_real_parser",
                "rs3_path": "",
            })

    write_jsonl(output_jsonl, rows)
    return rows


def build_rst_image_manifest(
    sample_manifest_path: Path, 
    image_root: Path, 
    output_csv: Path, 
    image_format: str = "png"
) -> list[dict[str, Any]]:
    """
    Build manifest of RST tree image paths.
    
    Tracks which paragraphs were successfully parsed for visualization.
    
    Args:
        sample_manifest_path: Path to cleaned sample manifest
        image_root: Directory to store rendered images
        output_csv: Output manifest CSV
        image_format: Image format (png, svg, etc.)
        
    Returns:
        List of image manifest records
    """
    manifest = load_json(sample_manifest_path)
    image_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    # Load RST artifacts to check parse status
    rst_artifacts_path = Path("data/artifacts/rst_artifacts.jsonl")
    rst_status_map: dict[str, str] = {}

    if rst_artifacts_path.exists():
        with rst_artifacts_path.open("r", encoding="utf-8") as f:
            for line in f:
                artifact = json.loads(line)
                rst_status_map[artifact["para_id"]] = artifact.get("parse_status", "unknown")

    for paragraph in manifest["paragraphs"]:
        para_id = paragraph["para_id"]
        image_name = f"{para_id}.{image_format}"
        image_path = image_root / image_name

        rst_parse_status = rst_status_map.get(para_id, "unknown")
        render_status = "pending_real_render" if rst_parse_status == "success" else rst_parse_status

        rows.append({
            "para_id": para_id,
            "image_path": str(image_path),
            "image_format": image_format,
            "render_status": render_status,
        })

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["para_id", "image_path", "image_format", "render_status"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return rows
