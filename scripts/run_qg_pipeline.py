"""
Wrapper script to run the QG pipeline for all modes.
"""
import argparse
from src.qg.qg_inference import run_qg_inference
from src.qg.model_loader import SUPPORTED_MODELS, BEST_MODEL

def main():
    parser = argparse.ArgumentParser(description="Run QG pipeline for SQuAD v2 data.")
    parser.add_argument('--model', type=str, default=BEST_MODEL, choices=SUPPORTED_MODELS, help='Model name to use')
    parser.add_argument('--modes', type=str, nargs='+', default=["hybrid_salient", "llm_salient", "zero_shot"], help='Modes to run')
    parser.add_argument('--input_dir', type=str, default="data/inference", help='Input directory')
    parser.add_argument('--output_dir', type=str, default="data/qg/outputs", help='Output directory')
    args = parser.parse_args()
    run_qg_inference(model_name=args.model, modes=args.modes, input_dir=args.input_dir, output_dir=args.output_dir)

if __name__ == "__main__":
    main()
