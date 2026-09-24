#!/usr/bin/env python3
"""
Fix Multiple-Choice Evaluation Labels (Exact Match & ROUGE-L) for ScienceQA and AI2D.

In ScienceQA (and AI2D), the prompt instructs the model to 'Answer with the option letter.'
(e.g., '(A) soft\n(B) bouncy'). The model outputs 'A' or 'B'.
However, the ground truth answers in the raw manifest contain the option text (e.g. ['soft']).
Comparing 'A' against 'soft' resulted in exact_match=0.0 and rougeL_to_target=0.0 across
all 2,000 samples, which caused scikit-learn's LogisticRegression to fail with:
'ValueError: This solver needs samples of at least 2 classes in the data, but the data contains only one class: np.int64(0)'

This script parses the multiple-choice options from each question, resolves the ground-truth
option letter, and updates exact_match and rougeL_to_target accordingly.
All 50 rollouts, log-likelihoods, and embeddings remain completely intact.
"""

import argparse
import os
import pickle
import re
from pathlib import Path


def parse_mc_options(prompt: str) -> dict[str, str]:
    """Extracts multiple choice options like (A) text, (B) text from prompt."""
    options = {}
    for line in prompt.split('\n'):
        line = line.strip()
        m = re.match(r'^\(([A-Z])\)\s*(.+)$', line)
        if m:
            options[m.group(1).upper()] = m.group(2).strip()
    return options


def resolve_target_letter(options: dict[str, str], answers: list[str]) -> str | None:
    """Finds which option letter corresponds to the reference answers."""
    for letter, text in options.items():
        for a in answers:
            a_clean = str(a).strip()
            # Direct text match or direct letter match
            if a_clean.lower() == text.lower() or a_clean.upper() == letter:
                return letter
    return None


def fix_generations_file(file_path: Path | str) -> int:
    """Fixes exact_match and rougeL_to_target for multiple-choice samples in a generations.pkl file."""
    path = Path(file_path)
    if not path.exists():
        print(f"[ERROR] File not found: {path}")
        return 0

    print(f"Loading {path}...")
    with open(path, 'rb') as f:
        data = pickle.load(f)

    fixed_count = 0
    correct_count = 0
    for s in data:
        q = s.get('question_text', '')
        ans = s.get('answers', [])
        pred = str(s.get('most_likely_generation_text', '')).strip(" '()\"\n\r\t").upper()

        options = parse_mc_options(q)
        if not options:
            continue

        target_letter = resolve_target_letter(options, ans)
        if target_letter:
            fixed_count += 1
            pred_letter = pred[:1] if pred else ''
            is_correct = 1.0 if pred_letter == target_letter else 0.0
            if is_correct == 1.0:
                correct_count += 1

            s['exact_match'] = is_correct
            s['rougeL_to_target'] = is_correct
            s['rouge1_to_target'] = is_correct
            s['rouge2_to_target'] = is_correct

    print(f"Processed {len(data)} samples: {fixed_count} MC questions resolved, {correct_count} correct ({correct_count / len(data):.1%}).")

    with open(path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Successfully saved updated file: {path}")
    return fixed_count


def main():
    parser = argparse.ArgumentParser(description="Fix multiple choice labels in generations.pkl")
    parser.add_argument('--input_file', type=str, default=None, help="Path to a single generations.pkl")
    parser.add_argument('--root_dir', type=str, default="output/generations", help="Root directory containing generations")
    args = parser.parse_args()

    if args.input_file:
        fix_generations_file(args.input_file)
    else:
        root = Path(args.root_dir)
        mc_datasets = ['scienceqa', 'ai2d']
        for arch in ['m3', 'mqt']:
            for ds in mc_datasets:
                gen_file = root / arch / ds / 'generations.pkl'
                if gen_file.exists():
                    print(f"\n--- Fixing {arch}/{ds} ---")
                    fix_generations_file(gen_file)


if __name__ == '__main__':
    main()
