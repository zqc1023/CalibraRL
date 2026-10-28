import argparse
import json
import math
import random
import re
import string
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

SYSTEM_PROMPT = (
    "You are a helpful, respectful, and honest assistant. "
    "Answer the question with a couple of words using the provided documents "
    "only when you are confident the answer is correct. Otherwise, respond "
    "exactly with 'I don't know'."
)

REFUSAL_EXPRESSIONS = {
    "noanswer",
    "no answer",
    "cannot answer",
    "i don't know",
    "i do not know",
    "i don't have the information",
    "i do not have the information",
    "i cannot answer that question",
    "uncertain",
    "i'm not sure",
    "i am not sure",
    "i'm not aware of",
    "i am not aware of",
    "i couldn't find",
    "i could not find",
    "unknown",
}


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_answer(text: str) -> str:
    text = str(text).lower()
    text = "".join(char for char in text if char not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


NORMALIZED_REFUSALS = {normalize_answer(value) for value in REFUSAL_EXPRESSIONS}
REFUSAL_PREFIXES = [
    "i dont know",
    "i do not know",
    "i cannot answer",
    "i dont have the information",
    "i do not have the information",
    "im not sure",
    "i am not sure",
    "im not aware of",
    "i am not aware of",
    "i couldnt find",
    "i could not find",
    "cannot answer",
]


def is_refusal(prediction: str) -> bool:
    normalized = normalize_answer(prediction)
    if normalized in NORMALIZED_REFUSALS:
        return True
    return any(
        normalized.startswith(normalize_answer(prefix) + " ")
        for prefix in REFUSAL_PREFIXES
    )


def sub_exact_match(prediction: str, golden_answers) -> bool:
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    return any(
        (normalized_gold := normalize_answer(gold))
        and normalized_gold in normalized_prediction
        for gold in golden_answers
    )


def load_qa_data(path: str, max_examples: int | None):
    records = []
    with open(path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                print(f"Skip line {line_number}: {error}")
                continue
            if "question" not in item or "answer" not in item:
                print(f"Skip line {line_number}: missing question or answer")
                continue
            item.setdefault("documents", [])
            records.append(item)
            if max_examples is not None and len(records) >= max_examples:
                break
    return records


def load_model(base_model_path: str, lora_model_path: str | None):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    if lora_model_path:
        model = PeftModel.from_pretrained(base_model, lora_model_path)
        model = model.merge_and_unload()
    else:
        model = base_model
    model.eval()
    return model, tokenizer


def build_model_inputs(item, tokenizer, model, max_input_length: int):
    documents = item.get("documents", [])
    if isinstance(documents, str):
        documents = [documents]
    final_documents = "\n".join(
        f"document {index}: {document}"
        for index, document in enumerate(documents, start=1)
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Question: {item['question']}\n" f"Documents: {final_documents}"
            ),
        },
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_length,
    )
    return {key: value.to(model.device) for key, value in inputs.items()}


def get_terminators(tokenizer):
    terminators = []
    if tokenizer.eos_token_id is not None:
        terminators.append(tokenizer.eos_token_id)
    eot_id = tokenizer.get_vocab().get("<|eot_id|>")
    if isinstance(eot_id, int) and eot_id >= 0:
        terminators.append(eot_id)
    return list(dict.fromkeys(terminators))


def calibrate_confidence(raw_confidence: float, temperature: float, bias: float):
    epsilon = 1e-6
    bounded = min(max(raw_confidence, epsilon), 1.0 - epsilon)
    raw_logit = math.log(bounded / (1.0 - bounded))
    calibrated_logit = (raw_logit + bias) / temperature
    if calibrated_logit >= 0:
        value = 1.0 / (1.0 + math.exp(-calibrated_logit))
    else:
        exp_value = math.exp(calibrated_logit)
        value = exp_value / (1.0 + exp_value)
    return float(min(max(value, 0.0), 1.0))


def generate_batch(
    item,
    model,
    tokenizer,
    num_samples: int,
    args,
):
    """Sample responses and compute mean content-token probability."""
    import torch

    model_inputs = build_model_inputs(
        item,
        tokenizer=tokenizer,
        model=model,
        max_input_length=args.max_input_length,
    )
    prompt_width = model_inputs["input_ids"].shape[1]
    terminators = get_terminators(tokenizer)
    eos_token_id = None
    if len(terminators) == 1:
        eos_token_id = terminators[0]
    elif len(terminators) > 1:
        eos_token_id = terminators

    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": True,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "num_return_sequences": num_samples,
        "pad_token_id": tokenizer.pad_token_id,
        "return_dict_in_generate": True,
        "output_scores": True,
    }
    if args.top_k >= 0:
        generation_kwargs["top_k"] = args.top_k
    if eos_token_id is not None:
        generation_kwargs["eos_token_id"] = eos_token_id

    with torch.inference_mode():
        generation = model.generate(**model_inputs, **generation_kwargs)
        transition_scores = model.compute_transition_scores(
            generation.sequences,
            generation.scores,
            normalize_logits=True,
        )

    score_width = transition_scores.shape[1]
    generated_ids = generation.sequences[
        :,
        prompt_width : prompt_width + score_width,
    ]
    predictions = tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )

    rows = []
    for index, prediction in enumerate(predictions):
        prediction = prediction.strip()
        refused = is_refusal(prediction)
        correct = (
            False
            if refused
            else sub_exact_match(
                prediction,
                item["answer"],
            )
        )

        content_mask = torch.ones_like(generated_ids[index], dtype=torch.bool)
        for terminator in terminators:
            content_mask &= generated_ids[index].ne(terminator)
        if tokenizer.pad_token_id is not None:
            content_mask &= generated_ids[index].ne(tokenizer.pad_token_id)

        valid_log_probs = transition_scores[index][content_mask]
        raw_confidence = (
            float(valid_log_probs.float().exp().mean().item())
            if valid_log_probs.numel()
            else 0.0
        )
        confidence = calibrate_confidence(
            raw_confidence,
            temperature=args.confidence_temperature,
            bias=args.confidence_bias,
        )
        rows.append(
            {
                "prediction": prediction,
                "refused": bool(refused),
                "correct": bool(correct),
                "raw_confidence": raw_confidence,
                "confidence": confidence,
            }
        )
    return rows


def sample_responses(item, model, tokenizer, num_samples: int, args):
    rows = []
    remaining = num_samples
    while remaining > 0:
        current = min(remaining, args.generation_batch_size)
        rows.extend(
            generate_batch(
                item,
                model=model,
                tokenizer=tokenizer,
                num_samples=current,
                args=args,
            )
        )
        remaining -= current
    return rows


def count_outcomes(rows):
    abstained = sum(row["refused"] for row in rows)
    correct = sum(row["correct"] for row in rows)
    wrong = len(rows) - abstained - correct
    return int(correct), int(wrong), int(abstained)


def run_sampling(data, model, tokenizer, args, output_dir: Path):
    primary_rows = []
    idk_rows = []

    primary_path = output_dir / "primary_rollouts.jsonl"
    idk_path = output_dir / "idk_reference_rollouts.jsonl"

    with open(primary_path, "w", encoding="utf-8") as primary_file, open(
        idk_path,
        "w",
        encoding="utf-8",
    ) as idk_file:
        for item_index, item in enumerate(tqdm(data, desc="sampling questions")):
            # Separate deterministic streams make the primary and reference
            # groups reproducible and independent of earlier IDK decisions.
            set_seed(args.seed + 2 * item_index)
            rollouts = sample_responses(
                item,
                model=model,
                tokenizer=tokenizer,
                num_samples=args.primary_samples,
                args=args,
            )
            correct, wrong, abstained = count_outcomes(rollouts)

            for sample_index, row in enumerate(rollouts):
                record = {
                    "question_index": item_index,
                    "sample_index": sample_index,
                    "question": item["question"],
                    "gold_answer": item["answer"],
                    "llm_know": item.get("llm_know"),
                    **row,
                }
                primary_rows.append(record)
                primary_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            primary_file.flush()

            if abstained == 0:
                continue

            set_seed(args.seed + 2 * item_index + 1)
            reference_rollouts = sample_responses(
                item,
                model=model,
                tokenizer=tokenizer,
                num_samples=args.reference_samples,
                args=args,
            )
            ref_correct, ref_wrong, ref_abstained = count_outcomes(reference_rollouts)
            num_reference_answers = ref_correct + ref_wrong
            reference_accuracy = (
                ref_correct / num_reference_answers
                if num_reference_answers > 0
                else None
            )
            idk_record = {
                "question_index": item_index,
                "question": item["question"],
                "gold_answer": item["answer"],
                "llm_know": item.get("llm_know"),
                "primary_correct": correct,
                "primary_wrong": wrong,
                "primary_idk": abstained,
                "reference_correct": ref_correct,
                "reference_wrong": ref_wrong,
                "reference_idk": ref_abstained,
                "reference_answer_accuracy": reference_accuracy,
                "reference_rollouts": reference_rollouts,
            }
            idk_rows.append(idk_record)
            idk_file.write(json.dumps(idk_record, ensure_ascii=False) + "\n")
            idk_file.flush()

    return primary_rows, idk_rows


def load_saved_records(output_dir: Path):
    primary_path = output_dir / "primary_rollouts.jsonl"
    idk_path = output_dir / "idk_reference_rollouts.jsonl"
    if not primary_path.exists() or not idk_path.exists():
        raise FileNotFoundError(
            "--plot_only requires primary_rollouts.jsonl and "
            "idk_reference_rollouts.jsonl in --output_dir"
        )

    def read_jsonl(path):
        rows = []
        with open(path, "r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    rows.append(json.loads(line))
        return rows

    return read_jsonl(primary_path), read_jsonl(idk_path)


def get_confidence_bins(labels, confidences, n_bins: int):
    labels = np.asarray(labels, dtype=np.float64)
    confidences = np.asarray(confidences, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.minimum(np.digitize(confidences, edges[1:-1]), n_bins - 1)
    rows = []
    for bin_index in range(n_bins):
        mask = bin_ids == bin_index
        count = int(mask.sum())
        rows.append(
            {
                "lower": float(edges[bin_index]),
                "upper": float(edges[bin_index + 1]),
                "count": count,
                "mean_confidence": (float(confidences[mask].mean()) if count else None),
                "empirical_accuracy": (float(labels[mask].mean()) if count else None),
            }
        )
    return rows


def plot_confidence_reliability(answer_rows, output_path: Path, n_bins: int):
    labels = np.asarray([int(row["correct"]) for row in answer_rows])
    confidences = np.asarray([row["confidence"] for row in answer_rows])
    bin_rows = get_confidence_bins(labels, confidences, n_bins)
    nonempty = [row for row in bin_rows if row["count"]]

    fig, axis = plt.subplots(figsize=(6.4, 5.6))
    axis.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="#8C8C8C",
        linewidth=1.4,
        label="Perfect calibration",
    )
    if nonempty:
        x = [row["mean_confidence"] for row in nonempty]
        y = [row["empirical_accuracy"] for row in nonempty]
        counts = np.asarray([row["count"] for row in nonempty])
        axis.plot(x, y, color="#2F65B0", linewidth=2.1, zorder=2)
        axis.scatter(
            x,
            y,
            s=38 + 4 * np.sqrt(counts),
            color="#2F65B0",
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
            label="Empirical accuracy",
        )
    else:
        axis.text(0.5, 0.5, "No answered responses", ha="center")

    ece = sum(
        row["count"]
        / max(len(answer_rows), 1)
        * abs(row["empirical_accuracy"] - row["mean_confidence"])
        for row in nonempty
    )
    axis.text(
        0.97,
        0.05,
        f"n = {len(answer_rows):,}\nECE = {ece:.3f}",
        ha="right",
        va="bottom",
        transform=axis.transAxes,
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#BBBBBB"},
    )
    axis.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="Mean confidence",
        ylabel="Empirical accuracy",
        title="(a) Confidence Reflects Answer Reliability",
    )
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return bin_rows, float(ece)


def plot_incorrect_confidence(
    incorrect_rows,
    output_path: Path,
    n_bins: int,
    threshold: float,
):
    values = np.asarray([row["confidence"] for row in incorrect_rows])
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    counts, _ = np.histogram(values, bins=edges)
    widths = np.diff(edges)
    centers = edges[:-1] + widths / 2
    colors = ["#D9534F" if center >= threshold else "#F2A65A" for center in centers]

    fig, axis = plt.subplots(figsize=(6.4, 5.6))
    axis.bar(
        edges[:-1],
        counts,
        width=widths,
        align="edge",
        color=colors,
        edgecolor="white",
        linewidth=0.8,
    )
    axis.axvline(
        threshold,
        color="#C62828",
        linestyle="--",
        linewidth=1.8,
        label=rf"High-confidence threshold $\gamma={threshold:.1f}$",
    )
    high_count = int((values >= threshold).sum())
    high_ratio = high_count / len(values) if len(values) else float("nan")
    ratio_text = (
        f"High-confidence errors\n{high_count:,}/{len(values):,} "
        f"({100 * high_ratio:.1f}%)"
        if len(values)
        else "No incorrect responses"
    )
    axis.text(
        0.97,
        0.95,
        ratio_text,
        ha="right",
        va="top",
        transform=axis.transAxes,
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#C62828"},
    )
    axis.set(
        xlim=(0, 1),
        xlabel="Confidence of incorrect responses",
        ylabel="Number of errors",
        title="(b) Incorrect Answers Have Heterogeneous Confidence",
    )
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return high_count, float(high_ratio)


def plot_idk_counterfactual(
    idk_rows,
    output_path: Path,
    n_bins: int,
    threshold: float,
):
    usable = [
        row for row in idk_rows if row.get("reference_answer_accuracy") is not None
    ]
    values = np.asarray(
        [row["reference_answer_accuracy"] for row in usable],
        dtype=np.float64,
    )
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    counts, _ = np.histogram(values, bins=edges)
    widths = np.diff(edges)
    centers = edges[:-1] + widths / 2
    colors = ["#E88178" if center >= threshold else "#82B77D" for center in centers]

    fig, axis = plt.subplots(figsize=(6.4, 5.6))
    axis.bar(
        edges[:-1],
        counts,
        width=widths,
        align="edge",
        color=colors,
        edgecolor="white",
        linewidth=0.8,
    )
    axis.axvline(
        threshold,
        color="#333333",
        linestyle="--",
        linewidth=1.8,
        label=rf"Answerability threshold $\tau={threshold:.1f}$",
    )
    justified = int((values < threshold).sum())
    over = int((values >= threshold).sum())
    insufficient = len(idk_rows) - len(usable)
    axis.text(
        0.03,
        0.95,
        f"Justified-IDK side\n{justified:,} questions",
        ha="left",
        va="top",
        color="#2E6B2E",
        transform=axis.transAxes,
    )
    axis.text(
        0.97,
        0.95,
        f"Potential over-IDK side\n{over:,} questions",
        ha="right",
        va="top",
        color="#A62828",
        transform=axis.transAxes,
    )
    if insufficient:
        axis.text(
            0.5,
            0.04,
            f"Excluded: {insufficient} all-IDK reference groups",
            ha="center",
            transform=axis.transAxes,
            fontsize=9,
        )
    axis.set(
        xlim=(0, 1),
        xlabel="Counterfactual answer accuracy among non-IDK samples",
        ylabel="Number of IDK questions",
        title="(c) IDK Responses Have Heterogeneous Counterfactual Value",
    )
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False, loc="upper center")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return justified, over, insufficient


def make_plots(primary_rows, idk_rows, args, output_dir: Path):
    answer_rows = [row for row in primary_rows if not row["refused"]]
    incorrect_rows = [row for row in answer_rows if not row["correct"]]
    if not answer_rows:
        raise ValueError("No answered primary responses; cannot plot confidence.")

    bin_rows, ece = plot_confidence_reliability(
        answer_rows,
        output_path=output_dir / "confidence_reliability.png",
        n_bins=args.n_bins,
    )
    high_count, high_ratio = plot_incorrect_confidence(
        incorrect_rows,
        output_path=output_dir / "incorrect_confidence_distribution.png",
        n_bins=args.n_bins,
        threshold=args.high_confidence_threshold,
    )
    justified, over, insufficient = plot_idk_counterfactual(
        idk_rows,
        output_path=output_dir / "idk_counterfactual_answerability.png",
        n_bins=args.n_bins,
        threshold=args.answerability_threshold,
    )

    summary = {
        "definitions": {
            "confidence": (
                "arithmetic mean of generated content-token probabilities, "
                "optionally followed by binary temperature scaling"
            ),
            "confidence_population": "non-IDK primary rollouts only",
            "high_confidence_error": (
                f"incorrect and confidence >= {args.high_confidence_threshold}"
            ),
            "idk_question": "at least one IDK in the primary rollout group",
            "counterfactual_answer_accuracy": (
                "reference_correct / (reference_correct + reference_wrong)"
            ),
            "potential_over_idk": (
                "counterfactual answer accuracy >= " f"{args.answerability_threshold}"
            ),
        },
        "sampling": {
            "primary_samples_per_question": args.primary_samples,
            "reference_samples_per_idk_question": args.reference_samples,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "seed": args.seed,
        },
        "confidence": {
            "num_answered_rollouts": len(answer_rows),
            "num_correct_rollouts": sum(row["correct"] for row in answer_rows),
            "num_incorrect_rollouts": len(incorrect_rows),
            "ece": ece,
            "bins": bin_rows,
            "high_confidence_error_count": high_count,
            "high_confidence_error_ratio_among_errors": high_ratio,
        },
        "idk_counterfactual": {
            "num_idk_questions": len(idk_rows),
            "num_usable_idk_questions": len(idk_rows) - insufficient,
            "num_justified_side": justified,
            "num_potential_over_idk_side": over,
            "num_insufficient_reference_evidence": insufficient,
            "potential_over_idk_ratio": (
                over / (justified + over) if justified + over else None
            ),
        },
    }
    with open(output_dir / "motivation_statistics.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate three empirical motivation figures"
    )
    parser.add_argument("--input", help="Input JSONL dataset")
    parser.add_argument("--base_model", help="Base model path or model name")
    parser.add_argument("--lora_model", default=None, help="Optional LoRA path")
    parser.add_argument("--output_dir", default="motivation_statistics")
    parser.add_argument("--primary_samples", type=int, default=16)
    parser.add_argument("--reference_samples", type=int, default=64)
    parser.add_argument("--generation_batch_size", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--max_input_length", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--confidence_temperature", type=float, default=1.0)
    parser.add_argument("--confidence_bias", type=float, default=0.0)
    parser.add_argument("--high_confidence_threshold", type=float, default=0.8)
    parser.add_argument("--answerability_threshold", type=float, default=0.5)
    parser.add_argument("--n_bins", type=int, default=10)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--plot_only",
        action="store_true",
        help="Replot from JSONL files already present in output_dir",
    )
    return parser.parse_args()


def validate_args(args):
    if not args.plot_only and (not args.input or not args.base_model):
        raise ValueError("--input and --base_model are required unless --plot_only")
    if args.primary_samples <= 0 or args.reference_samples <= 0:
        raise ValueError("sample counts must be positive")
    if args.generation_batch_size <= 0:
        raise ValueError("--generation_batch_size must be positive")
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive")
    if not 0 < args.top_p <= 1:
        raise ValueError("--top_p must lie in (0, 1]")
    if args.confidence_temperature <= 0:
        raise ValueError("--confidence_temperature must be positive")
    if not 0 <= args.high_confidence_threshold <= 1:
        raise ValueError("--high_confidence_threshold must lie in [0, 1]")
    if not 0 <= args.answerability_threshold <= 1:
        raise ValueError("--answerability_threshold must lie in [0, 1]")
    if args.n_bins < 2:
        raise ValueError("--n_bins must be at least 2")


def main():
    args = parse_args()
    validate_args(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        primary_rows, idk_rows = load_saved_records(output_dir)
    else:
        data = load_qa_data(args.input, args.max_examples)
        if not data:
            raise ValueError("No valid examples found in the input dataset")
        print(f"Loaded {len(data)} questions")
        model, tokenizer = load_model(args.base_model, args.lora_model)
        primary_rows, idk_rows = run_sampling(
            data,
            model=model,
            tokenizer=tokenizer,
            args=args,
            output_dir=output_dir,
        )

    summary = make_plots(primary_rows, idk_rows, args, output_dir)
    print("\nSaved empirical motivation analysis to:", output_dir.resolve())
    print("1.", (output_dir / "confidence_reliability.png").resolve())
    print("2.", (output_dir / "incorrect_confidence_distribution.png").resolve())
    print("3.", (output_dir / "idk_counterfactual_answerability.png").resolve())
    print("4.", (output_dir / "motivation_statistics.json").resolve())
    print(
        "High-confidence error ratio among errors:",
        summary["confidence"]["high_confidence_error_ratio_among_errors"],
    )
    print(
        "Potential over-IDK ratio:",
        summary["idk_counterfactual"]["potential_over_idk_ratio"],
    )


if __name__ == "__main__":
    main()
