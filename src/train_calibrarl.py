import argparse
import json
import logging
import math
import os
import random
import shutil
import string
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from datasets import Dataset as HFDataset
from transformers import AutoTokenizer, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer

try:
    from peft import LoraConfig, TaskType
except ImportError:
    LoraConfig = None
    TaskType = None

SYSTEM_PROMPT = "You are a helpful, respectful, and honest assistant. Answer the question with a couple of words using the provided documents only when you are confident the answer is correct. Otherwise, respond exactly with 'I don't know'."
DEFAULT_EVAL_RATIO = 0.1
DEFAULT_CALIBRATION_RATIO = 0.1
DEFAULT_LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
CANONICAL_ABSTENTION = "I don't know"
ABSTENTION_MARKERS = [
    "I don't know",
    "I don't know.",
    "i do not know",
    "not sure",
    "cannot verify",
    "can't verify",
    "insufficient information",
    "not enough information",
    "cannot determine",
    "unable to determine",
    "uncertain",
    "uncertain.",
    "I can't answer",
    "I can't answer.",
]


def _beta_continued_fraction(
    a: float,
    b: float,
    x: float,
    max_iterations: int = 200,
    tolerance: float = 3e-14,
) -> float:
    """Evaluate the continued fraction used by the incomplete beta CDF."""
    tiny = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    result = d

    for iteration in range(1, max_iterations + 1):
        even_index = 2 * iteration
        coefficient = (
            iteration * (b - iteration) * x / ((qam + even_index) * (a + even_index))
        )
        d = 1.0 + coefficient * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + coefficient / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        result *= d * c

        coefficient = (
            -(a + iteration)
            * (qab + iteration)
            * x
            / ((a + even_index) * (qap + even_index))
        )
        d = 1.0 + coefficient * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + coefficient / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        result *= delta
        if abs(delta - 1.0) <= tolerance:
            return result

    raise RuntimeError(
        "Incomplete beta continued fraction failed to converge "
        f"for a={a}, b={b}, x={x}."
    )


def regularized_incomplete_beta(
    x: float,
    a: float,
    b: float,
) -> float:
    """Return I_x(a, b), the CDF of a Beta(a, b) distribution."""
    if not 0.0 <= x <= 1.0:
        raise ValueError("x must be in [0, 1].")
    if a <= 0.0 or b <= 0.0:
        raise ValueError("Beta shape parameters must be positive.")
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0

    log_front = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    front = math.exp(log_front)
    split_point = (a + 1.0) / (a + b + 2.0)
    if x < split_point:
        value = front * _beta_continued_fraction(a, b, x) / a
    else:
        value = 1.0 - (front * _beta_continued_fraction(b, a, 1.0 - x) / b)
    return min(max(value, 0.0), 1.0)


def normalize_answer(text: str) -> str:
    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def remove_punctuation(value: str) -> str:
        exclude = set(string.punctuation)
        return "".join(character for character in value if character not in exclude)

    return " ".join(remove_articles(remove_punctuation(text.lower())).split())


def is_abstention(text: str) -> bool:
    normalized = normalize_answer(text)
    return normalized == normalize_answer(CANONICAL_ABSTENTION) or any(
        normalize_answer(marker) in normalized for marker in ABSTENTION_MARKERS
    )


def sub_exact_match(
    prediction: str,
    golden_answers: Sequence[str],
) -> bool:
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]

    normalized_prediction = normalize_answer(prediction)
    return any(
        normalize_answer(answer) in normalized_prediction for answer in golden_answers
    )


def extract_completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion.strip()
    if isinstance(completion, dict):
        return str(completion.get("content", "")).strip()
    if isinstance(completion, list) and completion:
        final_item = completion[-1]
        if isinstance(final_item, dict):
            return str(final_item.get("content", "")).strip()
        return str(final_item).strip()
    return str(completion).strip()


class ConfidenceCalibratedGroupReward:
    """Dynamic abstention and centered Brier confidence reward."""

    __name__ = "confidence_calibrated_group_reward"

    def __init__(
        self,
        tokenizer: Any,
        max_prompt_length: int,
        max_completion_length: int,
        group_size: int,
        confidence_weight: float = 0.1,
        group_reward_scale: float = 0.5,
        group_prior_alpha: float = 1.0,
        group_prior_beta: float = 1.0,
        confidence_temperature: float = 1.0,
        confidence_bias: float = 0.0,
        group_confidence_mode: str = "bayesian_tail",
        answerability_threshold: float = 0.5,
        debug: bool = False,
    ) -> None:
        if group_size < 2:
            raise ValueError("group_size must be at least 2.")
        if confidence_weight < 0:
            raise ValueError("confidence_weight must be non-negative.")
        if not 0 <= group_reward_scale <= 1:
            raise ValueError("group_reward_scale must be in [0, 1].")
        if group_prior_alpha <= 0 or group_prior_beta <= 0:
            raise ValueError("Beta-prior parameters must be positive.")
        if confidence_temperature <= 0:
            raise ValueError("confidence_temperature must be positive.")
        if not 0.0 < answerability_threshold < 1.0:
            raise ValueError("answerability_threshold must be in (0, 1).")
        if group_confidence_mode not in {
            "bayesian_tail",
            "empirical",
            "calibrated_mean",
        }:
            raise ValueError(
                "group_confidence_mode must be 'bayesian_tail', "
                "'empirical', or 'calibrated_mean'."
            )

        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.max_completion_length = max_completion_length
        self.group_size = group_size
        self.confidence_weight = confidence_weight
        self.group_reward_scale = group_reward_scale
        self.group_prior_alpha = group_prior_alpha
        self.group_prior_beta = group_prior_beta
        self.confidence_temperature = confidence_temperature
        self.confidence_bias = confidence_bias
        self.group_confidence_mode = group_confidence_mode
        self.answerability_threshold = answerability_threshold
        self.debug = debug
        self.trainer: Optional[GRPOTrainer] = None
        self.__name__ = "confidence_calibrated_group_reward"

    def bind_trainer(self, trainer: GRPOTrainer) -> None:
        self.trainer = trainer

    def set_calibration(
        self,
        temperature: float,
        bias: float,
    ) -> None:
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("Calibration temperature must be positive and finite.")
        if not math.isfinite(bias):
            raise ValueError("Calibration bias must be finite.")
        self.confidence_temperature = float(temperature)
        self.confidence_bias = float(bias)

    def _get_model(self) -> Any:
        if self.trainer is None or getattr(self.trainer, "model", None) is None:
            raise RuntimeError("Reward function is not bound to a trainer model.")
        return self.trainer.model

    @torch.no_grad()
    def _get_sequence_confidence(
        self,
        model: Any,
        prompt: str,
        response: str,
    ) -> float:
        prompt_ids = self.tokenizer(
            prompt,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_prompt_length,
        )["input_ids"]
        response_ids = self.tokenizer(
            response,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_completion_length,
        )["input_ids"]

        if not prompt_ids or not response_ids:
            return 0.0

        device = next(model.parameters()).device
        input_ids = torch.tensor(
            [prompt_ids + response_ids],
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.ones_like(input_ids)
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        # Qwen2.5 and recent Transformers releases can project only the
        # response positions to the vocabulary. Fall back for older versions.
        used_logits_to_keep = False
        with torch.inference_mode():
            try:
                outputs = model(
                    **model_inputs,
                    use_cache=False,
                    logits_to_keep=len(response_ids) + 1,
                )
                used_logits_to_keep = outputs.logits.shape[1] == len(response_ids) + 1
            except (TypeError, ValueError):
                outputs = model(
                    **model_inputs,
                    use_cache=False,
                )

        prompt_length = len(prompt_ids)
        if used_logits_to_keep:
            response_logits = outputs.logits[0, :-1, :].float()
        else:
            response_logits = outputs.logits[
                0,
                prompt_length - 1 : input_ids.shape[1] - 1,
                :,
            ].float()

        response_targets = input_ids[0, prompt_length:]
        if response_logits.shape[0] != response_targets.shape[0]:
            raise RuntimeError(
                "Response logits and targets have inconsistent "
                f"lengths: {response_logits.shape[0]} vs. "
                f"{response_targets.shape[0]}."
            )

        log_probs = torch.log_softmax(
            response_logits,
            dim=-1,
        )
        token_log_probs = log_probs.gather(
            dim=-1,
            index=response_targets.unsqueeze(-1),
        ).squeeze(-1)
        # c(x,y) = (1/T) * sum_t P(y_t | x, y_<t)
        return token_log_probs.exp().mean().item()

    def _calibrate_confidence(
        self,
        raw_confidence: float,
    ) -> float:
        """Return raw sequence confidence without temperature scaling."""
        return min(max(float(raw_confidence), 0.0), 1.0)

    def _group_answerability(
        self,
        records: Sequence[Dict[str, Any]],
    ) -> Dict[str, float]:
        answered = [record for record in records if not record["abstained"]]
        num_correct = sum(record["is_correct"] for record in answered)
        num_incorrect = len(answered) - num_correct
        posterior_alpha = num_correct + self.group_prior_alpha
        posterior_beta = num_incorrect + self.group_prior_beta
        posterior_mean = posterior_alpha / (posterior_alpha + posterior_beta)

        if self.group_confidence_mode == "calibrated_mean":
            if answered:
                answerability = sum(record["confidence"] for record in answered) / len(
                    answered
                )
            else:
                answerability = posterior_mean
            abstention_probability = 1.0 - answerability
        elif self.group_confidence_mode == "empirical":
            answerability = posterior_mean
            abstention_probability = 1.0 - answerability
        else:
            answerability = posterior_mean
            abstention_probability = regularized_incomplete_beta(
                self.answerability_threshold,
                posterior_alpha,
                posterior_beta,
            )

        return {
            "num_answered": float(len(answered)),
            "num_correct": float(num_correct),
            "num_incorrect": float(num_incorrect),
            "posterior_alpha": float(posterior_alpha),
            "posterior_beta": float(posterior_beta),
            "posterior_mean": float(posterior_mean),
            "answerability": float(answerability),
            "abstention_probability": float(abstention_probability),
        }

    def _score_group(
        self,
        records: Sequence[Dict[str, Any]],
    ) -> List[float]:
        group_stats = self._group_answerability(records)
        abstention_probability = group_stats["abstention_probability"]
        abstention_reward = self.group_reward_scale * (
            2.0 * abstention_probability - 1.0
        )
        rewards: List[float] = []

        for record in records:
            if record["abstained"]:
                reward = abstention_reward
            else:
                y = 1.0 if record["is_correct"] else 0.0
                c = record["raw_confidence"]
                reward = 2.0 * y - 1.0 - self.confidence_weight * (c - y) ** 2
            rewards.append(float(reward))

        if self.debug:
            reward_mean = sum(rewards) / len(rewards)
            reward_variance = sum(
                (reward - reward_mean) ** 2 for reward in rewards
            ) / len(rewards)
            print("\n" + "=" * 100)
            print(
                "Posterior Mean: "
                f"{group_stats['posterior_mean']:.6f} ||| "
                "P(p < tau): "
                f"{abstention_probability:.6f} ||| "
                "tau: "
                f"{self.answerability_threshold:.3f} ||| "
                "Abstention Reward: "
                f"{abstention_reward:.6f} ||| "
                "Reward Std: "
                f"{math.sqrt(reward_variance):.6f}"
            )
            for sample_index, (record, reward) in enumerate(
                zip(records, rewards),
                start=1,
            ):
                if record["abstained"]:
                    outcome = "abstention"
                elif record["is_correct"]:
                    outcome = "correct"
                else:
                    outcome = "hallucination"

                print(
                    f"Sample {sample_index} ||| "
                    f"Outcome: {outcome} ||| "
                    f"Response: {record['response']} ||| "
                    f"Ground Truth: {record['ground_truth']} ||| "
                    f"Raw Confidence: {record['raw_confidence']:.6f} ||| "
                    f"Calibrated Confidence: {record['confidence']:.6f} ||| "
                    f"Reward: {reward:.6f}"
                )
            print(f"Rewards: {rewards}")
            print("=" * 100, flush=True)

        return rewards

    def __call__(
        self,
        prompts: Sequence[str],
        completions: Sequence[Any],
        answers: Sequence[Sequence[str]],
        **_: Any,
    ) -> List[float]:
        if not (len(prompts) == len(completions) == len(answers)):
            raise ValueError(
                "prompts, completions, and answers must have " "equal lengths."
            )
        if len(completions) % self.group_size != 0:
            raise ValueError(
                "Number of completions must be divisible by "
                f"group_size={self.group_size}; received "
                f"{len(completions)}."
            )

        model = self._get_model()
        was_training = model.training
        model.eval()
        all_rewards: List[float] = []

        try:
            for start in range(
                0,
                len(completions),
                self.group_size,
            ):
                end = start + self.group_size
                group_prompts = prompts[start:end]
                group_completions = completions[start:end]
                group_answers = answers[start:end]

                if any(prompt != group_prompts[0] for prompt in group_prompts[1:]):
                    raise ValueError(
                        "Expected contiguous GRPO generations "
                        f"for one prompt in [{start}, {end})."
                    )

                records: List[Dict[str, Any]] = []
                for prompt, completion, golden_answers in zip(
                    group_prompts,
                    group_completions,
                    group_answers,
                ):
                    response = extract_completion_text(completion)
                    abstained = is_abstention(response)
                    correct = (
                        False
                        if abstained
                        else sub_exact_match(
                            response,
                            golden_answers,
                        )
                    )

                    if abstained:
                        raw_confidence = 0.0
                        confidence = 0.0
                    else:
                        raw_confidence = self._get_sequence_confidence(
                            model=model,
                            prompt=prompt,
                            response=response,
                        )
                        # Use raw confidence directly; no temperature scaling.
                        confidence = raw_confidence

                    records.append(
                        {
                            "response": response,
                            "ground_truth": list(golden_answers),
                            "abstained": abstained,
                            "is_correct": correct,
                            "raw_confidence": raw_confidence,
                            "confidence": confidence,
                        }
                    )

                all_rewards.extend(self._score_group(records))
        finally:
            if was_training:
                model.train()

        return all_rewards


def setup_logger(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("confidence_calibrated_grpo")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(
        output_dir / "train.log",
        mode="w",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def token_level_f1(
    prediction: str,
    ground_truths: Sequence[str],
) -> float:
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]

    prediction_tokens = normalize_answer(prediction).split()
    if not prediction_tokens:
        return 0.0

    best_f1 = 0.0
    for ground_truth in ground_truths:
        ground_truth_tokens = normalize_answer(ground_truth).split()
        if not ground_truth_tokens:
            continue

        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            continue

        precision = num_same / len(prediction_tokens)
        recall = num_same / len(ground_truth_tokens)
        f1 = 2.0 * precision * recall / (precision + recall)
        best_f1 = max(best_f1, f1)

    return best_f1


def build_chat_prompt(
    tokenizer: AutoTokenizer,
    question: str,
    documents: Sequence[str],
) -> str:
    documents_block = "\n".join(
        f"document {index}: {document}"
        for index, document in enumerate(documents, start=1)
    )
    user_prompt = f"Question: {question}\n" f"Documents:\n{documents_block}"

    if tokenizer.chat_template is not None:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    return f"System: {SYSTEM_PROMPT}\n" f"User: {user_prompt}\n" "Assistant:"


def load_jsonl_records(
    path: str,
    tokenizer: AutoTokenizer,
    seed: int,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []

    with open(path, "r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            line = line.strip()
            if not line:
                continue

            item = json.loads(line)
            if "question" not in item:
                raise ValueError(f"Missing 'question' at line {line_number}.")

            answers = item.get(
                "answer",
                item.get(
                    "answers",
                    item.get("right_answers", []),
                ),
            )
            if isinstance(answers, str):
                answers = [answers]
            answers = [str(answer) for answer in answers if str(answer).strip()]
            if not answers:
                raise ValueError(f"Missing non-empty answer at line {line_number}.")

            documents = item.get("documents", [])
            if isinstance(documents, str):
                documents = [documents]
            documents = [str(document) for document in documents]

            question = str(item["question"])
            record = {
                "prompt": build_chat_prompt(
                    tokenizer=tokenizer,
                    question=question,
                    documents=documents,
                ),
                "question": question,
                "documents": documents,
                "answers": answers,
            }

            # Retain optional metadata without making static knowledge labels
            # a requirement for training.
            for optional_key in (
                "label",
                "target",
                "quadrant",
                "source",
            ):
                if optional_key in item:
                    record[optional_key] = item[optional_key]

            records.append(record)

    random.Random(seed).shuffle(records)
    return records


def split_train_calibration_eval(
    records: List[Dict[str, Any]],
    calibration_ratio: float,
    eval_ratio: float,
    seed: int,
    dataset_number: int,
) -> Tuple[HFDataset, HFDataset, HFDataset]:
    if dataset_number <= 0:
        raise ValueError("dataset_number must be positive.")
    if not 0.0 <= calibration_ratio < 1.0:
        raise ValueError("calibration_ratio must be in [0, 1).")
    if not 0.0 <= eval_ratio < 1.0:
        raise ValueError("eval_ratio must be in [0, 1).")
    if calibration_ratio + eval_ratio >= 1.0:
        raise ValueError("calibration_ratio + eval_ratio must be smaller than 1.")

    rng = random.Random(seed)
    if len(records) > dataset_number:
        records = rng.sample(records, dataset_number)

    if not records:
        raise ValueError("No valid training records were loaded.")

    dataset = HFDataset.from_list(records)
    if len(dataset) == 1:
        return (
            dataset,
            HFDataset.from_list([]),
            HFDataset.from_list([]),
        )

    calibration_size = (
        max(1, int(len(dataset) * calibration_ratio)) if calibration_ratio > 0 else 0
    )
    eval_size = max(1, int(len(dataset) * eval_ratio)) if eval_ratio > 0 else 0
    if calibration_size + eval_size >= len(dataset):
        raise ValueError(
            "The dataset is too small for the requested calibration/eval "
            "ratios while retaining at least one training sample."
        )

    indices = list(range(len(dataset)))
    rng.shuffle(indices)

    calibration_indices = indices[:calibration_size]
    eval_start = calibration_size
    eval_indices = indices[eval_start : eval_start + eval_size]
    train_indices = indices[eval_start + eval_size :]
    return (
        dataset.select(train_indices),
        dataset.select(calibration_indices),
        dataset.select(eval_indices),
    )


def expected_calibration_error(
    probabilities: Sequence[float],
    labels: Sequence[int],
    num_bins: int = 10,
) -> float:
    if not probabilities:
        return 0.0

    total = len(probabilities)
    error = 0.0
    for bin_index in range(num_bins):
        lower = bin_index / num_bins
        upper = (bin_index + 1) / num_bins
        members = [
            index
            for index, probability in enumerate(probabilities)
            if lower <= probability < upper
            or (bin_index == num_bins - 1 and probability == 1.0)
        ]
        if not members:
            continue

        mean_confidence = sum(probabilities[index] for index in members) / len(members)
        mean_accuracy = sum(labels[index] for index in members) / len(members)
        error += (len(members) / total) * abs(mean_confidence - mean_accuracy)
    return error


def apply_confidence_calibration(
    raw_confidences: Sequence[float],
    temperature: float,
    bias: float,
) -> List[float]:
    calibrated: List[float] = []
    for confidence in raw_confidences:
        bounded = min(max(float(confidence), 1e-6), 1.0 - 1e-6)
        raw_logit = math.log(bounded / (1.0 - bounded))
        calibrated_logit = (raw_logit + bias) / temperature
        calibrated.append(1.0 / (1.0 + math.exp(-calibrated_logit)))
    return calibrated


def fit_confidence_calibrator(
    raw_confidences: Sequence[float],
    labels: Sequence[int],
    initial_temperature: float,
    initial_bias: float,
) -> Dict[str, Any]:
    if len(raw_confidences) != len(labels):
        raise ValueError("Calibration confidences and labels must have equal lengths.")
    if len(raw_confidences) < 2 or len(set(labels)) < 2:
        calibrated = apply_confidence_calibration(
            raw_confidences,
            initial_temperature,
            initial_bias,
        )
        brier = sum(
            (probability - label) ** 2 for probability, label in zip(calibrated, labels)
        ) / max(len(labels), 1)
        ece = expected_calibration_error(calibrated, labels)
        return {
            "fitted": False,
            "reason": "Calibration requires at least one correct and one incorrect answer.",
            "temperature": float(initial_temperature),
            "bias": float(initial_bias),
            "num_examples": len(raw_confidences),
            "num_correct": int(sum(labels)),
            "num_incorrect": int(len(labels) - sum(labels)),
            "brier_before": brier,
            "brier_after": brier,
            "ece_before": ece,
            "ece_after": ece,
        }

    raw_tensor = torch.tensor(
        [
            math.log(
                min(max(float(value), 1e-6), 1.0 - 1e-6)
                / (1.0 - min(max(float(value), 1e-6), 1.0 - 1e-6))
            )
            for value in raw_confidences
        ],
        dtype=torch.float64,
    )
    label_tensor = torch.tensor(labels, dtype=torch.float64)
    log_temperature = torch.nn.Parameter(
        torch.tensor(
            math.log(initial_temperature),
            dtype=torch.float64,
        )
    )
    bias_parameter = torch.nn.Parameter(torch.tensor(initial_bias, dtype=torch.float64))
    optimizer = torch.optim.LBFGS(
        [log_temperature, bias_parameter],
        lr=0.25,
        max_iter=100,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = torch.exp(log_temperature).clamp(0.05, 20.0)
        bias = bias_parameter.clamp(-20.0, 20.0)
        probabilities = torch.sigmoid((raw_tensor + bias) / temperature)
        loss = torch.mean((probabilities - label_tensor) ** 2)
        loss.backward()
        return loss

    optimizer.step(closure)
    fitted_temperature = float(
        torch.exp(log_temperature).clamp(0.05, 20.0).detach().item()
    )
    fitted_bias = float(bias_parameter.clamp(-20.0, 20.0).detach().item())

    before = apply_confidence_calibration(
        raw_confidences,
        initial_temperature,
        initial_bias,
    )
    after = apply_confidence_calibration(
        raw_confidences,
        fitted_temperature,
        fitted_bias,
    )
    brier_before = sum(
        (probability - label) ** 2 for probability, label in zip(before, labels)
    ) / len(labels)
    brier_after = sum(
        (probability - label) ** 2 for probability, label in zip(after, labels)
    ) / len(labels)
    if not math.isfinite(brier_after) or brier_after > brier_before:
        return {
            "fitted": False,
            "reason": "Fitted calibration did not improve Brier score.",
            "temperature": float(initial_temperature),
            "bias": float(initial_bias),
            "num_examples": len(labels),
            "num_correct": int(sum(labels)),
            "num_incorrect": int(len(labels) - sum(labels)),
            "brier_before": brier_before,
            "brier_after": brier_before,
            "ece_before": expected_calibration_error(before, labels),
            "ece_after": expected_calibration_error(before, labels),
        }

    return {
        "fitted": True,
        "temperature": fitted_temperature,
        "bias": fitted_bias,
        "num_examples": len(labels),
        "num_correct": int(sum(labels)),
        "num_incorrect": int(len(labels) - sum(labels)),
        "brier_before": brier_before,
        "brier_after": brier_after,
        "ece_before": expected_calibration_error(before, labels),
        "ece_after": expected_calibration_error(after, labels),
    }


@torch.no_grad()
def collect_calibration_rollouts(
    model: Any,
    tokenizer: AutoTokenizer,
    calibration_dataset: HFDataset,
    num_samples_per_prompt: int,
    max_samples: int,
    max_prompt_length: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> Tuple[List[float], List[int], Dict[str, int]]:
    if num_samples_per_prompt <= 0:
        raise ValueError("num_samples_per_prompt must be positive.")
    if max_samples <= 0:
        raise ValueError("max_samples must be positive.")

    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    raw_confidences: List[float] = []
    labels: List[int] = []
    num_abstained = 0
    num_generated = 0

    selected_size = min(len(calibration_dataset), max_samples)
    try:
        for item_index in range(selected_size):
            item = calibration_dataset[item_index]
            prompt = item["prompt"]
            answers = item["answers"]
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_prompt_length,
            ).to(device)
            prompt_width = inputs["input_ids"].shape[1]
            generation = model.generate(
                **inputs,
                do_sample=True,
                num_return_sequences=num_samples_per_prompt,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )
            if not generation.scores:
                continue

            score_model = model
            if not hasattr(score_model, "compute_transition_scores"):
                if hasattr(score_model, "get_base_model"):
                    score_model = score_model.get_base_model()
                else:
                    raise AttributeError(
                        "The model does not provide compute_transition_scores."
                    )
            transition_scores = score_model.compute_transition_scores(
                generation.sequences,
                generation.scores,
                normalize_logits=True,
            )
            generated_ids = generation.sequences[:, prompt_width:]
            score_width = transition_scores.shape[1]
            generated_ids = generated_ids[:, :score_width]

            for sequence_index in range(generated_ids.shape[0]):
                response_ids = generated_ids[sequence_index]
                response = tokenizer.decode(
                    response_ids,
                    skip_special_tokens=True,
                ).strip()
                num_generated += 1
                if is_abstention(response):
                    num_abstained += 1
                    continue

                valid_mask = torch.ones_like(
                    response_ids,
                    dtype=torch.bool,
                )
                if tokenizer.pad_token_id is not None:
                    valid_mask &= response_ids.ne(tokenizer.pad_token_id)
                if tokenizer.eos_token_id is not None:
                    valid_mask &= response_ids.ne(tokenizer.eos_token_id)
                valid_scores = transition_scores[sequence_index][valid_mask]
                if valid_scores.numel() == 0:
                    continue

                # Use the same arithmetic-mean confidence as reward scoring.
                raw_confidence = valid_scores.float().exp().mean().item()
                raw_confidences.append(raw_confidence)
                labels.append(int(sub_exact_match(response, answers)))
    finally:
        if was_training:
            model.train()

    stats = {
        "num_prompts": selected_size,
        "num_generated": num_generated,
        "num_abstained": num_abstained,
        "num_answered": len(labels),
    }
    return raw_confidences, labels, stats


def select_torch_dtype() -> Optional[torch.dtype]:
    if not torch.cuda.is_available():
        return None
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def build_quantization_config(
    args: argparse.Namespace,
) -> Optional[BitsAndBytesConfig]:
    if not args.load_in_4bit:
        return None

    compute_dtype = select_torch_dtype() or torch.float16
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def build_model_init_kwargs(
    args: argparse.Namespace,
) -> Dict[str, Any]:
    model_kwargs: Dict[str, Any] = {
        "low_cpu_mem_usage": True,
        "use_cache": not args.gradient_checkpointing,
    }
    dtype = select_torch_dtype()
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype

    quantization_config = build_quantization_config(args)
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config

    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation

    return model_kwargs


def build_peft_config(
    args: argparse.Namespace,
) -> Optional[LoraConfig]:
    if not args.use_peft:
        return None
    if LoraConfig is None or TaskType is None:
        raise ImportError("PEFT is required when --use_peft is enabled.")

    target_modules = [
        module.strip()
        for module in args.lora_target_modules.split(",")
        if module.strip()
    ]
    if not target_modules:
        target_modules = DEFAULT_LORA_TARGET_MODULES

    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )


def infer_world_size() -> int:
    return max(int(os.environ.get("WORLD_SIZE", "1")), 1)


def validate_grpo_batching(
    batch_size: int,
    gradient_accumulation_steps: int,
    group_size: int,
) -> None:
    world_size = infer_world_size()
    global_batch_size = batch_size * gradient_accumulation_steps * world_size
    if global_batch_size % group_size != 0:
        raise ValueError(
            "Effective global batch size must be divisible by group_size. "
            f"Received global_batch_size={global_batch_size}, "
            f"group_size={group_size}."
        )


@torch.no_grad()
def evaluate_model(
    model: Any,
    tokenizer: AutoTokenizer,
    eval_dataset: HFDataset,
    batch_size: int,
    max_prompt_length: int,
    max_new_tokens: int,
    confidence_scorer: Optional[ConfidenceCalibratedGroupReward] = None,
) -> Dict[str, float]:
    if len(eval_dataset) == 0:
        return {
            "eval_accuracy": 0.0,
            "eval_precision": 0.0,
            "eval_abstention_rate": 0.0,
            "eval_hallucination_rate": 0.0,
            "eval_truthfulness": 0.0,
            "eval_answer_f1": 0.0,
            "eval_confidence_brier": 0.0,
            "eval_confidence_ece": 0.0,
            "eval_correct_confidence": 0.0,
            "eval_hallucination_confidence": 0.0,
        }

    model.eval()
    device = next(model.parameters()).device
    total = 0
    num_correct = 0
    num_abstained = 0
    num_hallucinated = 0
    answer_f1_scores: List[float] = []
    answer_confidences: List[float] = []
    confidence_labels: List[int] = []
    correct_confidences: List[float] = []
    hallucination_confidences: List[float] = []

    for start in range(0, len(eval_dataset), batch_size):
        batch = eval_dataset[start : start + batch_size]
        prompts = batch["prompt"]
        answers_batch = batch["answers"]
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_prompt_length,
        ).to(device)

        prompt_width = inputs["input_ids"].shape[1]
        generated_ids = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        for index, answers in enumerate(answers_batch):
            response_ids = generated_ids[index, prompt_width:]
            response = tokenizer.decode(
                response_ids,
                skip_special_tokens=True,
            ).strip()

            total += 1
            if is_abstention(response):
                num_abstained += 1
                continue

            answer_f1_scores.append(token_level_f1(response, answers))
            is_correct = sub_exact_match(response, answers)
            if confidence_scorer is not None:
                raw_confidence = confidence_scorer._get_sequence_confidence(
                    model=model,
                    prompt=prompts[index],
                    response=response,
                )
                calibrated_confidence = confidence_scorer._calibrate_confidence(
                    raw_confidence
                )
                answer_confidences.append(calibrated_confidence)
                confidence_labels.append(int(is_correct))
                if is_correct:
                    correct_confidences.append(calibrated_confidence)
                else:
                    hallucination_confidences.append(calibrated_confidence)

            if is_correct:
                num_correct += 1
            else:
                num_hallucinated += 1

    accuracy = num_correct / max(total, 1)
    hallucination_rate = num_hallucinated / max(total, 1)
    abstention_rate = num_abstained / max(total, 1)
    precision = num_correct / max(
        num_correct + num_hallucinated,
        1,
    )
    truthfulness = accuracy - hallucination_rate
    confidence_brier = sum(
        (confidence - label) ** 2
        for confidence, label in zip(
            answer_confidences,
            confidence_labels,
        )
    ) / max(len(confidence_labels), 1)

    return {
        "eval_accuracy": accuracy,
        "eval_precision": precision,
        "eval_abstention_rate": abstention_rate,
        "eval_hallucination_rate": hallucination_rate,
        "eval_truthfulness": truthfulness,
        "eval_answer_f1": sum(answer_f1_scores) / max(len(answer_f1_scores), 1),
        "eval_confidence_brier": confidence_brier,
        "eval_confidence_ece": expected_calibration_error(
            answer_confidences,
            confidence_labels,
        ),
        "eval_correct_confidence": sum(correct_confidences)
        / max(len(correct_confidences), 1),
        "eval_hallucination_confidence": sum(hallucination_confidences)
        / max(len(hallucination_confidences), 1),
    }


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as output_file:
        json.dump(
            payload,
            output_file,
            ensure_ascii=False,
            indent=2,
            default=str,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Confidence-calibrated GRPO for document-grounded "
            "selective question answering."
        )
    )
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--dataset_number", type=int, required=True)
    parser.add_argument("--learning_rate", type=float, required=True)
    parser.add_argument("--num_epochs", type=float, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--eval_batch_size", type=int, required=True)
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        required=True,
    )
    parser.add_argument("--group_size", type=int, required=True)
    parser.add_argument("--max_prompt_length", type=int, required=True)
    parser.add_argument("--max_new_tokens", type=int, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--top_p", type=float, required=True)
    parser.add_argument("--clip_eps", type=float, required=True)
    parser.add_argument("--kl_coef", type=float, required=True)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--save_total_limit", type=int, default=20)
    parser.add_argument("--eval_ratio", type=float, default=0.1)
    parser.add_argument(
        "--calibration_ratio",
        type=float,
        default=DEFAULT_CALIBRATION_RATIO,
    )
    parser.add_argument(
        "--calibration_num_samples",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--calibration_max_samples",
        type=int,
        default=200,
    )
    parser.add_argument("--attn_implementation", default=None)

    parser.add_argument(
        "--gradient_checkpointing",
        dest="gradient_checkpointing",
        action="store_true",
    )
    parser.add_argument(
        "--no_gradient_checkpointing",
        dest="gradient_checkpointing",
        action="store_false",
    )
    parser.set_defaults(gradient_checkpointing=False)

    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--use_peft", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument(
        "--lora_target_modules",
        default=",".join(DEFAULT_LORA_TARGET_MODULES),
    )

    parser.add_argument(
        "--confidence_weight",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--group_reward_scale",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--group_prior_alpha",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--group_prior_beta",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--confidence_temperature",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--confidence_bias",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--group_confidence_mode",
        choices=[
            "bayesian_tail",
            "empirical",
            "calibrated_mean",
        ],
        default="bayesian_tail",
    )
    parser.add_argument(
        "--answerability_threshold",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--fit_confidence_calibration",
        dest="fit_confidence_calibration",
        action="store_true",
    )
    parser.add_argument(
        "--no_fit_confidence_calibration",
        dest="fit_confidence_calibration",
        action="store_false",
    )
    parser.set_defaults(fit_confidence_calibration=True)
    parser.add_argument("--reward_debug", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--script_path", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # This standalone variant does not fit or apply temperature scaling.
    args.fit_confidence_calibration = False
    args.calibration_ratio = 0.0
    args.confidence_temperature = 1.0
    args.confidence_bias = 0.0

    set_seed(args.seed)

    if not Path(args.train_file).is_file():
        raise FileNotFoundError(f"Training file does not exist: {args.train_file}")
    if args.load_in_4bit and not args.use_peft:
        raise ValueError("--load_in_4bit must be used together with --use_peft.")
    if args.group_size < 2:
        raise ValueError("--group_size must be at least 2.")

    validate_grpo_batching(
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        group_size=args.group_size,
    )

    timestamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    model_name = Path(args.model_name_or_path).name
    dataset_name = Path(args.train_file).parent.name
    output_dir = Path(args.output_root) / model_name / dataset_name / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)
    logging_dir = output_dir / "logs"
    logging_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output_dir)

    if args.script_path:
        source_script = Path(args.script_path)
        if source_script.is_file():
            shutil.copy2(source_script, output_dir / source_script.name)
        else:
            logger.warning(
                "Bash script was not copied because it does not exist: %s",
                source_script,
            )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    records = load_jsonl_records(
        path=args.train_file,
        tokenizer=tokenizer,
        seed=args.seed,
    )
    train_dataset, calibration_dataset, eval_dataset = split_train_calibration_eval(
        records=records,
        calibration_ratio=args.calibration_ratio,
        eval_ratio=args.eval_ratio,
        seed=args.seed,
        dataset_number=args.dataset_number,
    )
    logger.info(
        "Loaded %d training, %d calibration, and %d evaluation samples.",
        len(train_dataset),
        len(calibration_dataset),
        len(eval_dataset),
    )

    model_init_kwargs = build_model_init_kwargs(args)
    peft_config = build_peft_config(args)
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    training_args = GRPOConfig(
        output_dir=str(output_dir),
        logging_dir=str(logging_dir),
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.group_size,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        beta=args.kl_coef,
        epsilon=args.clip_eps,
        logging_steps=args.logging_steps,
        eval_strategy="epoch" if len(eval_dataset) else "no",
        save_strategy="epoch",
        save_total_limit=args.save_total_limit,
        remove_unused_columns=False,
        gradient_checkpointing=args.gradient_checkpointing,
        bf16=use_bf16,
        fp16=torch.cuda.is_available() and not use_bf16,
        report_to="tensorboard",
        seed=args.seed,
        model_init_kwargs=model_init_kwargs,
        use_vllm=False,
    )

    reward_fn = ConfidenceCalibratedGroupReward(
        tokenizer=tokenizer,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_new_tokens,
        group_size=args.group_size,
        confidence_weight=args.confidence_weight,
        group_reward_scale=args.group_reward_scale,
        group_prior_alpha=args.group_prior_alpha,
        group_prior_beta=args.group_prior_beta,
        confidence_temperature=args.confidence_temperature,
        confidence_bias=args.confidence_bias,
        group_confidence_mode=args.group_confidence_mode,
        answerability_threshold=args.answerability_threshold,
        debug=args.reward_debug,
    )

    trainer = GRPOTrainer(
        model=args.model_name_or_path,
        reward_funcs=reward_fn,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if len(eval_dataset) else None,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    if args.gradient_checkpointing:
        # Gradient checkpointing and KV caching are mutually exclusive during
        # training. Set this again after GRPO/PEFT wrapping because some model
        # wrappers may restore the pretrained configuration.
        trainer.model.config.use_cache = False
    reward_fn.bind_trainer(trainer)

    calibration_metrics: Dict[str, Any] = {
        "fitted": False,
        "reason": "Calibration fitting was disabled or no calibration split exists.",
        "temperature": args.confidence_temperature,
        "bias": args.confidence_bias,
    }
    if args.fit_confidence_calibration and len(calibration_dataset):
        calibration_model = trainer.accelerator.unwrap_model(trainer.model)
        raw_confidences, calibration_labels, rollout_stats = (
            collect_calibration_rollouts(
                model=calibration_model,
                tokenizer=tokenizer,
                calibration_dataset=calibration_dataset,
                num_samples_per_prompt=args.calibration_num_samples,
                max_samples=args.calibration_max_samples,
                max_prompt_length=args.max_prompt_length,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
        )
        calibration_metrics = fit_confidence_calibrator(
            raw_confidences=raw_confidences,
            labels=calibration_labels,
            initial_temperature=args.confidence_temperature,
            initial_bias=args.confidence_bias,
        )
        calibration_metrics.update(rollout_stats)
        reward_fn.set_calibration(
            temperature=calibration_metrics["temperature"],
            bias=calibration_metrics["bias"],
        )
        logger.info(
            "Confidence calibration:\n%s",
            json.dumps(calibration_metrics, indent=2),
        )

    run_config = vars(args).copy()
    run_config.update(
        {
            "train_size": len(train_dataset),
            "calibration_size": len(calibration_dataset),
            "eval_size": len(eval_dataset),
            "world_size": infer_world_size(),
            "output_dir": str(output_dir),
            "fitted_confidence_temperature": (reward_fn.confidence_temperature),
            "fitted_confidence_bias": reward_fn.confidence_bias,
        }
    )
    save_json(output_dir / "run_config.json", run_config)
    save_json(
        output_dir / "confidence_calibration_metrics.json",
        calibration_metrics,
    )
    logger.info(
        "Run configuration:\n%s",
        json.dumps(run_config, indent=2, default=str),
    )

    train_result = trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    trainer.accelerator.wait_for_everyone()
    if trainer.is_world_process_zero():
        save_json(
            output_dir / "train_metrics.json",
            dict(train_result.metrics),
        )

        if len(eval_dataset):
            eval_model = trainer.accelerator.unwrap_model(trainer.model)
            eval_metrics = evaluate_model(
                model=eval_model,
                tokenizer=tokenizer,
                eval_dataset=eval_dataset,
                batch_size=args.eval_batch_size,
                max_prompt_length=args.max_prompt_length,
                max_new_tokens=args.max_new_tokens,
                confidence_scorer=reward_fn,
            )
            save_json(
                output_dir / "eval_abstention_metrics.json",
                eval_metrics,
            )
            logger.info(
                "Evaluation metrics:\n%s",
                json.dumps(eval_metrics, indent=2),
            )


if __name__ == "__main__":
    main()
