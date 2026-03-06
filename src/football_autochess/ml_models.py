from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence


FeatureVector = dict[str, float]
TrainingExample = tuple[FeatureVector, int]


def _sigmoid(value: float) -> float:
    if value >= 0:
        exp_neg = math.exp(-value)
        return 1.0 / (1.0 + exp_neg)
    exp_pos = math.exp(value)
    return exp_pos / (1.0 + exp_pos)


def _clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


@dataclass(frozen=True)
class LogisticTrainingConfig:
    learning_rate: float = 0.04
    epochs: int = 8
    l2: float = 0.0001
    weight_clip: float = 12.0
    seed: int | None = None


@dataclass(frozen=True)
class SparseLogisticModel:
    name: str
    bias: float
    weights: dict[str, float]

    def predict_proba(self, features: Mapping[str, float]) -> float:
        z = self.bias
        for key, value in features.items():
            if value == 0.0:
                continue
            z += self.weights.get(key, 0.0) * value
        return _clamp(_sigmoid(z), 0.001, 0.999)

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "bias": self.bias,
            "weights": self.weights,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> SparseLogisticModel:
        raw_weights = payload.get("weights", {})
        weights: dict[str, float] = {}
        if isinstance(raw_weights, Mapping):
            for key, value in raw_weights.items():
                weights[str(key)] = float(value)

        return cls(
            name=str(payload.get("name", "logistic")),
            bias=float(payload.get("bias", 0.0)),
            weights=weights,
        )

    @classmethod
    def train(
        cls,
        name: str,
        examples: Sequence[TrainingExample],
        config: LogisticTrainingConfig | None = None,
    ) -> SparseLogisticModel:
        if not examples:
            raise ValueError(f"Cannot train {name}: no examples provided")

        train_config = config or LogisticTrainingConfig()
        rng = random.Random(train_config.seed)

        bias = 0.0
        weights: dict[str, float] = {}

        indices = list(range(len(examples)))

        for _ in range(train_config.epochs):
            rng.shuffle(indices)
            for idx in indices:
                features, label = examples[idx]
                z = bias
                for key, value in features.items():
                    if value == 0.0:
                        continue
                    z += weights.get(key, 0.0) * value

                prob = _sigmoid(z)
                error = float(label) - prob

                bias += train_config.learning_rate * error
                bias = _clamp(bias, -train_config.weight_clip, train_config.weight_clip)

                for key, value in features.items():
                    if value == 0.0:
                        continue
                    current = weights.get(key, 0.0)
                    gradient = (error * value) - (train_config.l2 * current)
                    updated = current + train_config.learning_rate * gradient
                    weights[key] = _clamp(updated, -train_config.weight_clip, train_config.weight_clip)

        return cls(name=name, bias=bias, weights=weights)


@dataclass(frozen=True)
class ModelBundle:
    event_models: dict[str, SparseLogisticModel] = field(default_factory=dict)
    playcall_model: SparseLogisticModel | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def predict_event_probability(self, event: str, features: Mapping[str, float], fallback: float) -> float:
        model = self.event_models.get(event)
        if model is None:
            return _clamp(fallback, 0.001, 0.999)
        return model.predict_proba(features)

    def predict_pass_call_probability(self, features: Mapping[str, float], fallback: float) -> float:
        if self.playcall_model is None:
            return _clamp(fallback, 0.001, 0.999)
        return self.playcall_model.predict_proba(features)

    def to_dict(self) -> dict[str, object]:
        return {
            "metadata": self.metadata,
            "event_models": {name: model.to_dict() for name, model in self.event_models.items()},
            "playcall_model": self.playcall_model.to_dict() if self.playcall_model else None,
        }

    def save_json(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> ModelBundle:
        raw_event_models = payload.get("event_models", {})
        event_models: dict[str, SparseLogisticModel] = {}
        if isinstance(raw_event_models, Mapping):
            for key, raw_model in raw_event_models.items():
                if isinstance(raw_model, Mapping):
                    event_models[str(key)] = SparseLogisticModel.from_dict(raw_model)

        playcall_model = None
        raw_playcall = payload.get("playcall_model")
        if isinstance(raw_playcall, Mapping):
            playcall_model = SparseLogisticModel.from_dict(raw_playcall)

        metadata: dict[str, object] = {}
        raw_metadata = payload.get("metadata", {})
        if isinstance(raw_metadata, Mapping):
            metadata = dict(raw_metadata)

        return cls(event_models=event_models, playcall_model=playcall_model, metadata=metadata)

    @classmethod
    def load_json(cls, path: str | Path) -> ModelBundle:
        source = Path(path)
        payload = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError(f"Model bundle payload must be an object: {source}")
        return cls.from_dict(payload)


def basic_classification_metrics(examples: Iterable[TrainingExample], model: SparseLogisticModel) -> dict[str, float]:
    total = 0
    positives = 0
    correct = 0
    logloss_sum = 0.0

    for features, label in examples:
        total += 1
        positives += int(label == 1)
        prob = model.predict_proba(features)
        pred = 1 if prob >= 0.5 else 0
        if pred == label:
            correct += 1

        if label == 1:
            logloss_sum -= math.log(prob)
        else:
            logloss_sum -= math.log(1.0 - prob)

    if total == 0:
        return {
            "count": 0.0,
            "positive_rate": 0.0,
            "accuracy": 0.0,
            "logloss": 0.0,
        }

    return {
        "count": float(total),
        "positive_rate": positives / total,
        "accuracy": correct / total,
        "logloss": logloss_sum / total,
    }
