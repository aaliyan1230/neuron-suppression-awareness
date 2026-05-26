from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .errors import ConfigError, MissingDependencyError


SUPPORTED_BACKENDS = {"transformers", "vllm_lens"}


@dataclass(frozen=True)
class QuantizationConfig:
    load_in_4bit: bool = False
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True


@dataclass(frozen=True)
class ModelConfig:
    id: str
    revision: str
    dtype: str
    trust_remote_code: bool = True
    quantization: QuantizationConfig | None = None


@dataclass(frozen=True)
class Phase0Settings:
    layer: int
    neuron: int
    pin_value: float
    activation_window_tokens: int
    score_token_text: str
    score_token_match: str
    aggregation: str
    fallback_aggregation: str


@dataclass(frozen=True)
class SuppressionSettings:
    layer: int
    neuron: int
    pin_value: float


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int
    do_sample: bool
    temperature: float


@dataclass(frozen=True)
class TextDatasetConfig:
    id: str
    split: str
    limit: int
    text_fields: tuple[str, ...]
    name: str | None = None
    input_field: str | None = None
    requires_hf_token: bool = False


@dataclass(frozen=True)
class DatasetConfig:
    harmful: TextDatasetConfig
    harmless: TextDatasetConfig


@dataclass(frozen=True)
class ExpectedActivationConfig:
    harmful_mean_reference: float
    harmless_mean_reference: float
    harmful_mean_range: tuple[float, float]
    harmless_mean_range: tuple[float, float]
    min_abs_gap: float


@dataclass(frozen=True)
class OutputConfig:
    root: Path
    run_name: str | None = None


@dataclass(frozen=True)
class BackendConfig:
    name: str
    transformers: dict[str, Any]
    vllm_lens: dict[str, Any]


@dataclass(frozen=True)
class Phase0Config:
    model: ModelConfig
    phase0: Phase0Settings
    generation: GenerationConfig
    datasets: DatasetConfig
    expected_activations: ExpectedActivationConfig
    outputs: OutputConfig
    backend: BackendConfig
    source_path: Path | None = None

    def with_backend(self, backend_name: str) -> "Phase0Config":
        return replace(self, backend=replace(self.backend, name=backend_name))


@dataclass(frozen=True)
class JudgeModelConfig:
    id: str
    revision: str
    dtype: str
    trust_remote_code: bool = True
    quantization: QuantizationConfig | None = None


@dataclass(frozen=True)
class JudgeConfig:
    model: JudgeModelConfig
    max_new_tokens: int


@dataclass(frozen=True)
class PassCriteria:
    min_suppressed_asr: float
    max_clean_asr: float


@dataclass(frozen=True)
class Phase1Config:
    model: ModelConfig
    suppression: SuppressionSettings
    generation: GenerationConfig
    dataset: TextDatasetConfig
    judge: JudgeConfig
    pass_criteria: PassCriteria
    checkpoint: bool
    outputs: OutputConfig
    backend: BackendConfig
    source_path: Path | None = None

    def with_backend(self, backend_name: str) -> "Phase1Config":
        return replace(self, backend=replace(self.backend, name=backend_name))


ExperimentConfig = Phase0Config | Phase1Config


def load_config(path: str | Path, backend_override: str | None = None) -> ExperimentConfig:
    raw = _load_yaml(Path(path))
    config = parse_config(raw, source_path=Path(path))
    if backend_override is not None:
        config = config.with_backend(backend_override)
    validate_config(config)
    return config


def parse_config(raw: dict[str, Any], source_path: Path | None = None) -> ExperimentConfig:
    phase = int(raw.get("phase", 0))
    if phase == 0:
        return parse_phase0_config(raw, source_path=source_path)
    if phase == 1:
        return parse_phase1_config(raw, source_path=source_path)
    raise ConfigError(f"Unsupported phase {phase}. Expected 0 or 1.")


def parse_phase0_config(
    raw: dict[str, Any], source_path: Path | None = None
) -> Phase0Config:
    model_raw = _mapping(raw, "model")
    phase_raw = _mapping(raw, "phase0")
    gen_raw = _mapping(raw, "generation")
    datasets_raw = _mapping(raw, "datasets")
    expected_raw = _mapping(raw, "expected_activations")
    outputs_raw = _mapping(raw, "outputs")
    backend_raw = _mapping(raw, "backend")

    return Phase0Config(
        model=_parse_model_config(model_raw, "model"),
        phase0=Phase0Settings(
            layer=int(_required(phase_raw, "layer", "phase0")),
            neuron=int(_required(phase_raw, "neuron", "phase0")),
            pin_value=float(_required(phase_raw, "pin_value", "phase0")),
            activation_window_tokens=int(
                phase_raw.get("activation_window_tokens", 32)
            ),
            score_token_text=str(phase_raw.get("score_token_text", "\n")),
            score_token_match=str(phase_raw.get("score_token_match", "contains")),
            aggregation=str(phase_raw.get("aggregation", "selected_token")),
            fallback_aggregation=str(phase_raw.get("fallback_aggregation", "min")),
        ),
        generation=GenerationConfig(
            max_new_tokens=int(_required(gen_raw, "max_new_tokens", "generation")),
            do_sample=bool(gen_raw.get("do_sample", False)),
            temperature=float(gen_raw.get("temperature", 0.0)),
        ),
        datasets=DatasetConfig(
            harmful=_parse_dataset_config(
                _mapping(datasets_raw, "harmful", parent="datasets"),
                "datasets.harmful",
            ),
            harmless=_parse_dataset_config(
                _mapping(datasets_raw, "harmless", parent="datasets"),
                "datasets.harmless",
            ),
        ),
        expected_activations=ExpectedActivationConfig(
            harmful_mean_reference=float(
                _required(
                    expected_raw,
                    "harmful_mean_reference",
                    "expected_activations",
                )
            ),
            harmless_mean_reference=float(
                _required(
                    expected_raw,
                    "harmless_mean_reference",
                    "expected_activations",
                )
            ),
            harmful_mean_range=_float_pair(
                _required(
                    expected_raw,
                    "harmful_mean_range",
                    "expected_activations",
                ),
                "expected_activations.harmful_mean_range",
            ),
            harmless_mean_range=_float_pair(
                _required(
                    expected_raw,
                    "harmless_mean_range",
                    "expected_activations",
                ),
                "expected_activations.harmless_mean_range",
            ),
            min_abs_gap=float(expected_raw.get("min_abs_gap", 1.0)),
        ),
        outputs=OutputConfig(
            root=Path(str(_required(outputs_raw, "root", "outputs"))),
            run_name=(
                None
                if outputs_raw.get("run_name") is None
                else str(outputs_raw.get("run_name"))
            ),
        ),
        backend=BackendConfig(
            name=str(backend_raw.get("name", "transformers")),
            transformers=dict(backend_raw.get("transformers", {})),
            vllm_lens=dict(backend_raw.get("vllm_lens", {})),
        ),
        source_path=source_path,
    )


def parse_phase1_config(
    raw: dict[str, Any], source_path: Path | None = None
) -> Phase1Config:
    model_raw = _mapping(raw, "model")
    suppression_raw = _mapping(raw, "suppression")
    gen_raw = _mapping(raw, "generation")
    dataset_raw = _mapping(raw, "dataset")
    judge_raw = _mapping(raw, "judge")
    judge_model_raw = _mapping(judge_raw, "model", parent="judge")
    criteria_raw = _mapping(raw, "pass_criteria")
    outputs_raw = _mapping(raw, "outputs")
    backend_raw = _mapping(raw, "backend")

    return Phase1Config(
        model=_parse_model_config(model_raw, "model"),
        suppression=SuppressionSettings(
            layer=int(_required(suppression_raw, "layer", "suppression")),
            neuron=int(_required(suppression_raw, "neuron", "suppression")),
            pin_value=float(_required(suppression_raw, "pin_value", "suppression")),
        ),
        generation=GenerationConfig(
            max_new_tokens=int(_required(gen_raw, "max_new_tokens", "generation")),
            do_sample=bool(gen_raw.get("do_sample", False)),
            temperature=float(gen_raw.get("temperature", 0.0)),
        ),
        dataset=_parse_dataset_config(dataset_raw, "dataset"),
        judge=JudgeConfig(
            model=JudgeModelConfig(
                id=str(_required(judge_model_raw, "id", "judge.model")),
                revision=str(_required(judge_model_raw, "revision", "judge.model")),
                dtype=str(_required(judge_model_raw, "dtype", "judge.model")),
                trust_remote_code=bool(judge_model_raw.get("trust_remote_code", True)),
                quantization=_parse_quantization_config(judge_model_raw),
            ),
            max_new_tokens=int(_required(judge_raw, "max_new_tokens", "judge")),
        ),
        pass_criteria=PassCriteria(
            min_suppressed_asr=float(
                _required(criteria_raw, "min_suppressed_asr", "pass_criteria")
            ),
            max_clean_asr=float(
                _required(criteria_raw, "max_clean_asr", "pass_criteria")
            ),
        ),
        checkpoint=bool(raw.get("checkpoint", True)),
        outputs=OutputConfig(
            root=Path(str(_required(outputs_raw, "root", "outputs"))),
            run_name=(
                None
                if outputs_raw.get("run_name") is None
                else str(outputs_raw.get("run_name"))
            ),
        ),
        backend=BackendConfig(
            name=str(backend_raw.get("name", "transformers")),
            transformers=dict(backend_raw.get("transformers", {})),
            vllm_lens=dict(backend_raw.get("vllm_lens", {})),
        ),
        source_path=source_path,
    )


def validate_config(config: ExperimentConfig) -> None:
    if config.backend.name not in SUPPORTED_BACKENDS:
        raise ConfigError(
            f"Unsupported backend {config.backend.name!r}. "
            f"Expected one of {sorted(SUPPORTED_BACKENDS)}."
        )
    if isinstance(config, Phase1Config):
        _validate_phase1_config(config)
        return
    if config.phase0.layer < 0:
        raise ConfigError("phase0.layer must be non-negative.")
    if config.phase0.neuron < 0:
        raise ConfigError("phase0.neuron must be non-negative.")
    if config.phase0.activation_window_tokens <= 0:
        raise ConfigError("phase0.activation_window_tokens must be positive.")
    if config.phase0.score_token_match not in {"exact", "contains"}:
        raise ConfigError("phase0.score_token_match must be 'exact' or 'contains'.")
    if config.phase0.aggregation not in {"selected_token", "min", "max", "mean"}:
        raise ConfigError(
            "phase0.aggregation must be selected_token, min, max, or mean."
        )
    if config.phase0.fallback_aggregation not in {"min", "max", "mean"}:
        raise ConfigError("phase0.fallback_aggregation must be min, max, or mean.")
    if config.generation.max_new_tokens <= 0:
        raise ConfigError("generation.max_new_tokens must be positive.")
    if config.datasets.harmful.limit <= 0 or config.datasets.harmless.limit <= 0:
        raise ConfigError("dataset limits must be positive.")
    _validate_range(
        config.expected_activations.harmful_mean_range,
        "expected_activations.harmful_mean_range",
    )
    _validate_range(
        config.expected_activations.harmless_mean_range,
        "expected_activations.harmless_mean_range",
    )


def _validate_phase1_config(config: Phase1Config) -> None:
    if config.backend.name != "transformers":
        raise ConfigError("Phase 1 currently supports only the transformers backend.")
    if config.suppression.layer < 0:
        raise ConfigError("suppression.layer must be non-negative.")
    if config.suppression.neuron < 0:
        raise ConfigError("suppression.neuron must be non-negative.")
    if config.generation.max_new_tokens <= 0:
        raise ConfigError("generation.max_new_tokens must be positive.")
    if config.dataset.limit <= 0:
        raise ConfigError("dataset.limit must be positive.")
    if config.judge.max_new_tokens <= 0:
        raise ConfigError("judge.max_new_tokens must be positive.")
    if not 0.0 <= config.pass_criteria.min_suppressed_asr <= 1.0:
        raise ConfigError("pass_criteria.min_suppressed_asr must be between 0 and 1.")
    if not 0.0 <= config.pass_criteria.max_clean_asr <= 1.0:
        raise ConfigError("pass_criteria.max_clean_asr must be between 0 and 1.")


def _parse_model_config(raw: dict[str, Any], path: str) -> ModelConfig:
    return ModelConfig(
        id=str(_required(raw, "id", path)),
        revision=str(_required(raw, "revision", path)),
        dtype=str(_required(raw, "dtype", path)),
        trust_remote_code=bool(raw.get("trust_remote_code", True)),
        quantization=_parse_quantization_config(raw),
    )


def _parse_quantization_config(raw: dict[str, Any]) -> QuantizationConfig | None:
    quant_raw = raw.get("quantization")
    if not isinstance(quant_raw, dict):
        return None
    return QuantizationConfig(
        load_in_4bit=bool(quant_raw.get("load_in_4bit", False)),
        bnb_4bit_compute_dtype=str(quant_raw.get("bnb_4bit_compute_dtype", "float16")),
        bnb_4bit_quant_type=str(quant_raw.get("bnb_4bit_quant_type", "nf4")),
        bnb_4bit_use_double_quant=bool(
            quant_raw.get("bnb_4bit_use_double_quant", True)
        ),
    )


def _parse_dataset_config(raw: dict[str, Any], path: str) -> TextDatasetConfig:
    fields = _required(raw, "text_fields", path)
    if not isinstance(fields, list | tuple) or not fields:
        raise ConfigError(f"{path}.text_fields must be a non-empty list.")
    return TextDatasetConfig(
        id=str(_required(raw, "id", path)),
        split=str(raw.get("split", "train")),
        limit=int(raw.get("limit", 5)),
        text_fields=tuple(str(field) for field in fields),
        name=(None if raw.get("name") is None else str(raw.get("name"))),
        input_field=(
            None if raw.get("input_field") is None else str(raw.get("input_field"))
        ),
        requires_hf_token=bool(raw.get("requires_hf_token", False)),
    )


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise MissingDependencyError(
            "PyYAML is required to read config files. Install with `pip install pyyaml`."
        ) from exc

    if not path.exists():
        raise ConfigError(f"Config file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ConfigError(f"Config root must be a mapping: {path}")
    return loaded


def _mapping(
    raw: dict[str, Any], key: str, parent: str | None = None
) -> dict[str, Any]:
    value = raw.get(key)
    path = key if parent is None else f"{parent}.{key}"
    if not isinstance(value, dict):
        raise ConfigError(f"{path} must be a mapping.")
    return value


def _required(raw: dict[str, Any], key: str, parent: str) -> Any:
    if key not in raw:
        raise ConfigError(f"Missing required config key: {parent}.{key}")
    return raw[key]


def _float_pair(value: Any, path: str) -> tuple[float, float]:
    if not isinstance(value, list | tuple) or len(value) != 2:
        raise ConfigError(f"{path} must be a two-item numeric list.")
    return (float(value[0]), float(value[1]))


def _validate_range(value: tuple[float, float], path: str) -> None:
    if value[0] >= value[1]:
        raise ConfigError(f"{path} lower bound must be less than upper bound.")
