from __future__ import annotations

import json
import types
from typing import Literal, TypeGuard, Union, get_args, get_origin

from pydantic import BaseModel, ConfigDict, Field

TaskName = Literal["triage", "summarization", "extraction"]
FieldStatus = Literal["present", "absent", "ambiguous"]
DocumentStatus = Literal["valid", "contradictory", "superseded", "unsupported"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EvidenceField(StrictModel):
    value: str | list[str] | None
    status: FieldStatus
    citation: str | None = None


class TriageOutput(StrictModel):
    queue: Literal[
        "card_dispute",
        "fraud_report",
        "account_servicing",
        "lending",
        "complaint",
        "escalate",
        "unsupported",
    ]
    escalation_required: bool
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    draft_reply: str
    human_review_required: Literal[True]
    customer_outcome: None = None


class SummarizationOutput(StrictModel):
    document_status: DocumentStatus
    title: EvidenceField
    version: EvidenceField
    effective_date: EvidenceField
    purpose: EvidenceField
    required_steps: EvidenceField
    exceptions: EvidenceField

    def evidence_fields(self) -> dict[str, EvidenceField]:
        return {
            "title": self.title,
            "version": self.version,
            "effective_date": self.effective_date,
            "purpose": self.purpose,
            "required_steps": self.required_steps,
            "exceptions": self.exceptions,
        }


class PolicyExtraction(StrictModel):
    document_status: DocumentStatus
    policy_name: EvidenceField
    version: EvidenceField
    effective_date: EvidenceField
    jurisdictions: EvidenceField
    beneficial_ownership_threshold: EvidenceField
    review_frequency: EvidenceField
    required_documents: EvidenceField

    def evidence_fields(self) -> dict[str, EvidenceField]:
        return {
            "policy_name": self.policy_name,
            "version": self.version,
            "effective_date": self.effective_date,
            "jurisdictions": self.jurisdictions,
            "beneficial_ownership_threshold": self.beneficial_ownership_threshold,
            "review_frequency": self.review_frequency,
            "required_documents": self.required_documents,
        }


OUTPUT_SCHEMAS: dict[TaskName, type[StrictModel]] = {
    "triage": TriageOutput,
    "summarization": SummarizationOutput,
    "extraction": PolicyExtraction,
}


def _is_model(annotation: object) -> TypeGuard[type[BaseModel]]:
    return isinstance(annotation, type) and issubclass(annotation, BaseModel)


def _annotation_parts(annotation: object) -> tuple[object, ...]:
    origin = get_origin(annotation)
    if origin is Union or origin is types.UnionType:
        return get_args(annotation)
    return (annotation,)


def _describe_annotation(annotation: object) -> str:
    if annotation is None or annotation is type(None):
        return "null"
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is Literal:
        return "one of " + ", ".join(json.dumps(arg) for arg in args)
    if origin is Union or origin is types.UnionType:
        return " | ".join(_describe_annotation(arg) for arg in args)
    if origin is list:
        inner = _describe_annotation(args[0]) if args else "any"
        return f"array of {inner}"
    if _is_model(annotation):
        return f"{annotation.__name__} object"
    if annotation is str:
        return "string"
    if annotation is bool:
        return "boolean"
    if annotation is int:
        return "integer"
    if annotation is float:
        return "number"
    return str(annotation).replace("typing.", "")


def _shape(annotation: object) -> object:
    if annotation is None or annotation is type(None):
        return None
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is Literal:
        if len(args) == 1:
            return args[0]
        return " | ".join(str(arg) for arg in args)
    if _is_model(annotation):
        return {
            name: _shape(field.annotation)
            for name, field in annotation.model_fields.items()
        }
    if origin is Union or origin is types.UnionType:
        return " | ".join(_describe_annotation(arg) for arg in args)
    if origin is list:
        inner = _shape(args[0]) if args else "any"
        return [inner]
    if annotation is str:
        return "string"
    if annotation is bool:
        return "boolean"
    if annotation is int:
        return "integer"
    if annotation is float:
        return "number"
    return _describe_annotation(annotation)


def _nested_models(annotation: object) -> list[type[BaseModel]]:
    found: list[type[BaseModel]] = []
    for part in _annotation_parts(annotation):
        origin = get_origin(part)
        if origin is list:
            for inner in get_args(part):
                found.extend(_nested_models(inner))
        elif _is_model(part):
            found.append(part)
    return found


def _format_model(model: type[BaseModel]) -> str:
    lines = [f"{model.__name__}"]
    extra = model.model_config.get("extra")
    if extra == "forbid":
        lines.append("Do not include any properties other than the fields listed.")
    for name, field in model.model_fields.items():
        requirement = "required" if field.is_required() else "optional"
        annotation = field.annotation
        type_text = _describe_annotation(annotation) if annotation is not None else "any"
        suffix = ""
        if not field.is_required():
            suffix = f", default {json.dumps(field.default)}"
        lines.append(f"- {name} ({requirement}): {type_text}{suffix}")
    if model.__name__ == "EvidenceField":
        lines.append(
            'When status is "present", citation must equal an exact section heading '
            'from the source document, such as "1. Document Control".'
        )
    return "\n".join(lines)


def schema_description(model: type[BaseModel]) -> str:
    """Return a generated description of a Pydantic model for use in prompts."""
    models: list[type[BaseModel]] = []
    seen: set[str] = set()

    def collect(candidate: type[BaseModel]) -> None:
        if candidate.__name__ in seen:
            return
        seen.add(candidate.__name__)
        models.append(candidate)
        for field in candidate.model_fields.values():
            if field.annotation is None:
                continue
            for nested in _nested_models(field.annotation):
                collect(nested)

    collect(model)
    sections = [_format_model(item) for item in models]
    sections.append(
        "Instance shape (replace the placeholders with values from the source):\n"
        + json.dumps(_shape(model), indent=2)
    )
    sections.append(
        "Return one JSON data instance matching this shape. "
        "Do not return a JSON Schema document. "
        "Do not include keys such as $defs, properties, additionalProperties, "
        "required, or type. "
        "Every required field must be present; use null when a value is absent."
    )
    return "\n\n".join(sections)

