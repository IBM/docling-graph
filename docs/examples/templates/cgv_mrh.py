""" 
Extraction model for French home insurance policy documents (Assurance Multirisque Habitation / MRH).

Robustness goals
- Be permissive: missing or partially extracted data should not fail Pydantic validation.
- Prefer dropping clearly invalid list items (e.g., dicts missing identifiers) rather than raising.
- Keep semantic entities (Guarantee/Option/Offering/Asset types) while allowing partial fields.

Design notes
- No section/subsection hierarchy.
- Traceability is represented by ContractClause + optional OCR/layout evidence spans.

Version: 3.0.0
"""

from __future__ import annotations

import logging
import re
from enum import Enum
from typing import Any, List

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1) Docling-Graph helper (edge metadata)
# ---------------------------------------------------------------------------

def edge(label: str, default: Any = None, **kwargs: Any) -> Any:
    """Helper to tag relationship fields for Docling-Graph (knowledge graph edges)."""
    if "default_factory" in kwargs:
        default_factory = kwargs.pop("default_factory")
        return Field(default_factory=default_factory, json_schema_extra={"edge_label": label}, **kwargs)
    return Field(default, json_schema_extra={"edge_label": label}, **kwargs)


# ---------------------------------------------------------------------------
# 2) Normalization helpers
# ---------------------------------------------------------------------------

def _parse_french_number(v: Any) -> Any:
    """Convert French-formatted numbers (e.g., '1 200,50') to float when possible."""
    if isinstance(v, int | float):
        return float(v)
    if isinstance(v, str):
        clean_v = re.sub(r"[^\d,\.-]", "", v).replace(",", ".")
        try:
            return float(clean_v)
        except ValueError:
            logger.warning("Could not parse number: %r", v)
            return None
    return None


def _normalize_currency(v: Any) -> str | None:
    """Normalize common currency symbols to ISO 4217."""
    if v is None:
        return None
    if not isinstance(v, str):
        return str(v)
    symbol_map = {"€": "EUR", "$": "USD", "£": "GBP"}
    v_clean = v.strip().upper()
    for sym, code in symbol_map.items():
        if sym in v_clean:
            return code
    return v_clean if len(v_clean) == 3 else "EUR"


def _filter_list_items(
    v: Any,
    field_name: str,
    required_keys: list[str] | None = None,
) -> Any:
    """Drop garbage items from lists (strings/footers/invalid dicts)."""
    if not isinstance(v, list):
        return v

    required_keys = required_keys or []
    out: list[Any] = []

    for item in v:
        if isinstance(item, BaseModel):
            out.append(item)
            continue

        if isinstance(item, dict):
            missing = [k for k in required_keys if not item.get(k)]
            if missing:
                logger.warning(
                    "Dropping invalid dict from %s (missing %s): %r",
                    field_name,
                    ",".join(missing),
                    {k: item.get(k) for k in required_keys},
                )
                continue
            out.append(item)
            continue

        if isinstance(item, str):
            logger.warning("Dropping garbage string from %s: %s...", field_name, item[:80])
            continue

        logger.warning("Dropping garbage item from %s: %r", field_name, item)

    return out


# ---------------------------------------------------------------------------
# 3) Enums
# ---------------------------------------------------------------------------

class PartyRole(str, Enum):
    INSURER = "insurer"
    INSURED = "insured"
    POLICYHOLDER = "policyholder"
    TENANT = "tenant"
    OWNER = "owner"
    THIRD_PARTY = "third_party"
    OTHER = "other"


class Periodicity(str, Enum):
    YEARLY = "yearly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    HALF_YEARLY = "half_yearly"
    OTHER = "other"


class OccupancyStatus(str, Enum):
    TENANT = "tenant"
    OWNER_OCCUPANT = "owner_occupant"
    OWNER_NON_OCCUPANT = "owner_non_occupant"
    CO_OWNER = "co_owner"
    OCCUPANT_FREE = "occupant_free"
    OTHER = "other"


class ResidenceUse(str, Enum):
    PRIMARY = "primary"
    SECONDARY = "secondary"
    RENTED_OUT = "rented_out"
    HOLIDAY_RENTAL = "holiday_rental"
    STUDENT = "student"
    OTHER = "other"


class PropertyType(str, Enum):
    APARTMENT = "apartment"
    HOUSE = "house"
    MOBILE_HOME = "mobile_home"
    OTHER = "other"


class AssetCategory(str, Enum):
    BUILDING = "building"
    OUTBUILDING = "outbuilding"
    CONTENTS = "contents"
    VALUABLES = "valuables"
    OUTDOOR = "outdoor"
    POOL = "pool"
    ENERGY = "energy"
    LIABILITY = "liability"
    ASSISTANCE = "assistance"
    OTHER = "other"


# ---------------------------------------------------------------------------
# 4) Value objects (is_entity=False)
# ---------------------------------------------------------------------------

class Money(BaseModel):
    """A monetary amount, permissive to partial extraction."""

    model_config = ConfigDict(is_entity=False, extra="ignore")

    amount: float | None = Field(
        None,
        description="Numeric amount. If not extracted, keep null (do not fail validation).",
    )
    currency: str | None = Field(
        "EUR",
        description="ISO 4217 currency code (default EUR for FR MRH).",
    )
    indexed_by: str | None = Field(
        None,
        description="Index name if expressed as 'x fois l'indice' (FFB, IRL, etc.).",
        examples=["FFB", "IRL", "Indice du prix de la construction"],
    )

    @model_validator(mode="before")
    @classmethod
    def coerce_from_number(cls, v: Any) -> Any:
        if isinstance(v, int | float):
            return {"amount": float(v), "currency": "EUR"}
        return v

    @field_validator("amount", mode="before")
    @classmethod
    def normalize_amount(cls, v: Any) -> Any:
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return float(v)
        return _parse_french_number(v)

    @field_validator("currency", mode="before")
    @classmethod
    def normalize_currency(cls, v: Any) -> Any:
        return _normalize_currency(v)


class Deductible(BaseModel):
    """Deductible/franchise; permissive (can be partially filled)."""

    model_config = ConfigDict(is_entity=False, extra="ignore")

    amount: Money | None = Field(None)
    deductible_type: str | None = Field(
        None,
        description="Type of deductible (fixed, percentage, legal CAT-NAT deductible, etc.).",
        examples=["Fixe", "Pourcentage", "Franchise légale"],
    )
    context: str | None = Field(
        None,
        description="Where/when it applies (per event, per claim, per guarantee).",
        examples=["Par sinistre", "Catastrophes naturelles", "Vol"],
    )


class CoverageCondition(BaseModel):
    """Condition / prerequisite for a coverage to apply."""

    model_config = ConfigDict(is_entity=False, extra="ignore")

    text: str | None = Field(
        None,
        description=(
            "Condition verbatim (prevention measures, security levels, occupancy limits, declarations). "
            "Leave null if not extracted."
        ),
        examples=[
            "Les portes doivent être verrouillées par une serrure 3 points.",
            "L'habitation ne doit pas rester inoccupée plus de 90 jours consécutifs.",
        ],
    )
    max_unoccupied_days: int | None = Field(
        None,
        description="If about 'inhabitation/inoccupation', extract the maximum days if stated.",
        examples=[30, 60, 90],
    )


class BoundingBox(BaseModel):
    """Simple bbox (page coordinate system depends on your OCR pipeline)."""

    model_config = ConfigDict(is_entity=False, extra="ignore")

    x0: float | None = None
    y0: float | None = None
    x1: float | None = None
    y1: float | None = None


class EvidenceSpan(BaseModel):
    """Deterministic provenance anchor from OCR/PDF extraction."""

    model_config = ConfigDict(is_entity=False, extra="ignore")

    page: int | None = Field(None, description="1-based page number if available.")
    block_id: str | None = Field(None, description="Stable OCR/layout block identifier if available.")
    bbox: BoundingBox | None = Field(None, description="Bounding box for the evidence region.")
    char_start: int | None = Field(None, description="Start offset within the block text if applicable.")
    char_end: int | None = Field(None, description="End offset within the block text if applicable.")
    text_snippet: str | None = Field(None, description="Short snippet for debugging (optional).")


# ---------------------------------------------------------------------------
# 5) Entities (graph nodes)
# ---------------------------------------------------------------------------

class DefinitionTerm(BaseModel):
    """A term defined in the policy glossary/lexicon."""

    model_config = ConfigDict(graph_id_fields=["term"], extra="ignore")

    term: str | None = Field(
        None,
        description="The defined term (keep exact wording).",
        examples=["Vétusté", "Effraction"],
    )
    definition: str | None = Field(None, description="Definition text (verbatim if possible).")


class Risk(BaseModel):
    """A peril/risk concept (covered or excluded)."""

    model_config = ConfigDict(graph_id_fields=["name"], extra="ignore")

    name: str | None = Field(None, examples=["Incendie", "Dégâts des eaux", "Vol"])

    @model_validator(mode="before")
    @classmethod
    def coerce_from_string(cls, v: Any) -> Any:
        if isinstance(v, str):
            return {"name": v}
        return v


class InsuredAssetType(BaseModel):
    """A type of insured asset explicitly mentioned by the policy."""

    model_config = ConfigDict(graph_id_fields=["name"], extra="ignore")

    name: str | None = Field(
        None,
        description="Asset type name as written (e.g., 'Dépendances', 'Mobilier', 'Piscine').",
    )
    category: AssetCategory | None = Field(None)
    description: str | None = Field(None)


class ExclusionClause(BaseModel):
    """An exclusion clause (specific non-covered situation/property)."""

    model_config = ConfigDict(graph_id_fields=["text_summary"], extra="ignore")

    text_summary: str | None = Field(
        None,
        description="Short summary identifier (keep short).",
    )
    full_text: str | None = Field(
        None,
        description="Full clause text (verbatim if possible).",
    )

    exclut_risques: List[Risk] = edge(
        label="EXCLUDES_RISK",
        default_factory=list,
        description="Risks explicitly excluded by this clause.",
    )
    exclut_biens: List[InsuredAssetType] = edge(
        label="EXCLUDES_ASSET_TYPE",
        default_factory=list,
        description="Asset types excluded by this clause.",
    )

    evidence: List[EvidenceSpan] = Field(default_factory=list)


class Guarantee(BaseModel):
    """A guarantee/coverage of the policy."""

    model_config = ConfigDict(graph_id_fields=["name"], extra="ignore")

    name: str | None = Field(None, description="Coverage name as written.")
    description_courte: str | None = None

    couvre_risques: List[Risk] = edge(label="COVERS_RISK", default_factory=list)
    couvre_biens: List[InsuredAssetType] = edge(label="COVERS_ASSET_TYPE", default_factory=list)

    a_exclusions: List[ExclusionClause] = edge(label="HAS_EXCLUSION", default_factory=list)
    a_conditions: List[CoverageCondition] = edge(label="HAS_CONDITION", default_factory=list)
    a_franchises: List[Deductible] = edge(label="HAS_DEDUCTIBLE", default_factory=list)

    plafond_garantie: Money | None = edge(label="HAS_LIMIT", default=None)

    evidence: List[EvidenceSpan] = Field(default_factory=list)


class Option(BaseModel):
    """An optional add-on (option/pack/renfort)."""

    model_config = ConfigDict(graph_id_fields=["name"], extra="ignore")

    name: str | None = Field(None, description="Option name as written.")
    description: str | None = None

    etend_garanties: List[Guarantee] = edge(label="EXTENDS_GUARANTEE", default_factory=list)
    couvre_biens: List[InsuredAssetType] = edge(label="COVERS_ASSET_TYPE", default_factory=list)

    evidence: List[EvidenceSpan] = Field(default_factory=list)


class InsuranceOffering(BaseModel):
    """A product offering / formula / tier / contract variant."""

    model_config = ConfigDict(graph_id_fields=["name"], extra="ignore")

    name: str | None = Field(
        None,
        description=(
            "Offering/formula name as written (e.g., 'Essentielle', 'Confort', 'PNO'). "
            "Leave null if not extracted."
        ),
    )
    tier: int | None = Field(None, description="Tier number if expressed as n°1/n°2/n°3.")
    occupancy_status: List[OccupancyStatus] = Field(default_factory=list)
    residence_use: List[ResidenceUse] = Field(default_factory=list)
    property_types: List[PropertyType] = Field(default_factory=list)

    includes_guarantees: List[Guarantee] = edge(label="INCLUDES_GUARANTEE", default_factory=list)
    optional_guarantees: List[Guarantee] = edge(label="OPTIONAL_GUARANTEE", default_factory=list)
    available_options: List[Option] = edge(label="AVAILABLE_OPTION", default_factory=list)

    notes: str | None = None

    evidence: List[EvidenceSpan] = Field(default_factory=list)


class ContractClause(BaseModel):
    """A traceable text unit (article/paragraph/table row) + deterministic evidence spans."""

    model_config = ConfigDict(graph_id_fields=["reference"], extra="ignore")

    reference: str | None = Field(
        None,
        description=(
            "Human-friendly reference if available (Article/Page/Table). "
            "Do not rely on LLM-invented headings; prefer OCR anchors in evidence."
        ),
    )
    title: str | None = None
    raw_text: str | None = Field(
        None,
        description="Raw extracted text of this clause (optional).",
    )

    evidence: List[EvidenceSpan] = Field(
        default_factory=list,
        description="OCR/PDF provenance anchors for this clause.",
    )

    definit_termes: List[DefinitionTerm] = edge(label="DEFINES_TERM", default_factory=list)
    mentionne_garanties: List[Guarantee] = edge(label="MENTIONS_GUARANTEE", default_factory=list)
    mentionne_options: List[Option] = edge(label="MENTIONS_OPTION", default_factory=list)
    mentionne_offres: List[InsuranceOffering] = edge(label="MENTIONS_OFFERING", default_factory=list)


class PolicyDocument(BaseModel):
    """Root model for a MRH policy document (CGV/conditions générales)."""

    model_config = ConfigDict(graph_id_fields=["document_ref"], extra="ignore")

    document_ref: str | None = Field(
        None,
        description="Document reference (cover/footer). Optional to avoid validation failures.",
    )
    nom_assureur: str | None = Field(
        None,
        description="Insurer/brand name. Optional to avoid validation failures.",
    )
    version_date: str | None = None
    product_name: str | None = None

    liste_offres: List[InsuranceOffering] = edge(label="HAS_OFFERING", default_factory=list)
    liste_biens_types: List[InsuredAssetType] = edge(label="MENTIONS_ASSET_TYPE", default_factory=list)
    liste_garanties: List[Guarantee] = edge(label="INCLUDES_GUARANTEE", default_factory=list)
    liste_options: List[Option] = edge(label="OFFERS_OPTION", default_factory=list)
    liste_exclusions_generales: List[ExclusionClause] = edge(label="HAS_GENERAL_EXCLUSION", default_factory=list)
    liste_clauses: List[ContractClause] = edge(label="HAS_CLAUSE", default_factory=list)

    # -------------------
    # Guardrails: drop garbage list items before validation
    # -------------------
    @field_validator("liste_offres", mode="before")
    @classmethod
    def _filter_offers(cls, v: Any) -> Any:
        return _filter_list_items(v, "liste_offres", required_keys=["name"])

    @field_validator("liste_biens_types", mode="before")
    @classmethod
    def _filter_asset_types(cls, v: Any) -> Any:
        return _filter_list_items(v, "liste_biens_types", required_keys=["name"])

    @field_validator("liste_garanties", mode="before")
    @classmethod
    def _filter_guarantees(cls, v: Any) -> Any:
        return _filter_list_items(v, "liste_garanties", required_keys=["name"])

    @field_validator("liste_options", mode="before")
    @classmethod
    def _filter_options(cls, v: Any) -> Any:
        return _filter_list_items(v, "liste_options", required_keys=["name"])

    @field_validator("liste_exclusions_generales", mode="before")
    @classmethod
    def _filter_exclusions(cls, v: Any) -> Any:
        # Allow partial exclusions (summary OR full_text). Drop only if both missing.
        if not isinstance(v, list):
            return v
        out = []
        for item in v:
            if isinstance(item, dict):
                if not item.get("text_summary") and not item.get("full_text"):
                    logger.warning("Dropping invalid exclusion dict with no text: %r", item)
                    continue
            out.append(item)
        return _filter_list_items(out, "liste_exclusions_generales")

    @field_validator("liste_clauses", mode="before")
    @classmethod
    def _filter_clauses(cls, v: Any) -> Any:
        # Clauses can be anchored only by evidence; don't require reference/raw_text.
        return _filter_list_items(v, "liste_clauses")