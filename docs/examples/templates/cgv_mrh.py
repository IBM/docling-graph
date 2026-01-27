"""
Modèle d'extraction pour les Conditions Générales de Vente (CGV) - Assurance Multirisque Habitation (MRH).

Ce modèle est conçu pour extraire la structure documentaire, les garanties, les exclusions
et les conditions financières des contrats d'assurance habitation français.

Version: 2.0.0 (Optimisée pour Docling-Graph)
"""

from __future__ import annotations

import logging
import re
from enum import Enum
from typing import Any, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Logger pour suivre les corrections automatiques de données
logger = logging.getLogger(__name__)


# --- 1. Helper Function (OBLIGATOIRE pour Docling-Graph) ---
def edge(label: str, **kwargs: Any) -> Any:
    """Fonction helper pour créer des arêtes (relations) dans le graphe de connaissances."""
    return Field(..., json_schema_extra={"edge_label": label}, **kwargs)


# --- 2. Fonctions de Normalisation (Helpers) ---


def _parse_french_number(v: Any) -> Any:
    """Convertit des chaînes numériques françaises (ex: '1 200,50') en float."""
    if isinstance(v, int | float):
        return v
    if isinstance(v, str):
        # Nettoyage : suppression des espaces insécables et symboles monétaires
        clean_v = re.sub(r"[^\d,\.-]", "", v).replace(",", ".")
        try:
            return float(clean_v)
        except ValueError:
            logger.warning(f"Impossible de convertir '{v}' en nombre.")
    return v


def _normalize_currency(v: Any) -> Any:
    """Normalise les symboles monétaires en codes ISO 4217."""
    if not isinstance(v, str):
        return v

    symbol_map = {"€": "EUR", "$": "USD", "£": "GBP"}
    v_clean = v.strip().upper()

    # Remplacement symbole -> code
    for sym, code in symbol_map.items():
        if sym in v_clean:
            return code

    return v_clean if len(v_clean) == 3 else "EUR"  # Défaut à EUR pour les contrats FR


# --- 3. Enums ---


class PartyRole(str, Enum):
    ASSUREUR = "assureur"
    ASSURE = "assure"
    SOUSCRIPTEUR = "souscripteur"
    LOCATAIRE = "locataire"
    PROPRIETAIRE = "proprietaire"
    TIERS = "tiers"
    AUTRE = "autre"


class Periodicity(str, Enum):
    ANNUELLE = "annuelle"
    MENSUELLE = "mensuelle"
    TRIMESTRIELLE = "trimestrielle"
    SEMESTRIELLE = "semestrielle"
    AUTRE = "autre"


# --- 4. Composants (Value Objects - is_entity=False) ---
# Ces objets sont dédupliqués par contenu. S'ils ont les mêmes valeurs, ils partagent le même nœud.


class Money(BaseModel):
    """Représente une valeur monétaire."""

    model_config = ConfigDict(is_entity=False)

    amount: float = Field(...)
    currency: str = Field("EUR")
    indexed_by: str | None = Field(None)

    @model_validator(mode="before")
    @classmethod
    def coerce_from_number(cls, v: Any) -> Any:
        """
        Fixes error: Input should be a valid dictionary or instance of Money.
        Converts raw numbers (e.g., 15000.0) into {'amount': 15000.0}
        """
        if isinstance(v, int | float):
            return {"amount": float(v), "currency": "EUR"}
        return v

    # Keep your existing validators below if needed
    @field_validator("amount", mode="before")
    @classmethod
    def normalize_amount(cls, v: Any) -> Any:
        return _parse_french_number(v)


class Franchise(BaseModel):
    """Franchise (montant restant à la charge de l'assuré)."""

    model_config = ConfigDict(is_entity=False)

    # Note: Pydantic will now automatically use Money.coerce_from_number
    # when it encounters a float here.
    amount: Money = Field(...)

    type_franchise: str | None = Field(None)
    context: str | None = Field(None)


class ConditionApplication(BaseModel):
    """Condition ou prérequis pour l'application d'une garantie."""

    model_config = ConfigDict(is_entity=False)

    text: str = Field(
        ...,
        description="Description textuelle de la condition (mesures de prévention, état des lieux, etc.).",
        examples=[
            "Les portes doivent être verrouillées par une serrure 3 points.",
            "L'habitation ne doit pas rester inoccupée plus de 90 jours consécutifs.",
            "Un détecteur de fumée doit être installé.",
        ],
    )
    max_unoccupied_days: int | None = Field(
        None,
        description="Si la condition concerne l'inhabitation, extraire le nombre maximum de jours autorisés.",
        examples=[30, 60, 90],
    )


# --- 5. Entités (Graph Nodes - graph_id_fields) ---
# Ces objets sont uniques et identifiables.


class DefinitionTerme(BaseModel):
    """Terme défini dans le lexique des CGV."""

    model_config = ConfigDict(graph_id_fields=["term"])

    term: str = Field(
        ...,
        description="Le terme exact défini (souvent en gras ou dans une section 'Définitions').",
        examples=["Accident", "Vétusté", "Mobilier", "Effraction"],
    )
    definition: str | None = Field(
        None,
        description="La définition complète du terme telle qu'écrite dans le contrat.",
        examples=["Tout événement soudain, imprévisible et extérieur à la victime..."],
    )


class Risque(BaseModel):
    """Un risque ou péril couvert (ou exclu)."""

    model_config = ConfigDict(graph_id_fields=["name"])

    name: str = Field(...)

    @model_validator(mode="before")
    @classmethod
    def coerce_from_string(cls, v: Any) -> Any:
        """
        Fixes error: Input should be a valid dictionary or instance of Risque.
        Converts raw strings (e.g., "Incendie") into {'name': "Incendie"}
        """
        if isinstance(v, str):
            return {"name": v}
        return v


class Exclusion(BaseModel):
    """Une clause d'exclusion spécifique."""

    model_config = ConfigDict(graph_id_fields=["text_summary"])

    text_summary: str = Field(
        ...,
        description="Résumé court et unique de l'exclusion (servant d'identifiant).",
        examples=["Exclusion vol sans effraction", "Exclusion dommages défaut entretien"],
    )
    full_text: str = Field(
        ...,
        description="Le texte complet de la clause d'exclusion.",
        examples=[
            "Sont exclus les vols commis sans effraction ni violence sur la personne.",
            "Dommages résultant d'un défaut d'entretien notoire.",
        ],
    )

    exclut_risques: List[Risque] = edge(
        label="EXCLUT_RISQUE",
        default_factory=list,
        description="Liste des risques/concepts spécifiquement exclus par cette clause.",
    )


class Garantie(BaseModel):
    """Une garantie principale du contrat (Nœud central)."""

    model_config = ConfigDict(graph_id_fields=["name"])

    name: str = Field(
        ...,
        description="Nom commercial ou technique de la garantie.",
        examples=[
            "Incendie et événements assimilés",
            "Vol et vandalisme",
            "Responsabilité Civile Vie Privée",
        ],
    )
    description_courte: str | None = Field(
        None,
        description="Brève description de ce que couvre la garantie.",
        examples=["Garantit les dommages causés aux biens assurés par le feu, l'explosion..."],
    )

    # Relations (Edges)
    couvre_risques: List[Risque] = edge(
        label="COUVRE_RISQUE",
        default_factory=list,
        description="Liste des risques couverts par cette garantie.",
    )
    a_exclusions: List[Exclusion] = edge(
        label="A_EXCLUSION",
        default_factory=list,
        description="Exclusions spécifiques limitant cette garantie.",
    )
    a_conditions: List[ConditionApplication] = edge(
        label="A_CONDITION",
        default_factory=list,
        description="Conditions (prérequis) pour que la garantie s'applique.",
    )
    a_franchises: List[Franchise] = edge(
        label="A_FRANCHISE",
        default_factory=list,
        description="Franchises applicables spécifiquement à cette garantie.",
    )
    plafond_garantie: Money | None = edge(
        label="A_PLAFOND",
        description="Montant maximum d'indemnisation pour cette garantie (Plafond).",
    )


class Option(BaseModel):
    """Option facultative pouvant être ajoutée au contrat."""

    model_config = ConfigDict(graph_id_fields=["name"])

    name: str = Field(
        ...,
        description="Nom de l'option ou du pack optionnel.",
        examples=["Pack Plein Air", "Option Valeur à Neuf", "Protection Juridique"],
    )
    description: str | None = Field(
        None,
        description="Description des avantages de l'option.",
        examples=["Extension de la garantie vol aux dépendances séparées."],
    )


class Clause(BaseModel):
    """Une unité de texte contractuel (Article, Paragraphe)."""

    model_config = ConfigDict(graph_id_fields=["reference"])

    reference: str = Field(
        ...,
        description="Référence unique de la clause (Numéro d'article, page, paragraphe).",
        examples=["Article 3.1", "Page 12 - Paragraphe Vol", "Art. L121-1"],
    )
    titre: str | None = Field(
        None,
        description="Titre de la clause ou de l'article.",
        examples=["Biens non garantis", "Déchéance de garantie"],
    )
    contenu_texte: str = Field(
        ...,
        description="Le contenu textuel brut de la clause.",
    )

    definit_termes: List[DefinitionTerme] = edge(
        label="DEFINIT_TERME",
        default_factory=list,
        description="Termes du lexique définis dans cette clause.",
    )


class Section(BaseModel):
    """Section ou Chapitre du document (Hiérarchie)."""

    model_config = ConfigDict(graph_id_fields=["title"])

    title: str = Field(
        ...,
        description="Titre de la section ou du chapitre.",
        examples=["VOS GARANTIES", "CE QUE NOUS EXCLUONS", "DISPOSITIONS GÉNÉRALES"],
    )

    contient_clauses: List[Clause] = edge(
        label="CONTIENT_CLAUSE",
        default_factory=list,
        description="Clauses ou articles contenus directement dans cette section.",
    )
    sous_sections: List[Section] = edge(
        label="A_SOUS_SECTION", default_factory=list, description="Sous-sections imbriquées."
    )


class DocumentCGV(BaseModel):
    """Document racine : Conditions Générales de Vente."""

    model_config = ConfigDict(graph_id_fields=["document_ref"])

    document_ref: str = Field(
        ...,
        description="Référence unique du document (souvent en pied de page ou couverture).",
        examples=["CG 410", "Réf. 2024-A", "CGV-MRH-2023"],
    )
    nom_assureur: str = Field(
        ...,
        description="Nom de la compagnie d'assurance émettrice.",
        examples=["AXA", "MACIF", "ALLIANZ", "GROUPAMA"],
    )
    version_date: str | None = Field(
        None,
        description="Date ou version du document.",
        examples=["Édition Janvier 2024", "01/2023"],
    )

    # Structure du graphe
    structure_sections: List[Section] = edge(
        label="A_SECTION",
        default_factory=list,
        description="Sections principales du document (Table des matières).",
    )

    # Entités métier extraites globalement
    liste_garanties: List[Garantie] = edge(
        label="INCLUT_GARANTIE",
        default_factory=list,
        description="Toutes les garanties identifiées dans le document.",
    )
    liste_options: List[Option] = edge(
        label="PROPOSE_OPTION",
        default_factory=list,
        description="Options facultatives mentionnées.",
    )
    liste_exclusions_generales: List[Exclusion] = edge(
        label="A_EXCLUSION_GENERALE",
        default_factory=list,
        description="Exclusions s'appliquant à tout le contrat (guerre, nucléaire, etc.).",
    )

    # Filter out bad data from the list
    @field_validator("structure_sections", mode="before")
    @classmethod
    def filter_garbage_sections(cls, v: Any) -> Any:
        """
        Filters out any item in the list that is not a dictionary.
        This removes the hallucinated strings/footers the LLM sometimes inserts.
        """
        if isinstance(v, list):
            valid_items = []
            for item in v:
                if isinstance(item, dict):
                    valid_items.append(item)
                elif isinstance(item, str):
                    logger.warning(f"Dropping garbage string from sections: {item[:50]}...")
            return valid_items
        return v
