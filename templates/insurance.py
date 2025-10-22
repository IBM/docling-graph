# insurance.py

from pydantic import BaseModel, Field
from typing import Optional, List, Any

def Edge(label: str, **kwargs: Any) -> Any:
    """Helper function to create a Pydantic Field with edge metadata."""
    return Field(..., json_schema_extra={'edge_label': label}, **kwargs)

# --- Enhanced Node Definitions ---

class ReimbursementCondition(BaseModel):
    """
    A structured model to capture specific conditions for reimbursement.
    This tells the model to look for and structure granular details.
    """
    item_category: str = Field(
        description="The category of item covered by this condition.",
        examples=["appareils électroménagers et informatiques", "meubles"]
    )
    age_limit_years: int = Field(
        description="The maximum age of the item in years to be eligible for the condition."
    )
    reimbursement_basis: str = Field(
        default="valeur à neuf",
        description="The basis for the reimbursement value.",
        examples=["valeur à neuf", "valeur de remplacement"]
    )

class Guarantee(BaseModel):
    """
    Represents a specific guarantee, capturing both its name from the main table
    and its detailed characteristics from the descriptive sections.
    """
    name: str = Field(
        description="The concise name of the guarantee or option.",
        examples=["Rééquipement à neuf", "Dépannage d'urgence", "Bris de vitre"]
    )
    description: Optional[str] = Field(
        default=None,
        description="The full, original descriptive paragraph for the guarantee."
    )
    covered_items: Optional[List[str]] = Field(
        default_factory=list,
        description="A list of specific goods, equipment, or locations covered by the guarantee.",
        examples=[["jardin", "arbres", "équipements"], ["piscines", "spas", "dômes", "moteurs"]]
    )
    covered_scenarios: Optional[List[str]] = Field(
        default_factory=list,
        description="A list of specific events, situations, or causes of damage that are covered.",
        examples=[
            "surtension", "foudre", "tempête", "catastrophe naturelle",
            "Porte claquée", "fuites", "panne électrique"
        ]
    )
    reimbursement_conditions: Optional[List[ReimbursementCondition]] = Field(
        default_factory=list,
        description="A structured list of specific conditions for reimbursement, such as item age."
    )

class InsurancePlan(BaseModel):
    """
    Represents an insurance package, categorizing guarantees by name.
    It links to the Guarantee nodes by name.
    """
    name: str = Field(
        description="The name of the insurance plan.",
        examples=["ESSENTIELLE", "CONFORT", "CONFORT PLUS"]
    )
    included_guarantees: Optional[List[str]] = Field(
        description="A list of the NAMES of guarantees explicitly INCLUDED in this plan, often marked with '✓'.",
        default_factory=list
    )
    optional_guarantees: Optional[List[str]] = Field(
        description="A list of the NAMES of optional add-on guarantees for this plan, often marked with 'En option'.",
        default_factory=list
    )
    excluded_guarantees: Optional[List[str]] = Field(
        description="A list of the NAMES of guarantees NOT available for this plan, often indicated by a blank space.",
        default_factory=list
    )

# --- Central Node with Edges ---

class HomeInsurance(BaseModel):
    """
    The central model for the insurance document. It contains all guarantees
    and all plans, forming a complete graph of the product offering.
    """
    product_name: str = Field(
        default="Assurance Habitation",
        description="The name of the insurance product line."
    )
    
    # This list contains the full DEFINITIONS of all guarantees.
    all_guarantees: List[Guarantee] = Edge(
        label="DEFINES_GUARANTEE",
        description="A comprehensive list of every unique guarantee and its structured details found on the page."
    )

    # This list defines the PLANS and links to the guarantees above by name.
    plans: List[InsurancePlan] = Edge(
        label="HAS_PLAN",
        description="A breakdown of each available insurance plan, linking to guarantees by name."
    )