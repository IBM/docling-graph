"""
Pydantic schema for extracting information from an Identity Card.

This schema defines the structure for capturing personal details and the document's
metadata, representing it as a graph with a central 'IdentityCard' node linked
to a 'Person' node, which in turn is linked to a 'Location' node for the place of birth.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Any

# A special object to define graph edges, consistent with other templates
def Edge(label: str, **kwargs: Any) -> Any:
    """Helper function to create a Pydantic Field with edge metadata."""
    return Field(..., json_schema_extra={'edge_label': label}, **kwargs)


# --- Node Definitions ---

class Location(BaseModel):
    """
    Represents a geographical location, such as a city of birth.
    """
    city: str = Field(
        description="The city, town, or municipality.",
        examples=["BORDEAUX"]
    )

class Person(BaseModel):
    """
    Represents an individual person, the holder of the identity card.
    """
    surname: str = Field(
        description="The last name or family name of the person.",
        examples=["CHEVAILLIER"]
    )
    given_names: List[str] = Field(
        description="A list of the person's first and middle names.",
        examples=[["Gisèle", "Audrey"]]
    )
    sex: str = Field(
        description="The legal sex of the person (e.g., 'F' or 'M').",
        examples=["F"]
    )
    nationality: str = Field(
        description="The nationality of the person, often as a three-letter code.",
        examples=["FRA"]
    )
    date_of_birth: str = Field(
        description="The person's date of birth in DD MM YYYY format.",
        examples=["01 04 1995"]
    )
    alternate_name: Optional[str] = Field(
        default=None,
        description="An alternate or usage name, such as a married name (Nom d'usage).",
        examples=["vve. DUBOIS"]
    )

    # --- Edge Definition ---
    place_of_birth: Location = Edge(label="BORN_IN")


class IdentityCard(BaseModel):
    """
    The central node representing the physical identity document.
    """
    issuing_country: str = Field(
        description="The country that issued the identity card.",
        examples=["RÉPUBLIQUE FRANÇAISE"]
    )
    document_type: str = Field(
        description="The official type of the document.",
        examples=["CARTE NATIONALE D'IDENTITÉ"]
    )
    document_number: str = Field(
        description="The unique alphanumeric identifier for the document.",
        examples=["T7X62TZ79"]
    )
    expiry_date: str = Field(
        description="The date the document expires in DD MM YYYY format.",
        examples=["27 01 2031"]
    )
    card_access_number: Optional[str] = Field(
        default=None,
        description="A secondary number or code on the card, possibly for electronic access.",
        examples=["240220"]
    )

    # --- Edge Definition ---
    holder: Person = Edge(label="HELD_BY")
