"""
Pydantic templates for French ID Card extraction.

These models include descriptions and concrete examples in each field to guide
the language model, improving the accuracy and consistency of the extracted data.
The schema is designed to be converted into a knowledge graph.
"""

from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing import Optional, Any, List
from datetime import date

# --- Edge Helper Function ---
def Edge(label: str, **kwargs: Any) -> Any:
    """
    Helper function to create a Pydantic Field with edge metadata.
    The 'edge_label' defines the type of relationship in the graph.
    """
    return Field(..., json_schema_extra={'edge_label': label}, **kwargs)

# --- Reusable Component: Address ---
class Address(BaseModel):
    """
    A flexible, generic model for a physical address.
    It's treated as a component, so it has no graph_id_fields.
    Its ID will be a hash of its content, making it unique to its context.
    """
    street_address: Optional[str] = Field(
        None,
        description="Street name and number",
        examples=["123 Rue de la Paix", "90 Boulevard Voltaire"]
    )
    city: Optional[str] = Field(
        None,
        description="City",
        examples=["Paris", "Lyon"]
    )
    state_or_province: Optional[str] = Field(
        None,
        description="State, province, or region",
        examples=["Île-de-France"]
    )
    postal_code: Optional[str] = Field(
        None,
        description="Postal or ZIP code",
        examples=["75001", "69002"]
    )
    country: Optional[str] = Field(
        None,
        description="Country",
        examples=["France"]
    )

    def __str__(self):
        parts = [self.street_address, self.city, self.state_or_province, self.postal_code, self.country]
        return ", ".join(p for p in parts if p)

# --- Reusable Entity: Person ---
class Person(BaseModel):
    """
    A generic model for a person.
    A person is uniquely identified by their full name and date of birth.
    """
    model_config = ConfigDict(graph_id_fields=['given_names', 'last_name', 'date_of_birth'])
    
    given_names: Optional[List[str]] = Field(
        default=None,
        description="List of given names (first names usually seperated with a comma) of the person, in order",
        examples=[["Pierre"], ["Pierre", "Louis"], ["Pierre", "Louis", "André"]]
    )
    last_name: Optional[str] = Field(
        None,
        description="The person's family name (surname)",
        examples=["Dupont", "Martin"]
    )
    alternate_name: Optional[str] = Field(
        None,
        description="The person's alterante name",
        examples=["Doe", "MJ"]
    )
    date_of_birth: Optional[date] = Field(
        None,
        description="Date of birth in YYYY-MM-DD format",
        examples=["1990-05-15"]
    )
    place_of_birth: Optional[str] = Field(
        None,
        description="City and/or country of birth",
        examples=["Paris", "Marseille (France)"]
    )
    gender: Optional[str] = Field(
        None,
        description="Gender or sex of the person",
        examples=["F", "M", "Female", "Male"]
    )
    
    # --- Edge Definition ---
    lives_at: Optional[Address] = Edge(
        label="LIVES_AT",
        description="Physical address (e.g., home address)"
    )

    # --- Validator ---
    @field_validator('given_names', mode='before')
    def ensure_list(cls, v):
        """Ensure given_names is always a list."""
        if isinstance(v, str):
            return [v]
        return v
    
    def __str__(self):
        parts = [self.first_name, self.last_name]
        return " ".join(p for p in parts if p)

# --- Root Document Model: IDCard ---
class IDCard(BaseModel):
    """
    A model for an identification document.
    It is uniquely identified by its document number.
    """
    model_config = ConfigDict(graph_id_fields=['document_number'])
    
    document_type: str = Field(
        "ID Card",
        description="Type of document (e.g., ID Card, Passport, Driver's License)",
        examples=["Carte Nationale d'Identité", "Passeport"]
    )
    document_number: str = Field(
        ...,
        description="The unique identifier for the document",
        examples=["23AB12345", "19XF56789"]
    )
    issuing_country: Optional[str] = Field(
        None,
        description="The country that issued the document (e.g., 'France', 'République Française')",
        examples=["France", "USA"]
    )
    issue_date: Optional[date] = Field(
        None,
        description=(
            "Date the document was issued, in DD-MM-YYYY format",
            "Look for text like 'Date of Issue', 'Issued on', 'Délivré le', or similar"
        ),
        examples=["20-01-2023", "23.12.2019", "05 07 2031"]
    )
    expiry_date: Optional[date] = Field(
        None,
        description=(
            "Date the document expires, in DD-MM-YYYY format",
            "Look for text like 'Expiry Date', 'Expires on', 'Valable jusqu’au', or similar"
        ),
        examples=["19-01-2033", "22.12.2029", "04 07 2041"]
    )

    # --- Edge Definition ---
    holder: Person = Edge(
        label="BELONGS_TO",
        description="The person this ID card belongs to"
    )

    def __str__(self):
        return f"{self.document_type} {self.document_number}"
