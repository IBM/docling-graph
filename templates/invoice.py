"""
Pydantic models defining the schema for invoice data extraction.

These models include descriptions and concrete examples in each field to guide
the language model, improving the accuracy and consistency of the extracted data.
The schema is designed to be converted into a knowledge graph.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Any

# A special object to define graph edges
def Edge(label: str, **kwargs: Any) -> Any:
    """Helper function to create a Pydantic Field with edge metadata."""
    return Field(..., json_schema_extra={'edge_label': label}, **kwargs)

# --- Node Definitions ---

class Address(BaseModel):
    """Represents a physical address entity."""
    street: str = Field(
        description="Street name and number",
        examples=["Marktgasse 28", "Rue du Lac 1268"]
    )
    postal_code: str = Field(
        description="Postal or ZIP code",
        examples=["9400", "2501"]
    )
    city: str = Field(
        description="City or town name",
        examples=["Rorschach", "Biel"]
    )
    country: Optional[str] = Field(
        default=None,
        description="Country, preferably as a two-letter code.",
        examples=["CH"]
    )

class Organization(BaseModel):
    """Represents a company or organization entity."""
    name: str = Field(
        description="The legal name of the organization",
        examples=["Robert Schneider AG"]
    )
    phone: Optional[str] = Field(
        default=None,
        description="Contact phone number",
        examples=["059/987 6540"]
    )
    email: Optional[str] = Field(
        default=None,
        description="Contact email address",
        examples=["robert@rschneider.ch"]
    )
    website: Optional[str] = Field(
        default=None,
        description="Company website URL",
        examples=["www.rschneider.ch"]
    )
    
    # --- Edge Definition ---
    located_at: Address = Edge(label="LOCATED_AT")

class Person(BaseModel):
    """Represents an individual person entity."""
    name: str = Field(
        description="Full name of the person",
        examples=["Pia Rutschmann", "Robert Schneider"]
    )
    
    # --- Edge Definition ---
    lives_at: Address = Edge(label="LIVES_AT")

class LineItem(BaseModel):
    """Represents a single line item within the invoice."""
    description: str = Field(
        description="Description of the service or product",
        examples=["Garden work", "Disposal of cuttings"]
    )
    quantity: float = Field(
        description="The quantity of the item",
        examples=[28.0, 1.0]
    )
    unit: str = Field(
        description="The unit of measurement for the quantity",
        examples=["Std.", "pcs", "hours"]
    )
    unit_price: float = Field(
        description="The price per unit",
        examples=[120.00, 307.35]
    )
    total: float = Field(
        description="The total price for this line item (quantity * unit_price)",
        examples=[3360.00, 307.35]
    )

# --- Central Node with Edges ---

class Invoice(BaseModel):
    """The central node representing the entire invoice document."""
    bill_no: str = Field(
        description="The unique invoice identifier or bill number",
        examples=["3139"]
    )
    date: str = Field(
        description="Date the invoice was issued, preferably in YYYY-MM-DD format",
        examples=["01.07.2020"]
    )
    currency: str = Field(
        description="The currency of the invoice amounts (e.g., 'CHF', 'USD', 'EUR')",
        examples=["CHF"]
    )
    subtotal: float = Field(
        description="The total amount before tax or other fees",
        examples=[3667.35]
    )
    vat_rate: float = Field(
        description="The Value Added Tax rate as a percentage",
        examples=[7.7]
    )
    vat_amount: float = Field(
        description="The total amount of VAT charged",
        examples=[282.40]
    )
    total: float = Field(
        description="The final, total amount to be paid",
        examples=[3949.75]
    )

    # --- Edge Definitions ---
    issued_by: Organization = Edge(label="ISSUED_BY")
    sent_to: Person = Edge(label="SENT_TO")
    contains_items: List[LineItem] = Edge(label="CONTAINS_ITEM")

