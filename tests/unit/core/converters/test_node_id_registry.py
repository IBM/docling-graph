import pytest
from pydantic import BaseModel

from docling_graph.core.converters.node_id_registry import NodeIDRegistry


class PersonModel(BaseModel):
    name: str
    age: int

    model_config = {"graph_id_fields": ["name"]}


class CompanyModel(BaseModel):
    name: str
    location: str

    model_config = {"graph_id_fields": ["name", "location"]}


@pytest.fixture
def registry():
    """Returns a clean NodeIDRegistry instance for each test."""
    return NodeIDRegistry()


def test_registry_init(registry):
    """Test that the registry initializes with empty structures."""
    assert registry.fingerprint_to_id == {}
    assert registry.id_to_fingerprint == {}
    assert registry.seen_classes == {}


def test_get_node_id_new_item(registry):
    """Test registering a new item."""
    person = PersonModel(name="Alice", age=30)
    node_id = registry.get_node_id(person)

    assert node_id.startswith("PersonModel_")
    assert len(node_id) > len("PersonModel_")


def test_get_node_id_existing_item(registry):
    """Test that registering the same item returns the same ID."""
    person1 = PersonModel(name="Alice", age=30)
    node_id_1 = registry.get_node_id(person1)

    # Same name (identity field), different age
    person2 = PersonModel(name="Alice", age=35)
    node_id_2 = registry.get_node_id(person2)

    # Should return the same ID since identity is based on 'name' only
    assert node_id_1 == node_id_2


def test_get_node_id_different_items(registry):
    """Test that different items get different IDs."""
    person1 = PersonModel(name="Alice", age=30)
    person2 = PersonModel(name="Bob", age=30)

    node_id_1 = registry.get_node_id(person1)
    node_id_2 = registry.get_node_id(person2)

    assert node_id_1 != node_id_2


def test_get_node_id_multiple_identity_fields(registry):
    """Test with multiple identity fields."""
    company1 = CompanyModel(name="Acme Inc.", location="NY")
    company2 = CompanyModel(name="Acme Inc.", location="LA")

    node_id_1 = registry.get_node_id(company1)
    node_id_2 = registry.get_node_id(company2)

    # Different locations should result in different IDs
    assert node_id_1 != node_id_2


def test_register_batch(registry):
    """Test batch registration."""
    person1 = PersonModel(name="Alice", age=30)
    person2 = PersonModel(name="Bob", age=25)

    registry.register_batch([person1, person2])

    stats = registry.get_stats()
    assert stats["total_entities"] == 2
    assert "PersonModel" in stats["classes"]


def test_get_stats(registry):
    """Test getting registry statistics."""
    person = PersonModel(name="Alice", age=30)
    company = CompanyModel(name="Acme Inc.", location="NY")

    registry.get_node_id(person)
    registry.get_node_id(company)

    stats = registry.get_stats()

    assert stats["total_entities"] == 2
    assert len(stats["classes"]) == 2
    assert "PersonModel" in stats["classes"]
    assert "CompanyModel" in stats["classes"]


def test_deterministic_ids(registry):
    """Test that IDs are deterministic across registry instances."""
    person1 = PersonModel(name="Alice", age=30)

    registry1 = NodeIDRegistry()
    node_id_1 = registry1.get_node_id(person1)

    registry2 = NodeIDRegistry()
    node_id_2 = registry2.get_node_id(person1)

    # Same model should produce same ID across different registries
    assert node_id_1 == node_id_2
