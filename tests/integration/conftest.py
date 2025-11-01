"""
Pytest configuration and fixtures for integration tests.
"""

import json
from pathlib import Path
from typing import Dict, Any

import pytest
import networkx as nx
from click.testing import CliRunner
from pydantic import BaseModel, Field


# ============================================================================
# SAMPLE MODELS
# ============================================================================

class Person(BaseModel):
    """Sample Person model."""
    name: str = Field(..., description="Full name")
    email: str = Field(..., description="Email address")
    age: int = Field(..., description="Age in years")


class Invoice(BaseModel):
    """Sample Invoice model."""
    invoice_number: str = Field(..., description="Invoice ID")
    date: str = Field(..., description="Invoice date")
    customer: str = Field(..., description="Customer name")
    amount: float = Field(..., description="Total amount")
    status: str = Field(default="pending", description="Payment status")


class Document(BaseModel):
    """Sample Document model."""
    title: str = Field(..., description="Document title")
    content: str = Field(..., description="Document content")
    author: str = Field(..., description="Document author")
    tags: list = Field(default_factory=list, description="Document tags")


# ============================================================================
# FIXTURES: TEST DATA FILES
# ============================================================================

@pytest.fixture
def sample_pdf(temp_dir):
    """Create a minimal sample PDF for integration tests."""
    pdf_path = temp_dir / "sample.pdf"
    pdf_content = (
        b"%PDF-1.4\n"
        b"1 0 obj\n"
        b"<< /Type /Catalog /Pages 2 0 R >>\n"
        b"endobj\n"
        b"2 0 obj\n"
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>\n"
        b"endobj\n"
        b"3 0 obj\n"
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\n"
        b"endobj\n"
        b"xref\n"
        b"0 4\n"
        b"0000000000 65535 f\n"
        b"0000000009 00000 n\n"
        b"0000000058 00000 n\n"
        b"0000000115 00000 n\n"
        b"trailer\n"
        b"<< /Size 4 /Root 1 0 R >>\n"
        b"startxref\n"
        b"190\n"
        b"%%EOF"
    )
    pdf_path.write_bytes(pdf_content)
    return pdf_path


@pytest.fixture
def sample_text_document(temp_dir):
    """Create a sample text document."""
    doc_path = temp_dir / "document.txt"
    content = """
    INVOICE #INV-2024-001
    Date: 2024-01-15
    Customer: John Doe
    Amount: $1,500.00
    Status: Paid
    
    Items:
    - Item 1: $500.00
    - Item 2: $1,000.00
    """
    doc_path.write_text(content)
    return doc_path


@pytest.fixture
def sample_markdown_document(temp_dir):
    """Create a sample markdown document."""
    doc_path = temp_dir / "document.md"
    content = """
    # Sample Document
    
    ## Invoice Information
    
    - Invoice Number: INV-2024-001
    - Date: 2024-01-15
    - Customer: Jane Smith
    - Amount: $2,500.00
    - Status: Pending
    
    ## Items
    
    1. Item A: $1,500.00
    2. Item B: $1,000.00
    """
    doc_path.write_text(content)
    return doc_path


# ============================================================================
# FIXTURES: SAMPLE GRAPHS
# ============================================================================

@pytest.fixture
def sample_graph():
    """Create a sample graph for integration tests."""
    graph = nx.DiGraph()
    
    # Add nodes with attributes
    graph.add_node("person1", label="Person", name="Alice", age=30, email="alice@example.com")
    graph.add_node("person2", label="Person", name="Bob", age=28, email="bob@example.com")
    graph.add_node("org1", label="Organization", name="TechCorp", industry="Tech")
    graph.add_node("org2", label="Organization", name="DataCorp", industry="Data")
    graph.add_node("invoice1", label="Invoice", number="INV-001", amount=5000)
    
    # Add edges with attributes
    graph.add_edge("person1", "org1", label="works_for", role="Engineer")
    graph.add_edge("person2", "org1", label="works_for", role="Manager")
    graph.add_edge("org1", "org2", label="partners_with", since=2020)
    graph.add_edge("person1", "invoice1", label="created", date="2024-01-01")
    graph.add_edge("org1", "invoice1", label="issues")
    
    return graph


@pytest.fixture
def complex_graph():
    """Create a complex graph with many nodes and edges."""
    graph = nx.DiGraph()
    
    # Add multiple people
    for i in range(10):
        graph.add_node(
            f"person{i}",
            label="Person",
            name=f"Person {i}",
            age=20+i,
            email=f"person{i}@example.com"
        )
    
    # Add organizations
    for i in range(3):
        graph.add_node(
            f"org{i}",
            label="Organization",
            name=f"Organization {i}",
            industry="Tech"
        )
    
    # Add relationships
    for i in range(10):
        graph.add_edge(f"person{i}", f"org{i % 3}", label="works_for")
    
    # Add inter-organization relationships
    for i in range(2):
        graph.add_edge(f"org{i}", f"org{i+1}", label="partners_with")
    
    return graph


@pytest.fixture
def person_graph(sample_graph):
    """Create a graph with Person nodes."""
    graph = nx.DiGraph()
    
    for i in range(5):
        graph.add_node(
            f"person{i}",
            label="Person",
            name=f"Person {i}",
            age=25+i,
            email=f"person{i}@test.com"
        )
    
    # Add some relationships
    for i in range(4):
        graph.add_edge(f"person{i}", f"person{i+1}", label="knows")
    
    return graph


# ============================================================================
# FIXTURES: CONFIGURATION FILES
# ============================================================================

@pytest.fixture
def config_file_minimal(temp_dir):
    """Create a minimal configuration file."""
    config = {
        "source": str(temp_dir / "sample.pdf"),
        "template": "tests.integration.conftest.Person",
        "processing_mode": "one-to-one",
        "backend_type": "llm",
    }
    config_path = temp_dir / "config_minimal.yaml"
    import yaml
    config_path.write_text(yaml.dump(config))
    return config_path


@pytest.fixture
def config_file_full(temp_dir):
    """Create a full configuration file with all options."""
    config = {
        "source": str(temp_dir / "sample.pdf"),
        "template": "tests.integration.conftest.Invoice",
        "processing_mode": "many-to-one",
        "backend_type": "llm",
        "inference": "local",
        "model": "llama3",
        "export_format": "csv",
        "output_dir": str(temp_dir / "outputs"),
        "docling_config": "ocr",
        "reverse_edges": False,
    }
    config_path = temp_dir / "config_full.yaml"
    import yaml
    config_path.write_text(yaml.dump(config))
    return config_path


# ============================================================================
# FIXTURES: EXPORTED DATA
# ============================================================================

@pytest.fixture
def sample_csv_content():
    """Sample CSV content for export validation."""
    return """id,label,name,email
person1,Person,Alice,alice@example.com
person2,Person,Bob,bob@example.com
org1,Organization,TechCorp,
"""


@pytest.fixture
def sample_json_content():
    """Sample JSON content for export validation."""
    return {
        "nodes": [
            {"id": "person1", "label": "Person", "name": "Alice", "email": "alice@example.com"},
            {"id": "org1", "label": "Organization", "name": "TechCorp"},
        ],
        "edges": [
            {"source": "person1", "target": "org1", "label": "works_for"},
        ],
        "metadata": {
            "node_count": 2,
            "edge_count": 1,
        }
    }


@pytest.fixture
def sample_cypher_content():
    """Sample Cypher script content."""
    return """// Cypher script generated by docling-graph
CREATE (person1:Person {id: 'person1', name: 'Alice', email: 'alice@example.com'})
CREATE (org1:Organization {id: 'org1', name: 'TechCorp'})
CREATE (person1)-[:works_for]->(org1)
"""


# ============================================================================
# FIXTURES: EXTRACTION RESULTS
# ============================================================================

@pytest.fixture
def extraction_result_one_to_one():
    """Sample extraction result from OneToOne strategy."""
    return [
        Person(name="Alice Johnson", email="alice@example.com", age=30),
        Person(name="Bob Smith", email="bob@example.com", age=28),
    ]


@pytest.fixture
def extraction_result_many_to_one():
    """Sample extraction result from ManyToOne strategy."""
    return [
        Invoice(
            invoice_number="INV-001",
            date="2024-01-15",
            customer="Acme Corp",
            amount=5000.0,
            status="paid"
        )
    ]


# ============================================================================
# FIXTURES: BENCHMARK DATA
# ============================================================================

@pytest.fixture(scope="session")
def benchmark_data():
    """Provide data for performance benchmarking."""
    return {
        "small_graph": _create_small_graph(),
        "medium_graph": _create_medium_graph(),
        "large_graph": _create_large_graph(),
    }


def _create_small_graph():
    """Create a small graph (10 nodes)."""
    g = nx.DiGraph()
    for i in range(10):
        g.add_node(f"node{i}", label="Node", value=i)
    for i in range(9):
        g.add_edge(f"node{i}", f"node{i+1}", weight=1.0)
    return g


def _create_medium_graph():
    """Create a medium graph (100 nodes)."""
    g = nx.DiGraph()
    for i in range(100):
        g.add_node(f"node{i}", label="Node", value=i)
    for i in range(99):
        g.add_edge(f"node{i}", f"node{i+1}", weight=1.0)
    return g


def _create_large_graph():
    """Create a large graph (1000 nodes)."""
    g = nx.DiGraph()
    for i in range(1000):
        g.add_node(f"node{i}", label="Node", value=i)
    for i in range(999):
        g.add_edge(f"node{i}", f"node{i+1}", weight=1.0)
    return g


# ============================================================================
# FIXTURES: MOCK DATA
# ============================================================================

@pytest.fixture
def mock_extraction_data():
    """Mock extraction data."""
    return {
        "Person": [
            {"name": "Alice", "email": "alice@example.com", "age": 30},
            {"name": "Bob", "email": "bob@example.com", "age": 28},
        ],
        "Invoice": [
            {"invoice_number": "INV-001", "date": "2024-01-15", "customer": "Corp A", "amount": 5000},
        ],
        "Document": [
            {"title": "Report", "content": "Content here", "author": "John", "tags": ["report", "2024"]},
        ]
    }

@pytest.fixture
def cli_runner():
    """Provide Click CLI runner for CLI tests."""
    return CliRunner()

@pytest.fixture
def mock_backend(mocker):
    """Mock LLM backend."""
    backend = mocker.Mock()
    backend.extract_from_document = mocker.Mock(return_value=[])
    backend.cleanup = mocker.Mock()
    return backend

@pytest.fixture
def mock_vlm_backend(mocker):
    """Mock VLM backend."""
    backend = mocker.Mock()
    backend.extract_from_document = mocker.Mock(return_value=[])
    backend.cleanup = mocker.Mock()
    return backend