"""
Unit tests for optional dependency management.
"""

from unittest.mock import Mock, patch

import pytest

from docling_graph.deps import OptionalDependency, DependencyStatus


class TestOptionalDependency:
    """Tests for OptionalDependency class."""

    def test_init_with_defaults(self):
        """Test initialization with default values."""
        dep = OptionalDependency(
            name="test_package",
            package="test_pkg"
        )
        
        assert dep.name == "test_package"
        assert dep.package == "test_pkg"
        assert dep.extra == "all"
        assert dep.inference_type is None

    def test_init_with_custom_extra(self):
        """Test initialization with custom extra."""
        dep = OptionalDependency(
            name="test_package",
            package="test_pkg",
            extra="gpu"
        )
        
        assert dep.extra == "gpu"

    def test_init_with_description(self):
        """Test initialization with description."""
        dep = OptionalDependency(
            name="test_package",
            package="test_pkg",
            description="Test LLM provider"
        )
        
        assert dep.description == "Test LLM provider"

    def test_default_description(self):
        """Test default description is generated."""
        dep = OptionalDependency(
            name="ollama",
            package="ollama"
        )
        
        assert "ollama" in dep.description.lower()

    def test_init_with_inference_type(self):
        """Test initialization with inference type."""
        dep = OptionalDependency(
            name="ollama",
            package="ollama",
            inference_type="local"
        )
        
        assert dep.inference_type == "local"


class TestOptionalDependencyStatus:
    """Tests for dependency status checking."""

    @patch("docling_graph.deps.importlib.util.find_spec")
    def test_is_installed_true(self, mock_find_spec):
        """Test is_installed returns True when found."""
        mock_find_spec.return_value = Mock()  # Package found
        
        dep = OptionalDependency(
            name="ollama",
            package="ollama"
        )
        
        assert dep.is_installed is True

    @patch("docling_graph.deps.importlib.util.find_spec")
    def test_is_installed_false(self, mock_find_spec):
        """Test is_installed returns False when not found."""
        mock_find_spec.return_value = None  # Package not found
        
        dep = OptionalDependency(
            name="ollama",
            package="ollama"
        )
        
        assert dep.is_installed is False

    @patch("docling_graph.deps.importlib.util.find_spec")
    def test_caches_status(self, mock_find_spec):
        """Test that status is cached after first check."""
        mock_find_spec.return_value = Mock()
        
        dep = OptionalDependency(
            name="ollama",
            package="ollama"
        )
        
        # First call
        status1 = dep.is_installed
        # Second call
        status2 = dep.is_installed
        
        # find_spec should only be called once
        assert mock_find_spec.call_count == 1
        assert status1 == status2 == True


class TestDependencyStatus:
    """Tests for DependencyStatus enum."""

    def test_status_values(self):
        """Test DependencyStatus values."""
        assert DependencyStatus.INSTALLED.value == "installed"
        assert DependencyStatus.NOT_INSTALLED.value == "not_installed"
        assert DependencyStatus.UNKNOWN.value == "unknown"

    def test_status_enum_members(self):
        """Test DependencyStatus has expected members."""
        assert hasattr(DependencyStatus, "INSTALLED")
        assert hasattr(DependencyStatus, "NOT_INSTALLED")
        assert hasattr(DependencyStatus, "UNKNOWN")


class TestOptionalDependencyComparison:
    """Tests for dependency comparison."""

    def test_dependency_equality_same_name(self):
        """Test dependencies with same name are comparable."""
        dep1 = OptionalDependency(name="ollama", package="ollama")
        dep2 = OptionalDependency(name="ollama", package="ollama")
        
        assert dep1.name == dep2.name
        assert dep1.package == dep2.package

    def test_dependency_inequality_different_name(self):
        """Test dependencies with different names."""
        dep1 = OptionalDependency(name="ollama", package="ollama")
        dep2 = OptionalDependency(name="mistral", package="mistralai")
        
        assert dep1.name != dep2.name


class TestOptionalDependencyValidation:
    """Tests for dependency validation."""

    def test_required_fields(self):
        """Test that required fields must be provided."""
        # This should work - required fields provided
        dep = OptionalDependency(
            name="test",
            package="test_pkg"
        )
        assert dep is not None

    def test_inference_types(self):
        """Test valid inference types."""
        dep_local = OptionalDependency(
            name="ollama",
            package="ollama",
            inference_type="local"
        )
        
        dep_remote = OptionalDependency(
            name="openai",
            package="openai",
            inference_type="remote"
        )
        
        assert dep_local.inference_type == "local"
        assert dep_remote.inference_type == "remote"

    def test_none_inference_type(self):
        """Test None as inference type."""
        dep = OptionalDependency(
            name="test",
            package="test",
            inference_type=None
        )
        
        assert dep.inference_type is None
