"""
Unit tests for init command module.
"""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch, mock_open

import pytest
import typer
import yaml

from docling_graph.cli.commands.init import init_command


class TestInitCommand:
    """Tests for init_command function."""

    @patch("docling_graph.cli.commands.init.print_next_steps_with_deps")
    @patch("docling_graph.cli.commands.init.print_next_steps")
    @patch("docling_graph.cli.commands.init.save_config")
    @patch("docling_graph.cli.commands.init.validate_and_warn_dependencies")
    @patch("docling_graph.cli.commands.init.build_config_interactive")
    @patch("docling_graph.cli.commands.init.rich_print")
    def test_init_command_success(
        self,
        mock_print,
        mock_build,
        mock_validate,
        mock_save,
        mock_print_steps,
        mock_print_deps,
        temp_dir,
        monkeypatch,
    ):
        """Test successful initialization."""
        monkeypatch.chdir(temp_dir)

        config_dict = {
            "defaults": {
                "processing_mode": "many-to-one",
                "backend_type": "llm",
                "inference": "local",
                "export_format": "csv",
            },
            "docling": {"pipeline": "ocr"},
            "models": {},
            "output": {},
        }

        mock_build.return_value = config_dict
        mock_validate.return_value = True

        init_command()

        # Verify config was saved
        mock_save.assert_called_once()
        assert mock_save.call_args[0][0] == config_dict

        # Verify next steps were printed
        mock_print_steps.assert_called_once_with(config_dict)
        mock_print_deps.assert_called_once_with(config_dict)

    @patch("docling_graph.cli.commands.init.typer.confirm")
    @patch("docling_graph.cli.commands.init.rich_print")
    def test_init_command_file_exists_cancel(self, mock_print, mock_confirm, temp_dir, monkeypatch):
        """Test initialization when config exists and user cancels."""
        monkeypatch.chdir(temp_dir)

        # Create existing config
        config_path = temp_dir / "config.yaml"
        config_path.write_text("existing: config")

        mock_confirm.return_value = False

        init_command()

        # Verify message about cancellation
        calls = [str(call) for call in mock_print.call_args_list]
        assert any("cancelled" in str(call).lower() for call in calls)

    @patch("docling_graph.cli.commands.init.print_next_steps_with_deps")
    @patch("docling_graph.cli.commands.init.print_next_steps")
    @patch("docling_graph.cli.commands.init.save_config")
    @patch("docling_graph.cli.commands.init.validate_and_warn_dependencies")
    @patch("docling_graph.cli.commands.init.typer.confirm")
    @patch("docling_graph.cli.commands.init.build_config_interactive")
    @patch("docling_graph.cli.commands.init.rich_print")
    def test_init_command_file_exists_overwrite(
        self,
        mock_print,
        mock_build,
        mock_confirm,
        mock_validate,
        mock_save,
        mock_print_steps,
        mock_print_deps,
        temp_dir,
        monkeypatch,
    ):
        """Test initialization with overwrite confirmation."""
        monkeypatch.chdir(temp_dir)

        # Create existing config
        config_path = temp_dir / "config.yaml"
        config_path.write_text("existing: config")

        mock_confirm.return_value = True
        mock_build.return_value = {"defaults": {}, "models": {}}
        mock_validate.return_value = True

        init_command()

        # Verify config was saved
        mock_save.assert_called_once()

    @patch("docling_graph.cli.commands.init.yaml.safe_load")
    @patch("docling_graph.cli.commands.init.print_next_steps_with_deps")
    @patch("docling_graph.cli.commands.init.print_next_steps")
    @patch("docling_graph.cli.commands.init.save_config")
    @patch("docling_graph.cli.commands.init.validate_and_warn_dependencies")
    @patch("docling_graph.cli.commands.init.build_config_interactive")
    @patch("docling_graph.cli.commands.init.rich_print")
    @patch("builtins.open", new_callable=mock_open, read_data="defaults: {}")
    def test_init_command_non_interactive_with_template(
        self,
        mock_file,
        mock_print,
        mock_build,
        mock_validate,
        mock_save,
        mock_print_steps,
        mock_print_deps,
        mock_yaml_load,
        temp_dir,
        monkeypatch,
    ):
        """Test initialization in non-interactive environment with template."""
        monkeypatch.chdir(temp_dir)

        # Simulate non-interactive environment
        mock_build.side_effect = EOFError()
        mock_yaml_load.return_value = {
            "defaults": {"processing_mode": "many-to-one"},
            "models": {},
        }
        mock_validate.return_value = True

        init_command()

        # Verify fallback to template was used
        calls = [str(call) for call in mock_print.call_args_list]
        assert any("default" in str(call).lower() for call in calls)

    @patch("docling_graph.cli.commands.init.print_next_steps_with_deps")
    @patch("docling_graph.cli.commands.init.print_next_steps")
    @patch("docling_graph.cli.commands.init.save_config")
    @patch("docling_graph.cli.commands.init.validate_and_warn_dependencies")
    @patch("docling_graph.cli.commands.init.build_config_interactive")
    @patch("docling_graph.cli.commands.init.rich_print")
    def test_init_command_with_dependency_warnings(
        self,
        mock_print,
        mock_build,
        mock_validate,
        mock_save,
        mock_print_steps,
        mock_print_deps,
        temp_dir,
        monkeypatch,
    ):
        """Test initialization with dependency warnings."""
        monkeypatch.chdir(temp_dir)

        config_dict = {"defaults": {}, "models": {}}
        mock_build.return_value = config_dict
        mock_validate.return_value = False  # Dependencies missing

        init_command()

        # Verify warning was printed
        calls = [str(call) for call in mock_print.call_args_list]
        assert any("dependencies" in str(call).lower() for call in calls)

    @patch("docling_graph.cli.commands.init.save_config")
    @patch("docling_graph.cli.commands.init.validate_and_warn_dependencies")
    @patch("docling_graph.cli.commands.init.build_config_interactive")
    @patch("docling_graph.cli.commands.init.rich_print")
    def test_init_command_save_error(
        self, mock_print, mock_build, mock_validate, mock_save, temp_dir, monkeypatch
    ):
        """Test initialization when save fails."""
        monkeypatch.chdir(temp_dir)

        mock_build.return_value = {"defaults": {}}
        mock_validate.return_value = True
        mock_save.side_effect = IOError("Cannot write file")

        with pytest.raises(typer.Exit) as exc_info:
            init_command()

        assert exc_info.value.exit_code == 1

    @patch("docling_graph.cli.commands.init.build_config_interactive")
    @patch("docling_graph.cli.commands.init.rich_print")
    def test_init_command_build_error(self, mock_print, mock_build, temp_dir, monkeypatch):
        """Test initialization when config building fails."""
        monkeypatch.chdir(temp_dir)

        mock_build.side_effect = Exception("Build failed")

        with pytest.raises(typer.Exit) as exc_info:
            init_command()

        assert exc_info.value.exit_code == 1
