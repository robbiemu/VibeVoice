import unittest
from unittest.mock import patch, MagicMock
import torch

from vibevoice.model import load_vibevoice_model


class TestModelLoading(unittest.TestCase):
    @patch("vibevoice.model.get_optimal_config")
    @patch("vibevoice.model.VibeVoiceProcessor.from_pretrained")
    @patch("vibevoice.model.VibeVoiceForConditionalGenerationInference.from_pretrained")
    def test_load_model_auto_config(
        self,
        mock_model_from_pretrained,
        mock_processor_from_pretrained,
        mock_get_optimal_config,
    ):
        # Arrange
        mock_model = MagicMock()
        mock_processor = MagicMock()
        mock_model_from_pretrained.return_value = mock_model
        mock_processor_from_pretrained.return_value = mock_processor
        mock_get_optimal_config.return_value = (
            "cuda",
            torch.float16,
            "flash_attention_2",
        )

        model_name = "test-model"

        # Act
        model, processor = load_vibevoice_model(model_name, use_compile=False) # Disable compile for this base test

        # Assert
        mock_get_optimal_config.assert_called_once()
        mock_model_from_pretrained.assert_called_once_with(
            model_name,
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2",
        )
        mock_processor_from_pretrained.assert_called_once_with(model_name)
        mock_model.to.assert_called_once_with(torch.device("cuda"))
        self.assertIsNotNone(model)
        self.assertIsNotNone(processor)

    @patch("vibevoice.model.get_optimal_config")
    @patch("vibevoice.model.VibeVoiceProcessor.from_pretrained")
    @patch("vibevoice.model.VibeVoiceForConditionalGenerationInference.from_pretrained")
    def test_load_model_explicit_args(
        self,
        mock_model_from_pretrained,
        mock_processor_from_pretrained,
        mock_get_optimal_config,
    ):
        # Arrange
        mock_model = MagicMock()
        mock_processor = MagicMock()
        mock_model_from_pretrained.return_value = mock_model
        mock_processor_from_pretrained.return_value = mock_processor
        mock_get_optimal_config.return_value = (
            "cuda",
            torch.float16,
            "flash_attention_2",
        )

        model_name = "test-model"
        device = "cpu"
        dtype = torch.float32
        attn_impl = "sdpa"

        # Act
        model, processor = load_vibevoice_model(
            model_name, device=device, torch_dtype=dtype, attn_implementation=attn_impl, use_compile=False
        )

        # Assert
        mock_get_optimal_config.assert_called_once()
        mock_model_from_pretrained.assert_called_once_with(
            model_name, torch_dtype=dtype, attn_implementation=attn_impl
        )
        mock_processor_from_pretrained.assert_called_once_with(model_name)
        mock_model.to.assert_called_once_with(torch.device(device))
        self.assertIsNotNone(model)
        self.assertIsNotNone(processor)

    @patch("torch.compile")
    @patch("vibevoice.model.get_optimal_config")
    @patch("vibevoice.model.VibeVoiceProcessor.from_pretrained")
    @patch("vibevoice.model.VibeVoiceForConditionalGenerationInference.from_pretrained")
    def test_load_model_with_compile_disabled(
        self,
        mock_model_from_pretrained,
        mock_processor_from_pretrained,
        mock_get_optimal_config,
        mock_torch_compile,
    ):
        # Arrange
        mock_model = MagicMock()
        mock_processor = MagicMock()
        mock_model_from_pretrained.return_value = mock_model
        mock_processor_from_pretrained.return_value = mock_processor
        mock_get_optimal_config.return_value = ("cuda", torch.float16, "flash_attention_2")
        model_name = "test-model"

        # Act
        model, processor = load_vibevoice_model(model_name, use_compile=False)

        # Assert
        mock_torch_compile.assert_not_called()
        self.assertIsNotNone(model)
        self.assertIsNotNone(processor)

    @patch("torch.compile")
    @patch("vibevoice.model.get_optimal_config")
    @patch("vibevoice.model.VibeVoiceProcessor.from_pretrained")
    @patch("vibevoice.model.VibeVoiceForConditionalGenerationInference.from_pretrained")
    def test_load_model_with_compile_cuda(
        self,
        mock_model_from_pretrained,
        mock_processor_from_pretrained,
        mock_get_optimal_config,
        mock_torch_compile,
    ):
        # Arrange
        mock_model = MagicMock()
        mock_compiled_model = MagicMock()
        mock_processor = MagicMock()
        mock_model_from_pretrained.return_value = mock_model
        mock_torch_compile.return_value = mock_compiled_model
        mock_processor_from_pretrained.return_value = mock_processor
        mock_get_optimal_config.return_value = ("cuda", torch.float16, "flash_attention_2")
        model_name = "test-model"

        # Act
        model, processor = load_vibevoice_model(model_name, device="cuda", use_compile=True)

        # Assert
        mock_torch_compile.assert_called_once_with(mock_model, backend="inductor", mode="reduce-overhead")
        self.assertEqual(model, mock_compiled_model)
        self.assertIsNotNone(processor)

    @patch("torch.compile")
    @patch("vibevoice.model.get_optimal_config")
    @patch("vibevoice.model.VibeVoiceProcessor.from_pretrained")
    @patch("vibevoice.model.VibeVoiceForConditionalGenerationInference.from_pretrained")
    def test_load_model_with_compile_mps(
        self,
        mock_model_from_pretrained,
        mock_processor_from_pretrained,
        mock_get_optimal_config,
        mock_torch_compile,
    ):
        # Arrange
        mock_model = MagicMock()
        mock_compiled_model = MagicMock()
        mock_processor = MagicMock()
        mock_model_from_pretrained.return_value = mock_model
        mock_torch_compile.return_value = mock_compiled_model
        mock_processor_from_pretrained.return_value = mock_processor
        mock_get_optimal_config.return_value = ("mps", torch.float16, "sdpa")
        model_name = "test-model"

        # Act
        model, processor = load_vibevoice_model(model_name, device="mps", use_compile=True)

        # Assert
        mock_torch_compile.assert_called_once_with(mock_model, mode="reduce-overhead")
        self.assertEqual(model, mock_compiled_model)
        self.assertIsNotNone(processor)

    @patch("warnings.warn")
    @patch("torch.compile")
    @patch("vibevoice.model.get_optimal_config")
    @patch("vibevoice.model.VibeVoiceProcessor.from_pretrained")
    @patch("vibevoice.model.VibeVoiceForConditionalGenerationInference.from_pretrained")
    def test_load_model_with_compile_failure(
        self,
        mock_model_from_pretrained,
        mock_processor_from_pretrained,
        mock_get_optimal_config,
        mock_torch_compile,
        mock_warn,
    ):
        # Arrange
        mock_model = MagicMock()
        mock_processor = MagicMock()
        mock_model_from_pretrained.return_value = mock_model
        mock_processor_from_pretrained.return_value = mock_processor
        mock_get_optimal_config.return_value = ("cuda", torch.float16, "flash_attention_2")
        mock_torch_compile.side_effect = Exception("Compilation failed")
        model_name = "test-model"

        # Act
        model, processor = load_vibevoice_model(model_name, device="cuda", use_compile=True)

        # Assert
        mock_torch_compile.assert_called_once()
        mock_warn.assert_called_once()
        self.assertEqual(model, mock_model) # Should return the original model
        self.assertIsNotNone(processor)


if __name__ == "__main__":
    unittest.main()
