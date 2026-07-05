"""Tests for transcription engine logic (API calls mocked)."""

import pytest
from unittest.mock import MagicMock, patch
from lo6_transcriber.engine import OpenAIWhisperEngine


class TestOpenAIWhisperEngine:
    def test_transcribe_calls_openai(self):
        """Verify the engine calls the OpenAI API with correct args."""
        engine = OpenAIWhisperEngine(api_key="sk-test", model="whisper-1")

        mock_segment = MagicMock()
        mock_segment.start = 0.0
        mock_segment.end = 2.0
        mock_segment.text = "Hello world"

        mock_result = MagicMock()
        mock_result.text = "Hello world"
        mock_result.segments = [mock_segment]
        mock_result.language = "en"
        mock_result.duration = 2.0

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = mock_result

        with (
            patch("builtins.open", create=True),
            patch("openai.OpenAI", return_value=mock_client),
        ):
            result = engine.transcribe("test.mp3")

        assert result.text == "Hello world"
        assert result.language == "en"
        assert result.duration_seconds == 2.0
        assert len(result.segments) == 1
        assert result.segments[0].text == "Hello world"

    def test_transcribe_passes_language(self):
        """Verify the language parameter is forwarded."""
        engine = OpenAIWhisperEngine(api_key="sk-test")

        mock_result = MagicMock()
        mock_result.text = "Hola"
        mock_result.segments = []
        mock_result.language = "es"
        mock_result.duration = 1.0

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = mock_result

        with (
            patch("builtins.open", create=True),
            patch("openai.OpenAI", return_value=mock_client),
        ):
            result = engine.transcribe("test.mp3", language="es")

        call_kwargs = mock_client.audio.transcriptions.create.call_args[1]
        assert call_kwargs["language"] == "es"
        assert result.language == "es"


class TestLocalWhisperEngine:
    @pytest.mark.skipif(
        True,
        reason="openai-whisper requires large model downloads — test manually",
    )
    def test_transcribe_mock_local(self):
        from lo6_transcriber.engine import LocalWhisperEngine

        engine = LocalWhisperEngine(model="tiny")
        engine._model = MagicMock()
        engine._model.transcribe.return_value = {
            "text": "hello world",
            "segments": [{"start": 0.0, "end": 2.0, "text": "hello world"}],
            "language": "en",
            "duration": 2.0,
        }
        result = engine.transcribe("test.mp3")
        assert result.text == "hello world"
        assert result.language == "en"
        assert len(result.segments) == 1
