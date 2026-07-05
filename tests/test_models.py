"""Tests for rvrb-transcriber models and engine."""

from rvrb_transcriber import Transcript, Segment


class TestSegment:
    def test_create_segment(self):
        seg = Segment(start=0.0, end=2.5, text="Hello world")
        assert seg.start == 0.0
        assert seg.end == 2.5
        assert seg.text == "Hello world"
        assert seg.duration == 2.5

    def test_timedelta_properties(self):
        seg = Segment(start=10.5, end=65.3, text="test")
        assert seg.start_timedelta.total_seconds() == 10.5
        assert seg.end_timedelta.total_seconds() == 65.3


class TestTranscript:
    def test_create_empty(self):
        t = Transcript(text="")
        assert t.text == ""
        assert t.segments == []
        assert t.language == "unknown"
        assert t.duration_seconds == 0.0

    def test_create_with_segments(self):
        segs = [
            Segment(start=0.0, end=2.0, text="Hello"),
            Segment(start=2.0, end=4.0, text="world"),
        ]
        t = Transcript(
            text="Hello world",
            segments=segs,
            language="en",
            duration_seconds=4.0,
        )
        assert len(t.segments) == 2
        assert t.language == "en"
        assert t.duration_seconds == 4.0

    def test_to_srt(self):
        segs = [
            Segment(start=0.0, end=2.5, text="Hello"),
            Segment(start=2.5, end=5.0, text="world"),
        ]
        t = Transcript(text="Hello world", segments=segs)
        srt = t.to_srt()
        assert "00:00:00,000 --> 00:00:02,500" in srt
        assert "Hello" in srt
        assert "00:00:02,500 --> 00:00:05,000" in srt
        assert "world" in srt

    def test_to_vtt(self):
        segs = [Segment(start=0.0, end=3.0, text="Test")]
        t = Transcript(text="Test", segments=segs)
        vtt = t.to_vtt()
        assert vtt.startswith("WEBVTT")
        assert "00:00:00.000 --> 00:00:03.000" in vtt
        assert "Test" in vtt

    def test_to_srt_empty_segments(self):
        t = Transcript(text="No segments")
        assert t.to_srt() == ""

    def test_json_serialization(self):
        segs = [Segment(start=0.0, end=1.0, text="Hi")]
        t = Transcript(text="Hi", segments=segs, language="en", duration_seconds=1.0)
        data = t.model_dump()
        assert data["text"] == "Hi"
        assert data["language"] == "en"
        assert data["duration_seconds"] == 1.0
        assert len(data["segments"]) == 1
