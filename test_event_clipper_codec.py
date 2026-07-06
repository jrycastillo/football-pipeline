#!/usr/bin/env python3
import contextlib
import io
import os
import shutil
import sys
import tempfile
import types

sys.modules["cv2"] = types.ModuleType("cv2")

from stats import event_clipper


class FakeFrame:
    def __init__(self, value):
        self.value = value

    def tobytes(self):
        return bytes([self.value])


class FakeCapture:
    def __init__(self, path):
        self.path = path
        self.pos = 0
        self.frames = [FakeFrame(i) for i in range(10)]

    def isOpened(self):
        return True

    def get(self, prop):
        cv2 = sys.modules["cv2"]
        if prop == cv2.CAP_PROP_FPS:
            return 10.0
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return 2
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return 2
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return len(self.frames)
        return 0

    def set(self, prop, value):
        cv2 = sys.modules["cv2"]
        if prop == cv2.CAP_PROP_POS_FRAMES:
            self.pos = int(value)

    def read(self):
        if self.pos >= len(self.frames):
            return False, None
        frame = self.frames[self.pos]
        self.pos += 1
        return True, frame

    def release(self):
        pass


class FakeWriter:
    def __init__(self, path, fourcc, fps, size):
        self.path = path
        self.frames = []

    def write(self, frame):
        self.frames.append(frame)

    def release(self):
        with open(self.path, "wb") as f:
            f.write(b"x" * len(self.frames))


def _install_fake_cv2():
    cv2 = sys.modules["cv2"]
    cv2.CAP_PROP_FPS = 1
    cv2.CAP_PROP_FRAME_WIDTH = 2
    cv2.CAP_PROP_FRAME_HEIGHT = 3
    cv2.CAP_PROP_FRAME_COUNT = 4
    cv2.CAP_PROP_POS_FRAMES = 5
    cv2.VideoCapture = FakeCapture
    cv2.VideoWriter = FakeWriter
    cv2.VideoWriter_fourcc = lambda *args: 1234


def _reset_ffmpeg_probe():
    event_clipper._FFMPEG_PROBED = False
    event_clipper._FFMPEG_PATH = None
    event_clipper._FFMPEG_UNAVAILABLE_WARNED = False


def test_ffmpeg_command():
    cmd = event_clipper._build_ffmpeg_command(
        "/usr/bin/ffmpeg", "clip.mp4", 25.0, 1920, 1080)
    assert cmd == [
        "/usr/bin/ffmpeg",
        "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", "1920x1080",
        "-r", "25.0",
        "-i", "-",
        "-an",
        "-vcodec", "libx264",
        "-preset", "veryfast",
        "-crf", "26",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "clip.mp4",
    ]


def test_codec_selection_with_cached_ffmpeg_probe():
    original_which = shutil.which
    calls = []
    try:
        _reset_ffmpeg_probe()
        shutil.which = lambda name: calls.append(name) or "/fake/ffmpeg"
        assert event_clipper._select_codec("h264") == ("h264", "/fake/ffmpeg")
        assert event_clipper._select_codec("h264") == ("h264", "/fake/ffmpeg")
        assert calls == ["ffmpeg"]

        _reset_ffmpeg_probe()
        calls.clear()
        shutil.which = lambda name: calls.append(name) or None
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            assert event_clipper._select_codec("h264") == ("mp4v", None)
            assert event_clipper._select_codec("h264") == ("mp4v", None)
        assert calls == ["ffmpeg"]
        assert buf.getvalue().count("ffmpeg not found") == 1

        assert event_clipper._select_codec("mp4v") == ("mp4v", None)
    finally:
        shutil.which = original_which
        _reset_ffmpeg_probe()


def test_manifest_fields_with_stubbed_mp4v_writer():
    _install_fake_cv2()
    with tempfile.TemporaryDirectory() as td:
        manifest = event_clipper.clip_events(
            "match.mp4",
            [{"type": "shot", "player": 7, "frame": 0, "confidence": 0.8}],
            td,
            vid_stride=1,
            pad_s=0.1,
            event_types=("shot",),
            codec="mp4v",
        )

        assert len(manifest) == 1
        entry = manifest[0]
        assert entry["clip"] == os.path.join("clips", "000_shot_p7_f1.mp4")
        assert entry["codec"] == "mp4v"
        assert entry["size_bytes"] == 3
        assert entry["source_frame"] == 1
        assert entry["clip_start_s"] == 0.0
        assert entry["clip_end_s"] == 0.2


def test_manifest_fields_with_stubbed_h264_writer():
    _install_fake_cv2()
    original_which = shutil.which
    original_ffmpeg_writer = event_clipper._write_clip_ffmpeg
    try:
        _reset_ffmpeg_probe()
        shutil.which = lambda name: "/fake/ffmpeg"

        def fake_ffmpeg_writer(cap, clip_path, start, end, fps, width, height, ffmpeg_path):
            with open(clip_path, "wb") as f:
                f.write(b"h264")
            return 3

        event_clipper._write_clip_ffmpeg = fake_ffmpeg_writer
        with tempfile.TemporaryDirectory() as td:
            manifest = event_clipper.clip_events(
                "match.mp4",
                [{"type": "shot", "player": 7, "frame": 0, "confidence": 0.8}],
                td,
                vid_stride=1,
                pad_s=0.1,
                event_types=("shot",),
                codec="h264",
            )

        assert len(manifest) == 1
        assert manifest[0]["codec"] == "h264"
        assert manifest[0]["size_bytes"] == 4
    finally:
        shutil.which = original_which
        event_clipper._write_clip_ffmpeg = original_ffmpeg_writer
        _reset_ffmpeg_probe()


def test_ffmpeg_encode_failure_falls_back_per_clip():
    _install_fake_cv2()
    original_which = shutil.which
    original_ffmpeg_writer = event_clipper._write_clip_ffmpeg
    try:
        _reset_ffmpeg_probe()
        shutil.which = lambda name: "/fake/ffmpeg"

        def fail_ffmpeg(*args, **kwargs):
            raise RuntimeError("boom")

        event_clipper._write_clip_ffmpeg = fail_ffmpeg
        with tempfile.TemporaryDirectory() as td:
            manifest = event_clipper.clip_events(
                "match.mp4",
                [{"type": "save", "player": 1, "frame": 0, "confidence": 0.7}],
                td,
                vid_stride=1,
                pad_s=0.1,
                event_types=("save",),
                codec="h264",
            )

        assert len(manifest) == 1
        assert manifest[0]["codec"] == "mp4v"
        assert manifest[0]["size_bytes"] == 3
    finally:
        shutil.which = original_which
        event_clipper._write_clip_ffmpeg = original_ffmpeg_writer
        _reset_ffmpeg_probe()


if __name__ == "__main__":
    test_ffmpeg_command()
    test_codec_selection_with_cached_ffmpeg_probe()
    test_manifest_fields_with_stubbed_mp4v_writer()
    test_manifest_fields_with_stubbed_h264_writer()
    test_ffmpeg_encode_failure_falls_back_per_clip()
    print("test_event_clipper_codec passed")
