"""Pure unit tests for the signed-output ffmpeg command builders. No model/GPU/ffmpeg run —
they assert the mapping, SAR, audio (copy/re-encode) and rotation-remux flags directly on the
argument list."""

import pytest

from veed_videoseal.sign import _build_encode_cmd, _build_rotate_remux_cmd, _valid_sar


def _cmd(**kw):
    base = dict(width=1920, height=1080, rate="25", audio_codecs=["aac"], in_path="in.mp4",
                out_path="out.mp4", sar=None)
    base.update(kw)
    return _build_encode_cmd(**base)


def _pairs(cmd):
    return list(zip(cmd, cmd[1:]))


def test_encode_cmd_does_not_stamp_rotation():
    # Rotation is handled by the separate remux pass, never the encode: -metadata rotate= is a
    # no-op on ffmpeg >=7, so emitting it here would silently drop rotation.
    cmd = _cmd()
    assert not any(a == "-metadata:s:v:0" and v.startswith("rotate=") for a, v in _pairs(cmd))
    assert "-display_rotation" not in cmd


def test_remux_stamps_negated_rotation():
    # -display_rotation is counter-clockwise while the rotate tag is clockwise, so the angle is
    # negated: -display_rotation -R makes _probe_rotation read back R (idempotent round-trip).
    cmd = _build_rotate_remux_cmd("pre.mp4", 90, "out.mp4")
    assert ("-display_rotation", "-90") in _pairs(cmd)
    assert ("-i", "pre.mp4") in _pairs(cmd)


def test_remux_is_stream_copy_and_output_last():
    # Must be a pure remux (-c copy) — a re-encode would drop the display matrix and re-do the work.
    cmd = _build_rotate_remux_cmd("pre.mp4", 270, "out.mp4")
    assert ("-c", "copy") in _pairs(cmd)
    assert ("-map", "0") in _pairs(cmd)  # carry every stream (video + all audio) across
    assert cmd[-1] == "out.mp4"


def test_copies_mp4_compatible_audio_untouched():
    # aac fits mp4, so stream-copy it — no re-encode, and crucially no apad/-shortest (which would
    # pad/clamp the audio away from the source length).
    cmd = _cmd(audio_codecs=["aac"])
    assert ("-c:a", "copy") in _pairs(cmd)
    assert "apad" not in cmd and "-shortest" not in cmd
    assert "aac" not in cmd[cmd.index("-c:a") + 2:]  # not also re-encoding


def test_reencodes_incompatible_audio_to_aac():
    # opus can't stream-copy into mp4, so it must be re-encoded — still no apad/-shortest.
    cmd = _cmd(audio_codecs=["opus"])
    assert ("-c:a", "aac") in _pairs(cmd)
    assert "copy" not in cmd
    assert "apad" not in cmd and "-shortest" not in cmd


def test_mixed_codec_tracks_reencode_all_not_copy():
    # First track copyable (aac) but a second is not (vorbis): copying would apply -c:a copy to the
    # vorbis track too, which the mp4 muxer rejects. So the whole set must be re-encoded, not copied.
    cmd = _cmd(audio_codecs=["aac", "vorbis"])
    assert ("-c:a", "aac") in _pairs(cmd)
    assert "copy" not in cmd


def test_present_but_unnamed_audio_is_reencoded_not_dropped():
    # A stream ffprobe reports with an empty codec_name is still audio: preserve it (re-encoded),
    # do not drop it (which conflating empty-codec with no-audio would do).
    cmd = _cmd(audio_codecs=[""])
    assert cmd.count("-i") == 2  # source is still mapped in as the audio input
    assert ("-c:a", "aac") in _pairs(cmd)


def test_no_audio_flags_when_source_has_no_audio():
    cmd = _cmd(audio_codecs=[])
    assert "-c:a" not in cmd
    assert "1:a?" not in cmd
    assert "-i" == cmd[cmd.index("-i")]  # only the pipe input (no second -i for the source)
    assert cmd.count("-i") == 1


def test_faststart_skipped_for_temp_encode():
    # The rotation temp encode passes faststart=False (the remux re-faststarts); a redundant
    # faststart here would be a wasted moov-atom relocation over the whole encoded file.
    assert "+faststart" not in _cmd(faststart=False)
    assert "+faststart" in _cmd()  # default keeps it for the direct (non-rotation) encode


def test_odd_dimensions_get_cropped_even():
    cmd = _cmd(width=271, height=481)
    assert ("-vf", "crop=270:480") in _pairs(cmd)


def test_even_dimensions_are_not_cropped():
    cmd = _cmd(width=640, height=480)
    assert not any(a == "-vf" for a, _ in _pairs(cmd))


def test_marker_tags_present_and_output_last():
    cmd = _cmd()
    # New signing uses the model-agnostic marker (not the frozen legacy tag).
    assert ("-metadata", "comment=VEED:AI-generated") in _pairs(cmd)
    assert cmd[-1] == "out.mp4"


def _vf(cmd):
    """Return the single -vf filter chain string, or None if absent."""
    return next((v for a, v in _pairs(cmd) if a == "-vf"), None)


def test_preserves_sar():
    # Anamorphic source: the SAR must be re-stamped or the output plays squished.
    assert "setsar=40/33" in (_vf(_cmd(sar="40/33")) or "")


def test_setsar_overrides_default_max_clamp():
    # setsar defaults to max=100, which silently re-rationalizes SARs with a component >100 (e.g.
    # the standard H.264 160:99) into a different, distorted ratio. We must pin a large max.
    assert "setsar=160/99:max=1000000" in (_vf(_cmd(sar="160/99")) or "")


def test_no_setsar_when_square():
    cmd = _cmd(sar=None)
    assert not any("setsar" in tok for tok in cmd)


def test_sar_and_crop_compose():
    # Odd dims (crop) + anamorphic SAR (setsar) share the one allowed -vf chain, comma-joined.
    cmd = _cmd(width=271, height=481, sar="40/33")
    assert ("-vf", "crop=270:480,setsar=40/33:max=1000000") in _pairs(cmd)


@pytest.mark.parametrize("sar", ["1:1", "0:1", "N/A", None, "", "x:1", "4:0"])
def test_valid_sar_rejects_undefined_square_and_malformed(sar):
    assert _valid_sar(sar) is None


def test_valid_sar_normalizes_anamorphic_to_slash():
    assert _valid_sar("40:33") == "40/33"
