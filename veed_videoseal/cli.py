"""`veed-videoseal` — embed (sign) or verify the VEED AI-generated-content watermark in a video.

One entrypoint with two subcommands so both halves of the watermark live behind a single,
shared tool:

    veed-videoseal sign   --video IN.mp4 --out OUT.mp4 [--json] [--device ...] [--ckpt-dir ...]
    veed-videoseal verify --video IN.mp4              [--json] [--out FILE] [--device ...] [--ckpt-dir ...]

With ``--json`` each subcommand prints exactly one JSON object to stdout (and nothing else),
for machine consumption (e.g. by a calling task processor); without it, human-readable
text is printed. ``verify --out FILE`` instead writes that JSON object to FILE, so a consumer can
read the verdict from a file rather than de-interleaving it from stdout.

``verify`` exits 0 on any successful check — the verdict (present/absent) is carried in the JSON /
``--out`` file / text, NOT the exit code — and non-zero only when the check itself errored (e.g. an
unreadable video). This keeps a caller that treats a non-zero exit as failure from mistaking a
legitimate "not detected" for a broken run.

The heavy torch/videoseal import is deferred into the subcommands: ``main`` emits an immediate
liveness line, then each subcommand wraps that import in a Heartbeat, so a caller's no-progress
timer keeps getting ticks through the multi-second import instead of one line and then silence.
"""

import argparse
import json
import sys

from ._heartbeat import Heartbeat


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--device", default="auto", help="auto (cuda→cpu), cpu, cuda, or mps")
    p.add_argument("--ckpt-dir", default=None, help="dir holding/caching VideoSeal weights")
    p.add_argument("--json", action="store_true", help="emit a single JSON object to stdout")


def _run_verify(args) -> int:
    # Imported here (not at module load) so `veed-videoseal` starts — and main() prints its liveness
    # line — before the multi-second torch/videoseal import. The import is itself silent and can
    # exceed a single no-progress interval, so wrap it in a Heartbeat so ticks keep flowing through
    # it, not just the one startup line.
    with Heartbeat("verify: loading dependencies"):
        from .constants import METADATA_TAGS
        from .detect import detect_watermark
        from .video_io import read_metadata_tags, read_video_frames

    # Heartbeat around each silent phase: a caller may kill a child that emits no output
    # for too long. Decode can be slow on a large/high-res input; model load + first inference are
    # silent too. All progress goes to stderr; stdout stays reserved for the single JSON object.
    print("verify: decoding", file=sys.stderr, flush=True)
    with Heartbeat("verify: decoding video"):
        frames = read_video_frames(args.video)
    print("verify: detecting", file=sys.stderr, flush=True)
    with Heartbeat("verify: loading model / detecting"):
        res = detect_watermark(frames, device=args.device, ckpt_dir=args.ckpt_dir)
    tags = read_metadata_tags(args.video)
    metadata_present = all(tags.get(k) == v for k, v in METADATA_TAGS.items())

    verdict = {
        "detected": res["detected"],
        "bit_accuracy": res["bit_accuracy"],
        "nbits": res["nbits"],
        "metadata_present": metadata_present,
    }

    # --out writes the verdict JSON to a file the caller named, so a consumer can read
    # that file instead of parsing stdout, which is interleaved with the stderr progress above.
    # --json still prints the same object to stdout; with neither, print human-readable text.
    if args.out:
        with open(args.out, "w") as f:
            json.dump(verdict, f)
    if args.json:
        print(json.dumps(verdict))
    if not args.json and not args.out:
        mark = "PRESENT" if res["detected"] else "ABSENT"
        print(f"Pixel watermark:   {mark}  (bit_accuracy={res['bit_accuracy']:.4f}, nbits={res['nbits']})")
        print(f"Metadata marker:   {'present' if metadata_present else 'absent'}")
    # Exit 0 on a successful check regardless of the verdict — "not detected" is a valid result,
    # not a failure. The verdict lives in the JSON/out-file/text above; a real error raises and
    # exits non-zero via the traceback.
    return 0


def _run_sign(args) -> int:
    with Heartbeat("sign: loading dependencies"):  # deferred, heartbeat-wrapped import (see _run_verify)
        from .sign import sign_video

    res = sign_video(args.video, args.out, device=args.device, ckpt_dir=args.ckpt_dir)
    if args.json:
        print(json.dumps(res))
    else:
        print(f"Signed {args.video} -> {res['out']}  ({res['frames']} frames)")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="veed-videoseal",
        description="Embed or verify the invisible VEED AI-generated-content watermark in a video.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    sign = sub.add_parser("sign", help="watermark an arbitrary video (preserves audio/fps/duration)")
    sign.add_argument("--video", required=True, help="path to the input video")
    sign.add_argument("--out", required=True, help="path to write the signed video")
    _add_common(sign)
    sign.set_defaults(func=_run_sign)

    verify = sub.add_parser("verify", help="check a video for the VEED watermark")
    verify.add_argument("--video", required=True, help="path to the video file to check")
    verify.add_argument("--out", default=None,
                        help="write the verdict JSON object to this file (for machine consumption)")
    _add_common(verify)
    verify.set_defaults(func=_run_verify)

    return p


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    # Immediate liveness line to stderr before the subcommand triggers the heavy torch/videoseal
    # import: it resets a caller's no-progress timer from the very start, so the silent import can
    # never look like a hang. stdout stays reserved for the single JSON object.
    print(f"veed-videoseal: {args.command} starting", file=sys.stderr, flush=True)
    return args.func(args)


def verify_main(argv=None) -> int:
    """Back-compat entrypoint for the old ``veed-videoseal-verify`` console script."""
    return main(["verify", *(sys.argv[1:] if argv is None else argv)])


if __name__ == "__main__":
    sys.exit(main())
