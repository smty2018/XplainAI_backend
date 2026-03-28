#!/usr/bin/env python
"""Main script to run the local parser."""

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).parent))

from src.parser import LocalParser
from src.reasoner import SolutionOrchestrator


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="XplainAI Parser and Reasoner")
    parser.add_argument("input", help="Input text, image path, PDF path, or saved parser JSON path")
    parser.add_argument(
        "--type",
        choices=["text", "image", "pdf", "json", "auto"],
        default="auto",
        help="Input type",
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Config file path",
    )
    parser.add_argument(
        "--prompt",
        default="",
        help="Optional extra instruction to combine with image or PDF input",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Show local storage stats after the run",
    )
    parser.add_argument(
        "--reason",
        action="store_true",
        help="Run the reasoning v1 model after parsing",
    )
    parser.add_argument(
        "--reason-trace",
        action="store_true",
        help="Include a public reasoning trace summary in the solution output",
    )
    parser.add_argument(
        "--scene-planner",
        action="store_true",
        help="Generate a second-pass Scene Planner prompt for a downstream Manim code generator",
    )
    parser.add_argument(
        "--manim-code",
        action="store_true",
        help="Generate final Manim code using the Scene Planner plus few-shot layout guidance",
    )
    parser.add_argument(
        "--animation-prompt",
        action="store_true",
        help=argparse.SUPPRESS,
    )

    args = parser.parse_args()

    if args.animation_prompt:
        args.scene_planner = True

    if args.manim_code:
        args.scene_planner = True

    if args.scene_planner:
        args.reason = True

    print("=" * 50)
    print("XplainAI Parser" + (" + Reasoner" if args.reason else ""))
    print("=" * 50)
    print("Initializing pipeline..." if args.reason else "Initializing parser...")
    print()

    run_start = perf_counter()
    parser_obj = None
    orchestrator = None
    if args.reason:
        orchestrator = SolutionOrchestrator(args.config)
        parser_obj = orchestrator.parser
    else:
        parser_obj = LocalParser(args.config)

    print(f"\nProcessing: {args.input}")
    print("-" * 50)

    if args.reason:
        result = orchestrator.process(
            args.input,
            input_type=args.type,
            prompt_text=args.prompt,
            include_reasoning_trace=args.reason_trace,
            generate_scene_planner=args.scene_planner,
            generate_manim_code=args.manim_code,
        )
    elif args.type == "auto":
        input_path = Path(args.input)
        if args.prompt and input_path.exists():
            suffix = input_path.suffix.lower()
            if suffix == ".pdf":
                result = parser_obj.parse_pdf(args.input, prompt_text=args.prompt)
            elif suffix in {".png", ".jpg", ".jpeg"}:
                from PIL import Image

                result = parser_obj.parse_image(Image.open(args.input), prompt_text=args.prompt)
            else:
                result = parser_obj.parse(args.input)
        else:
            result = parser_obj.parse(args.input)
    elif args.type == "text":
        result = parser_obj.parse_text(args.input)
    elif args.type == "pdf":
        result = parser_obj.parse_pdf(args.input, prompt_text=args.prompt)
    else:
        from PIL import Image

        result = parser_obj.parse_image(Image.open(args.input), prompt_text=args.prompt)

    if args.reason:
        print("\n" + "=" * 50)
        print("PARSED INPUT:")
        print("=" * 50)
        print(json.dumps(result.get("parsed_input", {}), indent=2, ensure_ascii=False))

        print("\n" + "=" * 50)
        print("SOLUTION OUTPUT:")
        print("=" * 50)
        print(json.dumps(result.get("solution", {}), indent=2, ensure_ascii=False))

        print("\n" + "=" * 50)
        print("PIPELINE METADATA:")
        print("=" * 50)
        print(json.dumps(result.get("pipeline_metadata", {}), indent=2, ensure_ascii=False))

        if args.scene_planner:
            print("\n" + "=" * 50)
            print("SCENE PLANNER:")
            print("=" * 50)
            print((result.get("scene_planner", {}) or {}).get("text", ""))

        if args.manim_code:
            print("\n" + "=" * 50)
            print("MANIM CODE:")
            print("=" * 50)
            print((result.get("manim_code", {}) or {}).get("text", ""))
    else:
        print("\n" + "=" * 50)
        print("PARSING RESULT:")
        print("=" * 50)
        print(json.dumps(result, indent=2, ensure_ascii=False))

        timing = result.get("_timing")
        if timing:
            print("\n" + "=" * 50)
            print("TIMING:")
            print("=" * 50)
            for key, value in timing.items():
                if isinstance(value, float):
                    print(f"{key}: {value:.3f} sec")
                else:
                    print(f"{key}: {value}")

    print(f"\nCLI total runtime: {perf_counter() - run_start:.3f} sec")

    if args.save:
        print("\n" + "=" * 50)
        print("STORAGE STATS:")
        print("=" * 50)
        stats = parser_obj.get_stats()
        for key, value in stats.items():
            print(f"{key}: {value:.2f} MB")


if __name__ == "__main__":
    main()
