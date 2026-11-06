from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler


MODEL_ID = "runwayml/stable-diffusion-v1-5"
IMAGE_SIZE = 512
IMAGES_PER_CLASS = 2000

CLASSES = [
    "Tomato__healthy",
    "Tomato__Early_blight",
    "Tomato__Late_blight",
]

NEGATIVE_PROMPT = (
    "blurry, low resolution, cartoon, painting, artificial, fake, watermark, "
    "out of focus, unnatural colors, deformed anatomy"
)

PROMPTS = {
    "Tomato__healthy": (
        "A highly detailed, 4k macro photo of a healthy green tomato leaf, "
        "sharp focus, realistic organic texture, natural greenhouse lighting"
    ),
    "Tomato__Early_blight": (
        "A highly detailed, 4k macro photo of a tomato leaf infected with early blight, "
        "showing sharp brown concentric spots, realistic dead plant tissue"
    ),
    "Tomato__Late_blight": (
        "A highly detailed, 4k macro photo of a tomato leaf infected with late blight, "
        "showing realistic dark water-soaked lesions, wilting plant tissue"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic tomato leaf images with Stable Diffusion img2img."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="Directory containing the real_validation class folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where synthetic_train images will be written.",
    )
    parser.add_argument(
        "--images-per-class",
        type=int,
        default=IMAGES_PER_CLASS,
        help=f"Number of images to generate per class (default: {IMAGES_PER_CLASS}).",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device (default: cuda).",
    )
    return parser.parse_args()


def create_pipeline(device: str):
    dtype = torch.float16 if device.startswith("cuda") else torch.float32

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config
    )
    return pipe.to(device)


def generate_images(
    pipe,
    source_dir: Path,
    output_dir: Path,
    images_per_class: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for class_name in CLASSES:
        source_class_dir = source_dir / class_name
        output_class_dir = output_dir / class_name
        output_class_dir.mkdir(parents=True, exist_ok=True)

        source_images = sorted(
            path
            for path in source_class_dir.iterdir()
            if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )

        if not source_images:
            raise RuntimeError(
                f"No source images found for {class_name}: {source_class_dir}"
            )

        count = 0
        while count < images_per_class:
            for image_path in source_images:
                if count >= images_per_class:
                    break

                init_image = (
                    Image.open(image_path)
                    .convert("RGB")
                    .resize((IMAGE_SIZE, IMAGE_SIZE))
                )

                generated = pipe(
                    prompt=PROMPTS[class_name],
                    negative_prompt=NEGATIVE_PROMPT,
                    image=init_image,
                    strength=0.65,
                    guidance_scale=8.0,
                    num_inference_steps=25,
                ).images[0]

                output_path = output_class_dir / f"synthetic_hq_{count}.png"
                generated.save(output_path)
                count += 1

                print(
                    f"{class_name}: {count}/{images_per_class} -> "
                    f"{output_path.name}"
                )


def main() -> None:
    args = parse_args()
    pipe = create_pipeline(args.device)

    generate_images(
        pipe=pipe,
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        images_per_class=args.images_per_class,
    )


if __name__ == "__main__":
    main()
