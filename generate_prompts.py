"""

This script generates text prompts (captions) for images in a specified folder

using a BLIP image-to-text pipeline. The generated prompts are saved as key-value
pairs in a JSON file, where the key is the image filename and the value is the
corresponding prompt

Added as part of masters project Spring 2025
"""

from json import dump
from os import listdir, path

from PIL import Image
from transformers import pipeline


def generate_prompts_for_folder(images_folder: str, output_file: str) -> None:
    """
    Processes all images in `images_folder` to generate a text prompt (caption)
    for each image using a BLIP image-to-text pipeline, and then saves them as
    key-value pairs in a JSON file (image filename: prompt).
    """
    captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

    prompts_dict = {}

    for filename in listdir(images_folder):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            continue  # Skip non-image files.

        image_path = path.join(images_folder, filename)
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            continue

        try:
            generated = captioner(image)

            prompt_text = generated[0][
                'generated_text'
            ]  # Extract the generated text from the output
        except Exception as e:
            print(f"Error generating caption for {filename}: {e}")
            prompt_text = "error generating caption"

        prompts_dict[filename] = prompt_text
        print(f"Processed {filename}: {prompt_text}")

    with open(output_file, "w") as f:
        dump(prompts_dict, f, indent=4)
    print(f"Saved prompts to {output_file}")


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(
        description="Generate image captions for all images in a folder and save to a JSON file."
    )
    parser.add_argument(
        "--images_folder",
        type=str,
        required=True,
        help="Path to the folder containing images",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="image_prompts.json",
        help="Path to the output JSON file",
    )
    args = parser.parse_args()

    generate_prompts_for_folder(args.images_folder, args.output_file)
