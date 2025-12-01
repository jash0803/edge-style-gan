#!/usr/bin/env python
"""Compute FID score using cleanfid library"""
import sys
import argparse
from cleanfid import fid


def main():
    parser = argparse.ArgumentParser(
        description="Compute FID score for a folder of generated images using cleanfid."
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default="qnnpack-quantized-outputs",
        help="Path to directory with generated images (default: qnnpack-quantized-outputs)",
    )
    args = parser.parse_args()

    fdir1 = args.images_dir

    print(f"Computing FID for: {fdir1}")
    print("Comparing against FFHQ dataset (1024x1024)...")
    print("This may take a while as it downloads FFHQ statistics if needed...")

    try:
        score = fid.compute_fid(
            fdir1,
            dataset_name="ffhq",
            dataset_res=1024,
            dataset_split="trainval70k",
            device="cpu",
            num_workers=0,
            verbose=True
        )
        print(f"\n{'='*50}")
        print(f"FID Score: {score:.4f}")
        print(f"{'='*50}")
        return score
    except Exception as e:
        print(f"Error computing FID: {e}")
        print("\nTrying alternative approach...")
        # Try with mode='legacy_torch' which might work better
        try:
            score = fid.compute_fid(
                fdir1,
                dataset_name="ffhq",
                dataset_res=1024,
                dataset_split="trainval70k",
                device="cpu",
                num_workers=0,
                mode="legacy_torch",
                verbose=True
            )
            print(f"\n{'='*50}")
            print(f"FID Score (legacy mode): {score:.4f}")
            print(f"{'='*50}")
            return score
        except Exception as e2:
            print(f"Error with legacy mode: {e2}")
            sys.exit(1)

if __name__ == "__main__":
    main()

