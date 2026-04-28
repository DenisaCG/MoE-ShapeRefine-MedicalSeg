import os
import zipfile
from pathlib import Path

def rebuild_pytorch_checkpoint(folder_path, output_file):
    folder = Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    print(f"Rebuilding checkpoint from: {folder}")
    print(f"Output file: {output_file}")

    # Create proper PyTorch-style zip archive
    with zipfile.ZipFile(output_file, "w", compression=zipfile.ZIP_DEFLATED) as zf:

        for root, _, files in os.walk(folder):
            for file in files:
                full_path = Path(root) / file

                # IMPORTANT: preserve internal structure
                arcname = full_path.relative_to(folder.parent)

                zf.write(full_path, arcname)

                print(f"Added: {arcname}")

    print("\n Checkpoint rebuilt successfully!")

if __name__ == "__main__":
    # Adjust if your folder has different name
    folder_path = "medsam_vit_b"
    output_file = "work_dir/MedSAM/medsam_vit_b.pth"

    rebuild_pytorch_checkpoint(folder_path, output_file)