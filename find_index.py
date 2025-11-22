import os
import argparse

ALLOWED = (".png", ".jpg", ".jpeg")

def main():
    parser = argparse.ArgumentParser(
        description="Find the 1-based index of an image inside its folder."
    )
    parser.add_argument("--path", required=True,
                        help="Full path to the image file")
    args = parser.parse_args()

    full_path = args.path
    folder = os.path.dirname(full_path)
    target = os.path.basename(full_path)

    if not os.path.exists(folder):
        print(f"Folder not found: {folder}")
        return

    # List and sort valid image files
    files = sorted([f for f in os.listdir(folder)
                    if f.lower().endswith(ALLOWED)])

    if target not in files:
        print(f"{target} was NOT found in folder: {folder}")
        return

    index = files.index(target) + 1  # 1-based index
    print(f"{target} is image #{index} out of {len(files)} images in the folder.")


if __name__ == "__main__":
    main()
