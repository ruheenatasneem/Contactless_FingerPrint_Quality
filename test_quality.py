import os
import glob
import pandas as pd

from quality_assessment import quality_gate


# ==========================================================
# Get Image List
# ==========================================================

def get_image_list(dataset_folder="test_dataset"):
    """
    Returns all image paths.
    """

    image_paths = []

    extensions = [
        "*.jpg",
        "*.jpeg",
        "*.png"
    ]

    categories = [
        "good",
        "blurry",
        "dark",
        "glare"
    ]

    for category in categories:

        folder = os.path.join(
            dataset_folder,
            category
        )

        for ext in extensions:

            files = glob.glob(
                os.path.join(folder, ext)
            )

            image_paths.extend(files)

    return image_paths


# ==========================================================
# Evaluate Dataset
# ==========================================================

def evaluate_dataset(dataset_folder="test_dataset"):

    image_paths = get_image_list(dataset_folder)

    if len(image_paths) == 0:

        print("No images found.")

        return

    records = []

    print("\n")
    print("=" * 90)
    print("CONTACTLESS FINGERPRINT QUALITY TEST")
    print("=" * 90)

    for image_path in image_paths:

        folder_name = os.path.basename(
            os.path.dirname(image_path)
        )

        file_name = os.path.basename(
            image_path
        )

        try:

            result = quality_gate(image_path)

            records.append({

                "Filename":
                    file_name,

                "Category":
                    folder_name,

                "Passed":
                    result["passed"],

                "Composite Score":
                    result["composite_score"],

                "Blur Score":
                    result["blur"]["blur_score"],

                "Brightness":
                    result["brightness"]["brightness"],

                "Glare":
                    result["glare"]["glare_fraction"],

                "ROI":
                    result["roi"]["roi_fraction"],

                "Ridge":
                    result["ridge"]["ridge_score"],

                "Guidance":
                    result["guidance"]

            })

            print(
                f"{file_name:20}"
                f"{folder_name:12}"
                f"{result['composite_score']:8}"
                f"{'PASS' if result['passed'] else 'FAIL':10}"
            )

        except Exception as e:

            print(file_name)

            print(e)

    print("=" * 90)

    df = pd.DataFrame(records)

    print("\n")
    print(df)

    df.to_csv(
        "test_results.csv",
        index=False
    )

    print("\n")
    print("CSV file saved as test_results.csv")

    print("\n")
    print("Total Images :", len(df))

    passed = len(df[df["Passed"] == True])

    failed = len(df[df["Passed"] == False])

    print("Passed      :", passed)

    print("Failed      :", failed)

    print("=" * 90)

    return df


# ==========================================================
# Category Summary
# ==========================================================

def category_summary(df):

    print("\n")

    print("=" * 60)

    print("CATEGORY SUMMARY")

    print("=" * 60)

    summary = df.groupby(
        "Category"
    ).agg(

        Images=("Filename", "count"),

        Average_Score=(
            "Composite Score",
            "mean"
        ),

        Passed=(
            "Passed",
            "sum"
        )

    )

    print(summary)

    print("=" * 60)


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    dataframe = evaluate_dataset()

    if dataframe is not None:

        category_summary(dataframe)