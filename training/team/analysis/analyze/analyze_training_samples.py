"""
Analyze Training Samples
========================
"""

import pandas as pd

from training.team.config import TEAM_DATASET_PATH


def load_dataset():

    print("=" * 60)
    print("LOAD DATASET")
    print("=" * 60)

    df = pd.read_csv(TEAM_DATASET_PATH)

    print(f"[INFO] Shape : {df.shape}")

    return df

def analyze_total_maps(df):

    print("\n" + "=" * 60)
    print("TOTAL MAPS DISTRIBUTION")
    print("=" * 60)

    summary = (

        df["Total Maps Played"]

        .value_counts()

        .sort_index()

    )

    print(summary)

    print()

    print(f"Mean   : {df['Total Maps Played'].mean():.2f}")

    print(f"Median : {df['Total Maps Played'].median():.2f}")

    print(f"Max    : {df['Total Maps Played'].max()}")
    
def analyze_threshold(df):

    print("\n" + "=" * 60)
    print("THRESHOLD ANALYSIS")
    print("=" * 60)

    total = len(df)

    for threshold in [1, 2, 3, 4, 5]:

        count = len(

            df[
                df["Total Maps Played"] >= threshold
            ]

        )

        print(

            f">= {threshold} Maps : "

            f"{count:4d} "

            f"({count / total:.1%})"

        )
        
def main():

    df = load_dataset()

    analyze_total_maps(df)

    analyze_threshold(df)


if __name__ == "__main__":
    main()