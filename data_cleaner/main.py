from __future__ import annotations
from pathlib import Path
from typing import Literal

import polars as pl


class AtlasDataCleaner:
    """Clean and filter Atlas 2010 dashboard data for analysis."""

    def __init__(self, file_path: str) -> None:
        self.file_path: Path = Path(file_path)
        self.df: pl.DataFrame | None = None

    def load_data(self) -> AtlasDataCleaner:
        """Load Excel file into Polars DataFrame."""
        print(f"Loading data from {self.file_path}...")
        self.df = pl.read_excel(self.file_path)
        print(f"Loaded {len(self.df)} rows and {len(self.df.columns)} columns")
        return self

    def show_info(self) -> AtlasDataCleaner:
        """Display basic information about the dataset."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        print("\n=== Dataset Info ===")
        print(f"Shape: {self.df.shape}")
        print(f"\nColumns: {self.df.columns}")
        print(f"\nData types:\n{self.df.dtypes}")
        print(f"\nFirst few rows:\n{self.df.head()}")
        print(f"\nNull counts:\n{self.df.null_count()}")
        return self

    def clean_data(self) -> AtlasDataCleaner:
        """Apply cleaning operations to the dataset."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        print("\n=== Cleaning Data ===")
        initial_rows: int = len(self.df)

        # Remove duplicate rows
        self.df = self.df.unique()
        print(f"Removed {initial_rows - len(self.df)} duplicate rows")

        # Remove rows where all values are null
        self.df = self.df.filter(~pl.all_horizontal(pl.all().is_null()))
        print(f"Removed rows with all null values")

        print(f"Rows after cleaning: {len(self.df)}")
        return self

    def filter_necessary_columns(self, columns: list[str] | None = None) -> AtlasDataCleaner:
        """Keep only necessary columns for analysis."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        if columns:
            print(f"\n=== Filtering to {len(columns)} columns ===")
            self.df = self.df.select(columns)
            print(f"Selected columns: {columns}")
        else:
            print("\nNo column filter applied. Specify columns parameter to filter.")

        return self

    def filter_rows(self, condition: pl.Expr) -> AtlasDataCleaner:
        """Apply custom row filter using Polars expression."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        initial_rows: int = len(self.df)
        self.df = self.df.filter(condition)
        print(f"\n=== Row Filter Applied ===")
        print(f"Retained {len(self.df)} of {initial_rows} rows")
        return self

    def remove_high_null_columns(self, threshold: float = 0.5) -> AtlasDataCleaner:
        """Remove columns with null percentage above threshold."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        print(f"\n=== Removing columns with >{threshold*100}% nulls ===")
        null_percentages: pl.DataFrame = self.df.null_count() / len(self.df)

        columns_to_keep: list[str] = [
            col for col, null_pct in zip(self.df.columns, null_percentages.row(0))
            if null_pct <= threshold
        ]

        removed: set[str] = set(self.df.columns) - set(columns_to_keep)
        if removed:
            print(f"Removing columns: {removed}")
            self.df = self.df.select(columns_to_keep)
        else:
            print("No columns removed")

        return self

    def save_cleaned_data(
        self,
        output_path: str,
        format: Literal["parquet", "csv", "excel"] = "parquet"
    ) -> AtlasDataCleaner:
        """Save cleaned data to file."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        output_file: Path = Path(output_path)
        print(f"\n=== Saving cleaned data to {output_file} ===")

        if format == "parquet":
            self.df.write_parquet(output_file)
        elif format == "csv":
            self.df.write_csv(output_file)
        elif format == "excel":
            self.df.write_excel(output_file)
        else:
            raise ValueError(f"Unsupported format: {format}")

        print(f"Saved successfully!")
        return self

    def get_dataframe(self) -> pl.DataFrame | None:
        """Return the cleaned DataFrame."""
        return self.df


def main() -> None:
    cleaner: AtlasDataCleaner = AtlasDataCleaner("atlas2010_dashboard.xlsx")

    # Load and inspect
    cleaner.load_data().show_info()

    # Apply cleaning pipeline
    cleaner.clean_data().remove_high_null_columns(threshold=0.3)

    # Apply data quality filters by category
    cleaner.filter_rows(
        # Demographic validity
        (pl.col('populacao_total') > 0) &
        (pl.col('populacao_total') < 20_000_000) &
        (pl.col('ano') == 2010) &

        # Health indicators
        (pl.col('espvida') > 0) &
        (pl.col('espvida') < 100) &
        (pl.col('mort1') >= 0) &
        (pl.col('mort5') >= 0) &
        (pl.col('mort5') >= pl.col('mort1')) &
        (pl.col('sobre60') > 0) &
        (pl.col('sobre60') <= 100) &

        # Fertility
        (pl.col('fectot') > 0) &
        (pl.col('fectot') < 8) &

        # Education indicators
        (pl.col('e_anosestudo') >= 0) &
        (pl.col('e_anosestudo') < 20) &
        (pl.col('t_analf18m') >= 0) &
        (pl.col('t_analf18m') <= 100) &

        # Economic indicators
        (pl.col('renda_per_capita') > 0) &
        (pl.col('renda_per_capita') < 50_000)
    )

    # Show final result
    df: pl.DataFrame | None = cleaner.get_dataframe()
    if df is not None:
        print(f"\n=== Final Dataset ===")
        print(f"Shape: {df.shape}")
        print(df.head())

    # Save cleaned data
    cleaner.save_cleaned_data("atlas2010_cleaned.parquet")


if __name__ == "__main__":
    main()
