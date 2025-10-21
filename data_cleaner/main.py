from __future__ import annotations
from pathlib import Path
from typing import Literal

import polars as pl


class FavelasDataCleaner:
    """Clean and filter Favelas e Comunidades Urbanas 2022 data for analysis."""

    def __init__(self, file_path: str, sheet_id: int = 2) -> None:
        self.file_path: Path = Path(file_path)
        self.sheet_id: int = sheet_id
        self.df: pl.DataFrame | None = None

    def load_data(self) -> FavelasDataCleaner:
        """Load Excel file into Polars DataFrame."""
        print(f"Loading data from {self.file_path} (Sheet {self.sheet_id})...")
        self.df = pl.read_excel(self.file_path, sheet_id=self.sheet_id)
        print(f"Loaded {len(self.df)} rows and {len(self.df.columns)} columns")
        return self

    def show_info(self) -> FavelasDataCleaner:
        """Display basic information about the dataset."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        print("\n=== Dataset Info ===")
        print(f"Shape: {self.df.shape}")
        print(f"\nColumns: {self.df.columns}")
        print(f"\nData types:")
        for col, dtype in zip(self.df.columns, self.df.dtypes):
            print(f"  {col}: {dtype}")
        
        print(f"\nFirst few rows:")
        print(self.df.head())
        
        print(f"\nNull counts:")
        null_df = self.df.null_count()
        for col in self.df.columns:
            count = null_df[col][0]
            pct = (count / len(self.df)) * 100 if len(self.df) > 0 else 0
            print(f"  {col}: {count} ({pct:.1f}%)")
        
        print(f"\nUnique values per column:")
        for col in self.df.columns:
            unique_count = self.df[col].n_unique()
            print(f"  {col}: {unique_count:,}")
        
        return self

    def clean_data(self) -> FavelasDataCleaner:
        """Apply cleaning operations to the dataset."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        print("\n=== Cleaning Data ===")
        initial_rows: int = len(self.df)

        # Remove duplicate rows based on CD_SETOR (each setor should be unique)
        before_dup = len(self.df)
        self.df = self.df.unique(subset=["CD_SETOR"])
        dup_removed = before_dup - len(self.df)
        if dup_removed > 0:
            print(f"Removed {dup_removed} duplicate rows (based on CD_SETOR)")
        else:
            print("No duplicate rows found")

        # Remove rows where all values are null (if any)
        before_null = len(self.df)
        self.df = self.df.filter(~pl.all_horizontal(pl.all().is_null()))
        null_removed = before_null - len(self.df)
        if null_removed > 0:
            print(f"Removed {null_removed} rows with all null values")

        # Strip whitespace from string columns
        string_cols = [col for col, dtype in zip(self.df.columns, self.df.dtypes) 
                      if dtype == pl.String]
        if string_cols:
            self.df = self.df.with_columns([
                pl.col(col).str.strip_chars().alias(col) for col in string_cols
            ])
            print(f"Stripped whitespace from {len(string_cols)} string columns")

        print(f"Total rows removed: {initial_rows - len(self.df)}")
        print(f"Rows after cleaning: {len(self.df)}")
        return self

    def filter_necessary_columns(self, columns: list[str] | None = None) -> FavelasDataCleaner:
        """Keep only necessary columns for analysis."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        if columns:
            # Validate that all columns exist
            missing_cols = set(columns) - set(self.df.columns)
            if missing_cols:
                print(f"Warning: Columns not found in dataset: {missing_cols}")
                columns = [col for col in columns if col in self.df.columns]
            
            if columns:
                print(f"\n=== Filtering to {len(columns)} columns ===")
                self.df = self.df.select(columns)
                print(f"Selected columns: {columns}")
        else:
            print("\nNo column filter applied. All columns retained.")

        return self

    def filter_rows(self, condition: pl.Expr) -> FavelasDataCleaner:
        """Apply custom row filter using Polars expression."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        initial_rows: int = len(self.df)
        self.df = self.df.filter(condition)
        removed_rows = initial_rows - len(self.df)
        print(f"\n=== Row Filter Applied ===")
        print(f"Removed: {removed_rows} rows ({removed_rows/initial_rows*100:.1f}%)")
        print(f"Retained: {len(self.df)} of {initial_rows} rows ({len(self.df)/initial_rows*100:.1f}%)")
        return self

    def remove_high_null_columns(self, threshold: float = 0.5) -> FavelasDataCleaner:
        """Remove columns with null percentage above threshold."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        print(f"\n=== Removing columns with >{threshold*100}% nulls ===")
        
        if len(self.df) == 0:
            print("Warning: DataFrame is empty, skipping null column removal")
            return self
        
        null_percentages: pl.DataFrame = self.df.null_count() / len(self.df)

        columns_to_keep: list[str] = [
            col for col, null_pct in zip(self.df.columns, null_percentages.row(0))
            if null_pct <= threshold
        ]

        removed: set[str] = set(self.df.columns) - set(columns_to_keep)
        if removed:
            print(f"Removing {len(removed)} columns: {removed}")
            self.df = self.df.select(columns_to_keep)
        else:
            print("No columns removed (all columns below threshold)")

        return self

    def add_geographic_aggregations(self) -> FavelasDataCleaner:
        """Add aggregated statistics by geographic level."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        print("\n=== Adding Geographic Aggregations ===")
        
        # Count favelas/comunidades per municipality
        mun_counts = self.df.group_by(["CD_MUN", "NM_MUN", "CD_UF", "NM_UF"]).agg([
            pl.col("CD_FCU").n_unique().alias("total_fcu_mun"),
            pl.col("CD_SETOR").n_unique().alias("total_setores_mun")
        ])
        
        # Count per UF
        uf_counts = self.df.group_by(["CD_UF", "NM_UF"]).agg([
            pl.col("CD_FCU").n_unique().alias("total_fcu_uf"),
            pl.col("CD_SETOR").n_unique().alias("total_setores_uf"),
            pl.col("CD_MUN").n_unique().alias("total_municipios_uf")
        ])
        
        # Join aggregations back to main dataframe
        self.df = (
            self.df
            .join(mun_counts, on=["CD_MUN", "NM_MUN", "CD_UF", "NM_UF"], how="left")
            .join(uf_counts, on=["CD_UF", "NM_UF"], how="left")
        )
        
        print(f"Added geographic aggregations:")
        print(f"  - Municipality level: total_fcu_mun, total_setores_mun")
        print(f"  - State level: total_fcu_uf, total_setores_uf, total_municipios_uf")
        
        return self

    def save_cleaned_data(
        self,
        output_path: str,
        format: Literal["parquet", "csv", "excel"] = "parquet"
    ) -> FavelasDataCleaner:
        """Save cleaned data to file."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        output_file: Path = Path(output_path)
        print(f"\n=== Saving cleaned data to {output_file} ===")
        print(f"Format: {format}")
        print(f"Rows: {len(self.df)}")
        print(f"Columns: {len(self.df.columns)}")

        # Create parent directory if it doesn't exist
        output_file.parent.mkdir(parents=True, exist_ok=True)

        if format == "parquet":
            self.df.write_parquet(output_file)
        elif format == "csv":
            self.df.write_csv(output_file)
        elif format == "excel":
            self.df.write_excel(output_file)
        else:
            raise ValueError(f"Unsupported format: {format}")

        print(f"✓ Saved successfully to {output_file}")
        return self

    def get_dataframe(self) -> pl.DataFrame | None:
        """Return the cleaned DataFrame."""
        return self.df


def main() -> None:
    """
    Main pipeline for cleaning Favelas e Comunidades Urbanas 2022 data.
    
    This dataset contains information about favelas and urban communities
    mapped to census sectors (setores censitários) across Brazil.
    """
    
    # Initialize cleaner with the input file
    cleaner: FavelasDataCleaner = FavelasDataCleaner(
        "FavelaseComunidadesUrbanas2022Setores_20250417.xlsx",
        sheet_id=2  # Data is in the second sheet
    )

    # Load and inspect the data
    print("=" * 70)
    print("FAVELAS E COMUNIDADES URBANAS 2022 - DATA CLEANING PIPELINE")
    print("=" * 70)
    
    cleaner.load_data().show_info()

    # Apply basic cleaning operations
    cleaner.clean_data()
    
    # Remove columns with high null percentage (if any)
    cleaner.remove_high_null_columns(threshold=0.5)

    # Apply data quality filters
    print("\n=== Applying Data Quality Filters ===")
    print("Filtering for valid geographic codes...")
    
    cleaner.filter_rows(
        # Ensure all codes are positive
        (pl.col('CD_SETOR') > 0) &
        (pl.col('CD_FCU') > 0) &
        (pl.col('CD_MUN') > 0) &
        (pl.col('CD_UF') > 0) &
        
        # Ensure names are not empty
        (pl.col('NM_FCU').str.len_chars() > 0) &
        (pl.col('NM_MUN').str.len_chars() > 0) &
        (pl.col('NM_UF').str.len_chars() > 0) &
        
        # Valid UF codes (11-53, Brazilian state codes)
        (pl.col('CD_UF') >= 11) &
        (pl.col('CD_UF') <= 53)
    )

    # Add geographic aggregations
    cleaner.add_geographic_aggregations()

    # Show final result
    df: pl.DataFrame | None = cleaner.get_dataframe()
    if df is not None:
        print(f"\n{'=' * 70}")
        print("FINAL CLEANED DATASET SUMMARY")
        print(f"{'=' * 70}")
        print(f"Total rows: {len(df):,}")
        print(f"Total columns: {len(df.columns)}")
        print(f"\nUnique counts:")
        print(f"  - Setores Censitários: {df['CD_SETOR'].n_unique():,}")
        print(f"  - Favelas/Comunidades: {df['CD_FCU'].n_unique():,}")
        print(f"  - Municípios: {df['CD_MUN'].n_unique():,}")
        print(f"  - Estados (UFs): {df['CD_UF'].n_unique()}")
        
        print(f"\nTop 10 municipalities by number of favelas/comunidades:")
        top_mun = (
            df.group_by(["NM_MUN", "NM_UF"])
            .agg(pl.col("CD_FCU").n_unique().alias("total_fcu"))
            .sort("total_fcu", descending=True)
            .head(10)
        )
        print(top_mun)
        
        print(f"\nSample of cleaned data:")
        print(df.head(5))

    # Save cleaned data in multiple formats
    cleaner.save_cleaned_data("favelas_comunidades_2022_cleaned.parquet", format="parquet")
    cleaner.save_cleaned_data("favelas_comunidades_2022_cleaned.csv", format="csv")
    
    print(f"\n{'=' * 70}")
    print("✓ CLEANING PIPELINE COMPLETED SUCCESSFULLY")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
