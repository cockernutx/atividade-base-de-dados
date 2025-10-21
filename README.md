# Atlas Brasil 2010 - Data Analysis Project

This project provides tools for cleaning, filtering, and visualizing socioeconomic data from Brazilian municipalities based on the Atlas do Desenvolvimento Humano no Brasil 2010.

## Project Structure

```
.
data_cleaner/          # Data cleaning and filtering module
 main.py           # Data cleaner implementation
dashboard/            # Interactive visualization dashboard
 main.py          # Streamlit dashboard application
 atlas2010_dashboard.xlsx  # Source data file
```

## Features

### Data Cleaner (`data_cleaner/main.py`)

The `AtlasDataCleaner` class provides:

- **Data Loading**: Read Excel files with Polars
- **Data Cleaning**: Remove duplicates and null rows
- **Column Filtering**: Remove columns with high null percentages
- **Row Filtering**: Apply custom filters using Polars expressions
- **Data Quality Filters**: Validate data ranges for:
  - Demographic indicators (population, year)
  - Health indicators (life expectancy, mortality rates, survival rates)
  - Fertility rates
  - Education indicators (years of schooling, illiteracy rates)
  - Economic indicators (per capita income)
- **Export**: Save cleaned data in multiple formats (Parquet, CSV, Excel)

### Interactive Dashboard (`dashboard/main.py`)

A comprehensive Streamlit dashboard with multiple analysis views:

- **Overview**: Dataset summary and column descriptions
- **Health Indicators**: Life expectancy, mortality rates, survival analysis
- **Education Indicators**: Schooling years, illiteracy rates, correlations
- **Economic Indicators**: Income distribution and correlations with health/education
- **Regional Analysis**: State-level comparisons with customizable indicators
- **Demographic Analysis**: Population distribution and fertility rates
- **Data Explorer**: Interactive filtering and data export

## Dataset

The dataset contains socioeconomic data for 5,565 Brazilian municipalities from 2010, including:

| Column | Description |
|--------|-------------|
| `ano` | Year of data collection |
| `uf` | State (Unidade Federativa) |
| `nome_mun` | Municipality name |
| `espvida` | Life expectancy at birth (years) |
| `fectot` | Total fertility rate |
| `mort1` | Infant mortality rate (per 1,000 live births) |
| `mort5` | Child mortality rate under 5 (per 1,000) |
| `sobre60` | Probability of survival to age 60 (%) |
| `e_anosestudo` | Average years of schooling |
| `t_analf18m` | Illiteracy rate for 18+ population (%) |
| `renda_per_capita` | Per capita income (BRL) |
| `populacao_total` | Total population |

## Requirements

- Python 3.10+
- UV package manager
- Dependencies:
  - polars
  - streamlit
  - plotly
  - pandas
  - openpyxl

## Usage

### 1. Clean the Data

Run the data cleaner to process and filter the raw Excel file:

```bash
uv run python data_cleaner/main.py
```

This will:
- Load the raw Excel file
- Apply cleaning and validation filters
- Save the cleaned data as `favelas_comunidades_2022_cleaned.parquet` and CSV

### 2. Launch the Dashboard

Start the interactive dashboard (it will read `favelas_comunidades_2022_cleaned.parquet` by default):

```bash
uv run streamlit run dashboard/main.py
```

The dashboard will be available at `http://localhost:8501`

## Data Quality Filters

The cleaner applies the following validation filters:

- **Demographics**: Population > 0, Year = 2010
- **Health**: Life expectancy 0-100 years, mortality rates e 0, survival rates 0-100%
- **Fertility**: Rate between 0-8
- **Education**: Years of schooling 0-20, illiteracy rate 0-100%
- **Economics**: Per capita income 0-50,000 BRL

## License

Data source: Atlas do Desenvolvimento Humano no Brasil 2010 (IPEA/PNUD)

## Notes

- All text in the dashboard is in Portuguese
- The project uses Polars for efficient data processing
- Visualizations are built with Plotly for interactivity
