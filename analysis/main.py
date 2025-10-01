"""Analyze Atlas Brasil 2010 data and print key statistics to console."""
from __future__ import annotations
import polars as pl


def main() -> None:
    """Load data and print relevant statistics for manual analysis."""
    df = pl.read_parquet('../atlas2010_cleaned.parquet')

    print("="*80)
    print("ATLAS BRASIL 2010 - DATA ANALYSIS")
    print("="*80)

    # Dataset overview
    print(f"\n📊 DATASET OVERVIEW")
    print(f"Total municipalities: {len(df):,}")
    print(f"Total states: {df['uf'].n_unique()}")
    print(f"Columns: {', '.join(df.columns)}")

    # National averages
    print(f"\n📈 NATIONAL AVERAGES")
    nat_avg = df.select([
        pl.col('espvida').mean().alias('Life Expectancy'),
        pl.col('mort1').mean().alias('Infant Mortality'),
        pl.col('e_anosestudo').mean().alias('Years of Schooling'),
        pl.col('t_analf18m').mean().alias('Illiteracy Rate'),
        pl.col('renda_per_capita').mean().alias('Per Capita Income'),
        pl.col('fectot').mean().alias('Fertility Rate'),
        pl.col('populacao_total').sum().alias('Total Population')
    ])

    print(f"Life Expectancy: {nat_avg['Life Expectancy'][0]:.2f} years")
    print(f"Infant Mortality: {nat_avg['Infant Mortality'][0]:.2f} per 1,000 births")
    print(f"Years of Schooling: {nat_avg['Years of Schooling'][0]:.2f} years")
    print(f"Illiteracy Rate (18+): {nat_avg['Illiteracy Rate'][0]:.2f}%")
    print(f"Per Capita Income: R$ {nat_avg['Per Capita Income'][0]:.2f}")
    print(f"Fertility Rate: {nat_avg['Fertility Rate'][0]:.2f} children/woman")
    print(f"Total Population: {nat_avg['Total Population'][0]:,.0f}")

    # State-level statistics
    print(f"\n🏛️  STATE-LEVEL STATISTICS")
    state_stats = df.group_by('uf').agg([
        pl.count('nome_mun').alias('municipalities'),
        pl.col('populacao_total').sum().alias('population'),
        pl.col('espvida').mean().alias('life_exp'),
        pl.col('mort1').mean().alias('infant_mort'),
        pl.col('e_anosestudo').mean().alias('schooling'),
        pl.col('t_analf18m').mean().alias('illiteracy'),
        pl.col('renda_per_capita').mean().alias('income')
    ]).sort('life_exp', descending=True)

    print("\n🥇 TOP 10 STATES (by Life Expectancy):")
    print(state_stats.head(10))

    print("\n🔴 BOTTOM 10 STATES (by Life Expectancy):")
    print(state_stats.tail(10).reverse())

    # Municipal extremes
    print(f"\n🏘️  MUNICIPAL EXTREMES")

    print("\n✨ TOP 10 MUNICIPALITIES (by Life Expectancy):")
    top_munic = df.select([
        'uf', 'nome_mun', 'espvida', 'renda_per_capita', 'e_anosestudo',
        't_analf18m', 'mort1', 'populacao_total'
    ]).sort('espvida', descending=True).head(10)
    print(top_munic)

    print("\n⚠️  BOTTOM 10 MUNICIPALITIES (by Life Expectancy):")
    bottom_munic = df.select([
        'uf', 'nome_mun', 'espvida', 'renda_per_capita', 'e_anosestudo',
        't_analf18m', 'mort1', 'populacao_total'
    ]).sort('espvida').head(10)
    print(bottom_munic)

    # Correlations
    print(f"\n📊 CORRELATION ANALYSIS")

    print("\n🔗 Correlation Coefficients:")
    correlations = df.select([
        pl.corr('renda_per_capita', 'espvida').alias('Income vs Life Exp'),
        pl.corr('renda_per_capita', 'mort1').alias('Income vs Infant Mort'),
        pl.corr('renda_per_capita', 'mort5').alias('Income vs Child Mort'),
        pl.corr('mort1', 'espvida').alias('Infant Mort vs Life Exp'),
        pl.corr('mort5', 'espvida').alias('Child Mort vs Life Exp'),
        pl.corr('e_anosestudo', 'espvida').alias('Education vs Life Exp')
    ])
    print(correlations)

    print("\n💰 Income Quartile Analysis:")
    income_quartiles = df.select([
        pl.col('renda_per_capita').quantile(0.25).alias('Q1_25%'),
        pl.col('renda_per_capita').quantile(0.50).alias('Q2_50%'),
        pl.col('renda_per_capita').quantile(0.75).alias('Q3_75%')
    ])
    print("Quartile Thresholds:")
    print(income_quartiles)

    q1_threshold = income_quartiles['Q1_25%'][0]
    q2_threshold = income_quartiles['Q2_50%'][0]
    q3_threshold = income_quartiles['Q3_75%'][0]

    quartile_stats = df.select([
        pl.when(pl.col('renda_per_capita') <= q1_threshold)
          .then(pl.lit('Q1_Poorest'))
          .when(pl.col('renda_per_capita') <= q2_threshold)
          .then(pl.lit('Q2_Low'))
          .when(pl.col('renda_per_capita') <= q3_threshold)
          .then(pl.lit('Q3_Medium'))
          .otherwise(pl.lit('Q4_Richest'))
          .alias('quartile'),
        pl.col('renda_per_capita'),
        pl.col('espvida'),
        pl.col('mort1'),
        pl.col('mort5')
    ]).group_by('quartile').agg([
        pl.count('renda_per_capita').alias('municipalities'),
        pl.col('renda_per_capita').mean().alias('avg_income'),
        pl.col('espvida').mean().alias('avg_life_exp'),
        pl.col('mort1').mean().alias('avg_infant_mort'),
        pl.col('mort5').mean().alias('avg_child_mort')
    ]).sort('quartile')

    print("\nHealth Outcomes by Income Quartile:")
    print(quartile_stats)

    print("\n🔴 Extreme Poverty (income < R$ 200):")
    extreme_poor = df.filter(pl.col('renda_per_capita') < 200).select([
        pl.count('nome_mun').alias('count'),
        pl.col('renda_per_capita').mean().alias('avg_income'),
        pl.col('espvida').mean().alias('avg_life_exp'),
        pl.col('mort1').mean().alias('avg_infant_mort'),
        pl.col('mort5').mean().alias('avg_child_mort')
    ])
    print(extreme_poor)

    print("\n🟢 High Income (income > R$ 1000):")
    high_income = df.filter(pl.col('renda_per_capita') > 1000).select([
        pl.count('nome_mun').alias('count'),
        pl.col('renda_per_capita').mean().alias('avg_income'),
        pl.col('espvida').mean().alias('avg_life_exp'),
        pl.col('mort1').mean().alias('avg_infant_mort'),
        pl.col('mort5').mean().alias('avg_child_mort')
    ])
    print(high_income)

    print("\n👶 Child Mortality Groups:")
    mort_groups = df.select([
        pl.when(pl.col('mort1') < 15)
          .then(pl.lit('Low_<15'))
          .when(pl.col('mort1') < 25)
          .then(pl.lit('Medium_15-25'))
          .otherwise(pl.lit('High_>25'))
          .alias('mort_group'),
        pl.col('mort1'),
        pl.col('mort5'),
        pl.col('espvida'),
        pl.col('renda_per_capita')
    ]).group_by('mort_group').agg([
        pl.count('mort1').alias('municipalities'),
        pl.col('mort1').mean().alias('avg_infant_mort'),
        pl.col('mort5').mean().alias('avg_child_mort'),
        pl.col('espvida').mean().alias('avg_life_exp'),
        pl.col('renda_per_capita').mean().alias('avg_income')
    ]).sort('mort_group')
    print(mort_groups)

    print("\n📍 Extreme Cases:")
    print("\nLowest Income Municipalities:")
    print(df.sort('renda_per_capita').head(5).select([
        'uf', 'nome_mun', 'renda_per_capita', 'mort1', 'espvida'
    ]))

    print("\nHighest Income Municipalities:")
    print(df.sort('renda_per_capita', descending=True).head(5).select([
        'uf', 'nome_mun', 'renda_per_capita', 'mort1', 'espvida'
    ]))

    # Regional patterns
    print(f"\n🗺️  REGIONAL PATTERNS")
    print("\nStates by region (approximate classification):")

    northeast = ['Alagoas', 'Bahia', 'Ceara', 'Maranhao', 'Paraiba', 'Pernambuco', 'Piaui', 'Sergipe', 'Rio Grande do Norte']
    south = ['Rio Grande do Sul', 'Santa Catarina', 'Parana']
    southeast = ['Sao Paulo', 'Rio de Janeiro', 'Minas Gerais', 'Espirito Santo']

    northeast_df = df.filter(pl.col('uf').is_in(northeast))
    south_df = df.filter(pl.col('uf').is_in(south))
    southeast_df = df.filter(pl.col('uf').is_in(southeast))

    print(f"\nNortheast ({len(northeast_df)} municipalities):")
    print(f"  Avg Life Expectancy: {northeast_df['espvida'].mean():.2f} years")
    print(f"  Avg Income: R$ {northeast_df['renda_per_capita'].mean():.2f}")
    print(f"  Avg Illiteracy: {northeast_df['t_analf18m'].mean():.2f}%")

    print(f"\nSouth ({len(south_df)} municipalities):")
    print(f"  Avg Life Expectancy: {south_df['espvida'].mean():.2f} years")
    print(f"  Avg Income: R$ {south_df['renda_per_capita'].mean():.2f}")
    print(f"  Avg Illiteracy: {south_df['t_analf18m'].mean():.2f}%")

    print(f"\nSoutheast ({len(southeast_df)} municipalities):")
    print(f"  Avg Life Expectancy: {southeast_df['espvida'].mean():.2f} years")
    print(f"  Avg Income: R$ {southeast_df['renda_per_capita'].mean():.2f}")
    print(f"  Avg Illiteracy: {southeast_df['t_analf18m'].mean():.2f}%")

    # Inequality metrics
    print(f"\n⚖️  INEQUALITY METRICS")
    print(f"Life Expectancy Range: {df['espvida'].min():.2f} - {df['espvida'].max():.2f} years (gap: {df['espvida'].max() - df['espvida'].min():.2f})")
    print(f"Income Range: R$ {df['renda_per_capita'].min():.2f} - R$ {df['renda_per_capita'].max():.2f} (ratio: {df['renda_per_capita'].max() / df['renda_per_capita'].min():.1f}x)")
    print(f"Illiteracy Range: {df['t_analf18m'].min():.2f}% - {df['t_analf18m'].max():.2f}%")
    print(f"Infant Mortality Range: {df['mort1'].min():.2f} - {df['mort1'].max():.2f} per 1,000")

    print("\n" + "="*80)
    print("✓ Analysis complete. Use this data to construct your manual analysis.")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
