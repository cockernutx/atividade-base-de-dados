from __future__ import annotations

import polars as pl
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path


def load_data(file_path: str) -> pl.DataFrame:
    """Load cleaned data from parquet file."""
    path = Path(file_path)
    if not path.exists():
        st.error(f"Data file not found: {file_path}")
        st.info("Please run the data cleaner first: `uv run python data_cleaner/main.py`")
        st.stop()
    return pl.read_parquet(path)


def show_overview(df: pl.DataFrame) -> None:
    """Display dataset overview and main statistics."""
    st.header("📊 Dataset Overview")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Municipalities", f"{len(df):,}")
    with col2:
        st.metric("Total Population", f"{df['populacao_total'].sum():,.0f}")
    with col3:
        st.metric("States Covered", df['uf'].n_unique())

    st.subheader("Dataset Information")
    st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
    st.write(f"**Year:** {df['ano'].unique().to_list()}")

    # Display column descriptions
    st.subheader("Column Descriptions")
    descriptions = {
        "ano": "Year of data collection",
        "uf": "State (Unidade Federativa)",
        "nome_mun": "Municipality name",
        "espvida": "Life expectancy at birth (years)",
        "fectot": "Total fertility rate",
        "mort1": "Infant mortality rate (per 1,000 live births)",
        "mort5": "Child mortality rate under 5 (per 1,000)",
        "sobre60": "Probability of survival to age 60 (%)",
        "e_anosestudo": "Average years of schooling",
        "t_analf18m": "Illiteracy rate for 18+ population (%)",
        "renda_per_capita": "Per capita income (BRL)",
        "populacao_total": "Total population"
    }

    desc_df = pl.DataFrame({
        "Column": list(descriptions.keys()),
        "Description": list(descriptions.values())
    })
    st.dataframe(desc_df, width='stretch', hide_index=True)


def show_health_analysis(df: pl.DataFrame) -> None:
    """Display health indicators analysis."""
    st.header("🏥 Health Indicators Analysis")

    # Convert to pandas for plotly
    pdf = df.to_pandas()

    # Life expectancy distribution
    st.subheader("Life Expectancy Distribution")
    st.write("""
    Life expectancy varies significantly across Brazilian municipalities,
    reflecting differences in healthcare access, living conditions, and socioeconomic factors.
    """)

    fig = px.histogram(
        pdf,
        x='espvida',
        nbins=50,
        title='Life Expectancy Distribution',
        labels={'espvida': 'Life Expectancy (years)', 'count': 'Number of Municipalities'},
        color_discrete_sequence=['#1f77b4']
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, config={'displayModeBar': False})
    st.download_button(
        label="Download chart as HTML",
        data=fig.to_html(),
        file_name="life_expectancy_distribution.html",
        mime="text/html"
    )

    # Statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Mean", f"{df['espvida'].mean():.2f} years")
    with col2:
        st.metric("Median", f"{df['espvida'].median():.2f} years")
    with col3:
        st.metric("Minimum", f"{df['espvida'].min():.2f} years")
    with col4:
        st.metric("Maximum", f"{df['espvida'].max():.2f} years")

    # Mortality rates
    st.subheader("Child Mortality")
    st.write("""
    Mortality rates are key indicators of healthcare quality and access.
    Lower rates indicate better health services and living conditions.
    """)

    fig = go.Figure()
    fig.add_trace(go.Box(y=pdf['mort1'], name='Under 1 year', marker_color='#ff7f0e'))
    fig.add_trace(go.Box(y=pdf['mort5'], name='Under 5 years', marker_color='#2ca02c'))
    fig.update_layout(
        title='Mortality Rates Distribution',
        yaxis_title='Deaths per 1,000 live births',
        showlegend=True
    )
    st.plotly_chart(fig, config={'displayModeBar': False})
    st.download_button(
        label="Download chart as HTML",
        data=fig.to_html(),
        file_name="mortality_rates_distribution.html",
        mime="text/html"
    )

    # Survival to 60
    st.subheader("Survival to Age 60")
    st.write("""
    The probability of survival to age 60 reflects long-term health outcomes
    and quality of life throughout adulthood.
    """)

    fig = px.violin(
        pdf,
        y='sobre60',
        box=True,
        title='Probability of Survival to Age 60 Distribution',
        labels={'sobre60': 'Survival Probability (%)'},
        color_discrete_sequence=['#d62728']
    )
    st.plotly_chart(fig, config={'displayModeBar': False})
    st.download_button(
        label="Download chart as HTML",
        data=fig.to_html(),
        file_name="survival_to_60_distribution.html",
        mime="text/html"
    )


def show_education_analysis(df: pl.DataFrame) -> None:
    """Display education indicators analysis."""
    st.header("📚 Education Indicators Analysis")

    pdf = df.to_pandas()

    # Years of schooling
    st.subheader("Average Years of Schooling")
    st.write("""
    Education levels vary widely across municipalities,
    reflecting disparities in educational access and quality.
    """)

    fig = px.histogram(
        pdf,
        x='e_anosestudo',
        nbins=40,
        title='Average Years of Schooling Distribution',
        labels={'e_anosestudo': 'Years of Schooling', 'count': 'Number of Municipalities'},
        color_discrete_sequence=['#9467bd']
    )
    st.plotly_chart(fig, config={'displayModeBar': False})
    st.download_button(
        label="Download chart as HTML",
        data=fig.to_html(),
        file_name="years_of_schooling_distribution.html",
        mime="text/html"
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Mean", f"{df['e_anosestudo'].mean():.2f} years")
    with col2:
        st.metric("Median", f"{df['e_anosestudo'].median():.2f} years")
    with col3:
        st.metric("Std Dev", f"{df['e_anosestudo'].std():.2f} years")

    # Illiteracy rate
    st.subheader("Illiteracy Rate (18+ population)")
    st.write("""
    Adult illiteracy rates indicate historical educational challenges
    and barriers to basic education access.
    """)

    fig = px.box(
        pdf,
        y='t_analf18m',
        title='Illiteracy Rates Distribution',
        labels={'t_analf18m': 'Illiteracy Rate (%)'},
        color_discrete_sequence=['#8c564b']
    )
    st.plotly_chart(fig, config={'displayModeBar': False})
    st.download_button(
        label="Download chart as HTML",
        data=fig.to_html(),
        file_name="illiteracy_rates_distribution.html",
        mime="text/html"
    )

    # Correlation between education metrics
    st.subheader("Relationship Between Education Metrics")
    fig = px.scatter(
        pdf,
        x='e_anosestudo',
        y='t_analf18m',
        title='Years of Schooling vs Illiteracy Rate',
        labels={
            'e_anosestudo': 'Average Years of Schooling',
            't_analf18m': 'Illiteracy Rate (%)'
        },
        opacity=0.5,
        color_discrete_sequence=['#e377c2']
    )
    st.plotly_chart(fig, config={'displayModeBar': False})
    st.download_button(
        label="Download chart as HTML",
        data=fig.to_html(),
        file_name="schooling_vs_illiteracy.html",
        mime="text/html"
    )


def show_economic_analysis(df: pl.DataFrame) -> None:
    """Display economic indicators analysis."""
    st.header("💰 Economic Indicators Analysis")

    pdf = df.to_pandas()

    st.subheader("Per Capita Income Distribution")
    st.write("""
    Income inequality is evident across Brazilian municipalities,
    with significant variations in economic development and opportunities.
    """)

    fig = px.histogram(
        pdf,
        x='renda_per_capita',
        nbins=50,
        title='Per Capita Income Distribution',
        labels={'renda_per_capita': 'Per Capita Income (BRL)', 'count': 'Number of Municipalities'},
        color_discrete_sequence=['#2ca02c']
    )
    st.plotly_chart(fig, config={'displayModeBar': False})
    st.download_button(
        label="Download chart as HTML",
        data=fig.to_html(),
        file_name="income_distribution.html",
        mime="text/html"
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Mean", f"R$ {df['renda_per_capita'].mean():.2f}")
    with col2:
        st.metric("Median", f"R$ {df['renda_per_capita'].median():.2f}")
    with col3:
        st.metric("Minimum", f"R$ {df['renda_per_capita'].min():.2f}")
    with col4:
        st.metric("Maximum", f"R$ {df['renda_per_capita'].max():.2f}")

    # Income vs Life Expectancy
    st.subheader("Income vs Life Expectancy")
    st.write("""
    Higher income generally correlates with better health outcomes,
    reflecting improved access to healthcare, nutrition, and living conditions.
    """)

    fig = px.scatter(
        pdf,
        x='renda_per_capita',
        y='espvida',
        title='Per Capita Income vs Life Expectancy',
        labels={
            'renda_per_capita': 'Per Capita Income (BRL)',
            'espvida': 'Life Expectancy (years)'
        },
        opacity=0.5,
        color='espvida',
        color_continuous_scale='Viridis'
    )
    st.plotly_chart(fig, config={'displayModeBar': False})
    st.download_button(
        label="Download chart as HTML",
        data=fig.to_html(),
        file_name="income_vs_life_expectancy.html",
        mime="text/html"
    )

    # Income vs Education
    st.subheader("Income vs Education")
    st.write("""
    Education and income are closely linked,
    with better education leading to higher earning potential.
    """)

    fig = px.scatter(
        pdf,
        x='e_anosestudo',
        y='renda_per_capita',
        title='Years of Schooling vs Per Capita Income',
        labels={
            'e_anosestudo': 'Average Years of Schooling',
            'renda_per_capita': 'Per Capita Income (BRL)'
        },
        opacity=0.5,
        color='renda_per_capita',
        color_continuous_scale='Plasma'
    )
    st.plotly_chart(fig, config={'displayModeBar': False})
    st.download_button(
        label="Download chart as HTML",
        data=fig.to_html(),
        file_name="schooling_vs_income.html",
        mime="text/html"
    )


def show_regional_analysis(df: pl.DataFrame) -> None:
    """Display regional comparison analysis."""
    st.header("🗺️ Regional Analysis")

    pdf = df.to_pandas()

    st.subheader("Comparison by State")
    st.write("""
    Regional disparities reflect historical, economic, and geographic differences
    among Brazilian states.
    """)

    # Select indicator
    indicator = st.selectbox(
        "Select indicator to compare:",
        options=[
            ('espvida', 'Life Expectancy'),
            ('renda_per_capita', 'Per Capita Income'),
            ('e_anosestudo', 'Years of Schooling'),
            ('t_analf18m', 'Illiteracy Rate'),
            ('mort1', 'Infant Mortality'),
            ('fectot', 'Fertility Rate')
        ],
        format_func=lambda x: x[1]
    )

    col_name, col_label = indicator

    # Calculate state averages
    state_avg = df.group_by('uf').agg(
        pl.col(col_name).mean().alias('avg_value'),
        pl.col('populacao_total').sum().alias('total_pop')
    ).sort('avg_value', descending=True)

    state_pdf = state_avg.to_pandas()

    # Invert color scale for negative indicators (lower is better)
    reverse_indicators = ['mort1', 'mort5', 't_analf18m']
    color_scale = 'RdYlGn_r' if col_name in reverse_indicators else 'RdYlGn'

    fig = px.bar(
        state_pdf,
        x='uf',
        y='avg_value',
        title=f'Average {col_label} by State',
        labels={'uf': 'State', 'avg_value': col_label},
        color='avg_value',
        color_continuous_scale=color_scale
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, config={'displayModeBar': False})
    st.download_button(
        label="Download chart as HTML",
        data=fig.to_html(),
        file_name=f"state_comparison_{col_name}.html",
        mime="text/html"
    )

    # Top and bottom municipalities
    st.subheader(f"Top 10 Municipalities - {col_label}")
    top10 = df.select(['nome_mun', 'uf', col_name]).sort(col_name, descending=True).head(10)
    st.dataframe(top10.to_pandas(), width='stretch', hide_index=True)

    st.subheader(f"Bottom 10 Municipalities - {col_label}")
    bottom10 = df.select(['nome_mun', 'uf', col_name]).sort(col_name).head(10)
    st.dataframe(bottom10.to_pandas(), width='stretch', hide_index=True)


def show_demographic_analysis(df: pl.DataFrame) -> None:
    """Display demographic analysis."""
    st.header("👥 Demographic Analysis")

    pdf = df.to_pandas()

    st.subheader("Population Distribution")
    st.write("""
    Most Brazilian municipalities have small populations,
    with a few large urban centers concentrating significant population.
    """)

    fig = px.histogram(
        pdf,
        x='populacao_total',
        nbins=50,
        title='Municipal Population Distribution',
        labels={'populacao_total': 'Population', 'count': 'Number of Municipalities'},
        log_y=True,
        color_discrete_sequence=['#17becf']
    )
    st.plotly_chart(fig, config={'displayModeBar': False})
    st.download_button(
        label="Download chart as HTML",
        data=fig.to_html(),
        file_name="population_distribution.html",
        mime="text/html"
    )

    # Largest municipalities
    st.subheader("Top 20 Most Populous Municipalities")
    largest = df.select(['nome_mun', 'uf', 'populacao_total']).sort('populacao_total', descending=True).head(20)
    largest_pdf = largest.to_pandas()

    fig = px.bar(
        largest_pdf,
        x='nome_mun',
        y='populacao_total',
        title='Top 20 Most Populous Municipalities',
        labels={'nome_mun': 'Municipality', 'populacao_total': 'Population'},
        color='populacao_total',
        color_continuous_scale='Blues'
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, config={'displayModeBar': False})
    st.download_button(
        label="Download chart as HTML",
        data=fig.to_html(),
        file_name="top_20_populous_municipalities.html",
        mime="text/html"
    )

    # Fertility rate
    st.subheader("Fertility Rate Analysis")
    st.write("""
    Fertility rates indicate demographic trends and family planning patterns
    across different regions and socioeconomic contexts.
    """)

    fig = px.box(
        pdf,
        y='fectot',
        title='Total Fertility Rate Distribution',
        labels={'fectot': 'Total Fertility Rate'},
        color_discrete_sequence=['#bcbd22']
    )
    st.plotly_chart(fig, config={'displayModeBar': False})
    st.download_button(
        label="Download chart as HTML",
        data=fig.to_html(),
        file_name="fertility_rate_distribution.html",
        mime="text/html"
    )


def show_data_explorer(df: pl.DataFrame) -> None:
    """Interactive data explorer."""
    st.header("🔍 Data Explorer")

    st.write("""
    Explore the raw data and apply custom filters to analyze specific municipalities or regions.
    """)

    # Filters
    st.subheader("Filters")

    col1, col2 = st.columns(2)

    with col1:
        selected_states = st.multiselect(
            "Select States:",
            options=sorted(df['uf'].unique().to_list()),
            default=None
        )

    with col2:
        pop_range = st.slider(
            "Population Range:",
            min_value=int(df['populacao_total'].min()),
            max_value=int(df['populacao_total'].max()),
            value=(int(df['populacao_total'].min()), int(df['populacao_total'].max()))
        )

    # Apply filters
    filtered_df = df

    if selected_states:
        filtered_df = filtered_df.filter(pl.col('uf').is_in(selected_states))

    filtered_df = filtered_df.filter(
        (pl.col('populacao_total') >= pop_range[0]) &
        (pl.col('populacao_total') <= pop_range[1])
    )

    st.write(f"**Showing {len(filtered_df)} of {len(df)} municipalities**")

    # Display data
    st.dataframe(filtered_df.to_pandas(), width='stretch', hide_index=True)

    # Download option
    csv = filtered_df.write_csv()
    st.download_button(
        label="Download filtered data as CSV",
        data=csv,
        file_name="atlas2010_filtered.csv",
        mime="text/csv"
    )


def main() -> None:
    """Main dashboard application."""
    st.set_page_config(
        page_title="Atlas Brasil 2010 Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("📊 Atlas Brasil 2010 - Municipal Socioeconomic Dashboard")
    st.markdown("""
    This dashboard provides an interactive analysis of Brazilian municipal socioeconomic data from 2010.
    The data includes health, education, economic, and demographic indicators for all Brazilian municipalities.
    """)

    # Load data
    df = load_data("atlas2010_cleaned.parquet")

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Analysis:",
        [
            "📊 Overview",
            "🏥 Health Indicators",
            "📚 Education Indicators",
            "💰 Economic Indicators",
            "🗺️ Regional Analysis",
            "👥 Demographic Analysis",
            "🔍 Data Explorer"
        ]
    )

    # Display selected page
    if page == "📊 Overview":
        show_overview(df)
    elif page == "🏥 Health Indicators":
        show_health_analysis(df)
    elif page == "📚 Education Indicators":
        show_education_analysis(df)
    elif page == "💰 Economic Indicators":
        show_economic_analysis(df)
    elif page == "🗺️ Regional Analysis":
        show_regional_analysis(df)
    elif page == "👥 Demographic Analysis":
        show_demographic_analysis(df)
    elif page == "🔍 Data Explorer":
        show_data_explorer(df)

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **Data Source:** Atlas do Desenvolvimento Humano no Brasil 2010

    **About:** This dashboard visualizes socioeconomic indicators of Brazilian municipalities.
    """)


if __name__ == "__main__":
    main()
