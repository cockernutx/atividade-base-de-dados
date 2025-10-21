from __future__ import annotations

import polars as pl
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path


def load_data(file_path: str) -> pl.DataFrame:
    """Load cleaned data from parquet file. Defaults to the favelas dataset."""
    path = Path(file_path)
    if not path.exists():
        st.error(f"Data file not found: {file_path}")
        st.info("Execute o limpador de dados primeiro: `uv run python data_cleaner/main.py`")
        st.stop()
    return pl.read_parquet(path)


def load_gini_data() -> pl.DataFrame:
    """Load cleaned Gini index data."""
    path = Path("indice_gini_cleaned.parquet")
    if not path.exists():
        st.error("Arquivo de índice de Gini não encontrado!")
        st.info("O arquivo 'indice_gini_cleaned.parquet' deve estar no diretório raiz.")
        st.stop()
    return pl.read_parquet(path)


def show_overview(df: pl.DataFrame) -> None:
    """Display dataset overview and main statistics."""
    st.header("📊 Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Setores Censitários", f"{df['CD_SETOR'].n_unique():,}")
    with col2:
        st.metric("Favelas/Comunidades", f"{df['CD_FCU'].n_unique():,}")
    with col3:
        st.metric("Municípios", f"{df['CD_MUN'].n_unique():,}")
    with col4:
        st.metric("Estados (UFs)", df['CD_UF'].n_unique())

    st.subheader("Dataset Information")
    st.write(f"**Shape:** {df.shape[0]:,} rows × {df.shape[1]} columns")
    st.write(f"**Year:** 2022 (Mapeamento 2022)")
    st.write(f"**Source:** IBGE - Favelas e Comunidades Urbanas")

    # Display column descriptions
    st.subheader("Column Descriptions")
    descriptions = {
        "CD_SETOR": "Código do Setor Censitário",
        "CD_FCU": "Código da Favela/Comunidade Urbana",
        "NM_FCU": "Nome da Favela/Comunidade",
        "CD_MUN": "Código do Município",
        "NM_MUN": "Nome do Município",
        "CD_UF": "Código da Unidade Federativa",
        "NM_UF": "Nome do Estado",
        "total_fcu_mun": "Total de Favelas/Comunidades no Município",
        "total_setores_mun": "Total de Setores no Município",
        "total_fcu_uf": "Total de Favelas/Comunidades no Estado",
        "total_setores_uf": "Total de Setores no Estado",
        "total_municipios_uf": "Total de Municípios com Favelas/Comunidades no Estado"
    }

    desc_df = pl.DataFrame({
        "Column": list(descriptions.keys()),
        "Description": list(descriptions.values())
    })
    st.dataframe(desc_df, use_container_width=True, hide_index=True)


def show_geographic_distribution(df: pl.DataFrame) -> None:
    """Display geographic distribution of favelas and communities."""
    st.header("🗺️ Distribuição Geográfica")

    pdf = df.to_pandas()

    st.subheader("Favelas e Comunidades por Estado")
    st.write("""
    A distribuição de favelas e comunidades urbanas varia significativamente entre os estados brasileiros,
    refletindo padrões de urbanização, crescimento populacional e desenvolvimento socioeconômico.
    """)

    # State-level aggregation
    state_stats = df.group_by(['CD_UF', 'NM_UF']).agg([
        pl.col('CD_FCU').n_unique().alias('total_fcu'),
        pl.col('CD_SETOR').n_unique().alias('total_setores'),
        pl.col('CD_MUN').n_unique().alias('total_municipios')
    ]).sort('total_fcu', descending=True)

    state_pdf = state_stats.to_pandas()

    fig = px.bar(
        state_pdf,
        x='NM_UF',
        y='total_fcu',
        title='Número de Favelas/Comunidades por Estado',
        labels={'NM_UF': 'Estado', 'total_fcu': 'Número de Favelas/Comunidades'},
        color='total_fcu',
        color_continuous_scale='Reds',
        text='total_fcu'
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(xaxis_tickangle=-45, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total de Estados", len(state_stats))
    with col2:
        st.metric("Média por Estado", f"{state_stats['total_fcu'].mean():.0f}")
    with col3:
        st.metric("Mediana por Estado", f"{state_stats['total_fcu'].median():.0f}")

    # Top states table
    st.subheader("Maiores 10 Estados com Mais Favelas/Comunidades")
    st.dataframe(state_pdf.head(10)[['NM_UF', 'total_fcu', 'total_setores', 'total_municipios']], 
                 use_container_width=True, hide_index=True)


def show_municipal_analysis(df: pl.DataFrame) -> None:
    """Display municipal-level analysis."""
    st.header("🏙️ Análise Municipal")

    pdf = df.to_pandas()

    st.subheader("Municípios com Mais Favelas/Comunidades")
    st.write("""
    Grandes centros urbanos concentram o maior número de favelas e comunidades,
    refletindo processos históricos de urbanização acelerada e desigualdade social.
    """)

    # Municipal aggregation
    mun_stats = df.group_by(['CD_MUN', 'NM_MUN', 'NM_UF']).agg([
        pl.col('CD_FCU').n_unique().alias('total_fcu'),
        pl.col('CD_SETOR').n_unique().alias('total_setores')
    ]).sort('total_fcu', descending=True).head(30)

    mun_pdf = mun_stats.to_pandas()
    mun_pdf['label'] = mun_pdf['NM_MUN'] + ' - ' + mun_pdf['NM_UF']

    fig = px.bar(
        mun_pdf.head(20),
        x='total_fcu',
        y='label',
        title='Maiores 20 Municípios com Mais Favelas/Comunidades',
        labels={'label': 'Município', 'total_fcu': 'Número de Favelas/Comunidades'},
        orientation='h',
        color='total_fcu',
        color_continuous_scale='Viridis',
        text='total_fcu'
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(yaxis={'categoryorder': 'total ascending'}, showlegend=False, height=600)
    st.plotly_chart(fig, use_container_width=True)

    # Statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total de Municípios", df['CD_MUN'].n_unique())
    with col2:
        top_mun = mun_stats.head(1)
        st.metric("Município Líder", f"{top_mun['NM_MUN'][0]} - {top_mun['NM_UF'][0]}")
    with col3:
        st.metric("Favelas/Comunidades", f"{top_mun['total_fcu'][0]}")

    # Detailed table
    st.subheader("Maiores 30 Municípios - Dados Detalhados")
    display_df = mun_pdf[['NM_MUN', 'NM_UF', 'total_fcu', 'total_setores']].copy()
    display_df.columns = ['Município', 'Estado', 'Favelas/Comunidades', 'Setores']
    st.dataframe(display_df, use_container_width=True, hide_index=True)


def show_community_analysis(df: pl.DataFrame) -> None:
    """Display analysis of individual communities."""
    st.header("🏘️ Análise de Comunidades")

    st.subheader("Comunidades Mais Presentes")
    st.write("""
    Algumas comunidades e favelas aparecem em múltiplos setores censitários,
    indicando áreas territorialmente extensas ou complexas.
    """)

    # Communities with most sectors
    comm_stats = df.group_by(['CD_FCU', 'NM_FCU', 'NM_MUN', 'NM_UF']).agg([
        pl.col('CD_SETOR').n_unique().alias('num_setores')
    ]).sort('num_setores', descending=True).head(30)

    comm_pdf = comm_stats.to_pandas()
    comm_pdf['label'] = comm_pdf['NM_FCU'] + ' (' + comm_pdf['NM_MUN'] + ' - ' + comm_pdf['NM_UF'] + ')'

    fig = px.bar(
        comm_pdf.head(20),
        x='num_setores',
        y='label',
        title='Maiores 20 Favelas/Comunidades com Mais Setores Censitários',
        labels={'label': 'Comunidade', 'num_setores': 'Número de Setores'},
        orientation='h',
        color='num_setores',
        color_continuous_scale='Blues',
        text='num_setores'
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(yaxis={'categoryorder': 'total ascending'}, showlegend=False, height=600)
    st.plotly_chart(fig, use_container_width=True)

    # Statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total de Comunidades Únicas", df['CD_FCU'].n_unique())
    with col2:
        st.metric("Média de Setores por Comunidade", f"{df.group_by('CD_FCU').agg(pl.col('CD_SETOR').n_unique()).select(pl.col('CD_SETOR').mean())[0,0]:.2f}")
    with col3:
        max_setores = comm_stats['num_setores'].max()
        st.metric("Máximo de Setores", max_setores)

    # Detailed table
    st.subheader("Maiores 30 Comunidades - Dados Detalhados")
    display_df = comm_pdf[['NM_FCU', 'NM_MUN', 'NM_UF', 'num_setores']].copy()
    display_df.columns = ['Nome da Comunidade', 'Município', 'Estado', 'Número de Setores']
    st.dataframe(display_df, use_container_width=True, hide_index=True)


def show_inequality_analysis(df: pl.DataFrame) -> None:
    """Display analysis of favelas/communities vs Gini index (social inequality)."""
    st.header("📊 Desigualdade Social e Favelas/Comunidades")
    
    st.write("""
    Esta análise relaciona a quantidade de favelas e comunidades urbanas por estado 
    com o Índice de Gini (2024), que mede a desigualdade de renda. 
    
    **Índice de Gini:** varia de 0 (igualdade perfeita) a 1 (desigualdade máxima).
    Valores mais altos indicam maior concentração de renda.
    """)
    
    # Load Gini data
    df_gini = load_gini_data()
    
    # Aggregate favelas data by state
    state_stats = df.group_by(['CD_UF', 'NM_UF']).agg([
        pl.col('CD_FCU').n_unique().alias('total_fcu'),
        pl.col('CD_SETOR').n_unique().alias('total_setores'),
        pl.col('CD_MUN').n_unique().alias('total_municipios')
    ])
    
    # Join with Gini data
    combined = state_stats.join(df_gini, on='CD_UF', how='left', suffix='_gini')
    
    # Sort by total favelas
    combined = combined.sort('total_fcu', descending=True)
    
    # Convert to pandas for plotting
    combined_pdf = combined.to_pandas()
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Estados Analisados", len(combined))
    with col2:
        avg_gini = combined['indice_gini'].mean()
        st.metric("Gini Médio Brasil", f"{avg_gini:.3f}")
    with col3:
        max_gini_state = combined.filter(pl.col('indice_gini') == pl.col('indice_gini').max())
        st.metric("Maior Desigualdade", f"{max_gini_state['NM_UF'][0]}: {max_gini_state['indice_gini'][0]:.3f}")
    with col4:
        min_gini_state = combined.filter(pl.col('indice_gini') == pl.col('indice_gini').min())
        st.metric("Menor Desigualdade", f"{min_gini_state['NM_UF'][0]}: {min_gini_state['indice_gini'][0]:.3f}")
    
    # Scatter plot: Gini vs Number of Favelas
    st.subheader("Correlação: Índice de Gini × Quantidade de Favelas/Comunidades")
    
    fig_scatter = px.scatter(
        combined_pdf,
        x='indice_gini',
        y='total_fcu',
        size='total_municipios',
        color='indice_gini',
        hover_name='NM_UF',
        hover_data={
            'indice_gini': ':.3f',
            'total_fcu': ':,',
            'total_setores': ':,',
            'total_municipios': ':,'
        },
        labels={
            'indice_gini': 'Índice de Gini (2024)',
            'total_fcu': 'Número de Favelas/Comunidades',
            'total_municipios': 'Municípios com Favelas'
        },
        title='Relação entre Desigualdade Social e Presença de Favelas/Comunidades',
        color_continuous_scale='RdYlGn_r',  # Reversed: Green (low inequality) to Red (high inequality)
        size_max=30
    )
    
    fig_scatter.update_layout(height=500)
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Calculate correlation
    correlation = combined.select([
        pl.corr('indice_gini', 'total_fcu').alias('correlacao')
    ])
    corr_value = correlation['correlacao'][0]
  
    
    # Dual bar chart
    st.subheader("Comparação por Estado: Gini vs Favelas/Comunidades")
    
    # Create figure with secondary y-axis
    fig_dual = go.Figure()
    
    # Sort by Gini for better visualization
    combined_sorted = combined.sort('indice_gini', descending=True)
    combined_sorted_pdf = combined_sorted.to_pandas()
    
    # Normalize Gini values for color mapping (green to red)
    gini_normalized = (combined_sorted_pdf['indice_gini'] - combined_sorted_pdf['indice_gini'].min()) / \
                      (combined_sorted_pdf['indice_gini'].max() - combined_sorted_pdf['indice_gini'].min())
    
    # Create color list from green to red
    colors_gini = [f'rgb({int(255*val)}, {int(255*(1-val))}, 0)' for val in gini_normalized]
    
    fig_dual.add_trace(go.Bar(
        name='Índice de Gini',
        x=combined_sorted_pdf['NM_UF'],
        y=combined_sorted_pdf['indice_gini'],
        yaxis='y',
        marker_color=colors_gini,
        opacity=0.8,
        text=combined_sorted_pdf['indice_gini'].round(3),
        textposition='outside'
    ))
    
    fig_dual.add_trace(go.Bar(
        name='Favelas/Comunidades (escala ajustada)',
        x=combined_sorted_pdf['NM_UF'],
        y=combined_sorted_pdf['total_fcu'] / combined_sorted_pdf['total_fcu'].max() * 0.6,  # Normalize to 0-0.6 range
        yaxis='y',
        marker_color='steelblue',
        opacity=0.5,
        text=combined_sorted_pdf['total_fcu'],
        textposition='inside'
    ))
    
    fig_dual.update_layout(
        title='Estados Ordenados por Índice de Gini',
        xaxis=dict(title='Estado', tickangle=-45),
        yaxis=dict(
            title=dict(text='Índice de Gini', font=dict(color='indianred')),
            tickfont=dict(color='indianred')
        ),
        barmode='overlay',
        height=500,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_dual, use_container_width=True)
    
    # Detailed table
    st.subheader("Tabela Completa: Estados, Gini e Favelas/Comunidades")
    
    display_df = combined_pdf[['NM_UF', 'indice_gini', 'total_fcu', 'total_setores', 'total_municipios']].copy()
    display_df.columns = ['Estado', 'Índice de Gini (2024)', 'Favelas/Comunidades', 'Setores Censitários', 'Municípios']
    display_df = display_df.sort_values('Índice de Gini (2024)', ascending=False)
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # Download option
    csv = combined.write_csv()
    st.download_button(
        label="📥 Download dados combinados (CSV)",
        data=csv,
        file_name="gini_favelas_por_estado.csv",
        mime="text/csv"
    )
    
    # Insights section
    st.subheader("💡 Insights")
    
    # Top 5 states by Gini
    top5_gini = combined.sort('indice_gini', descending=True).head(5)
    # Top 5 states by favelas
    top5_favelas = combined.sort('total_fcu', descending=True).head(5)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Top 5 Estados - Maior Desigualdade (Gini)**")
        for row in top5_gini.iter_rows(named=True):
            st.write(f"- {row['NM_UF']}: {row['indice_gini']:.3f} ({row['total_fcu']:,} favelas)")
    
    with col2:
        st.write("**Top 5 Estados - Mais Favelas/Comunidades**")
        for row in top5_favelas.iter_rows(named=True):
            st.write(f"- {row['NM_UF']}: {row['total_fcu']:,} (Gini: {row['indice_gini']:.3f})")


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
        # Use NM_UF as display values
        state_options = sorted(df['NM_UF'].unique().to_list())
        selected_states = st.multiselect(
            "Selecionar Estados:",
            options=state_options,
            default=None
        )

    with col2:
        st.write("\n")
        st.caption("Filtre por município ou comunidade usando o campo de busca abaixo")

    # Apply filters
    filtered_df = df

    if selected_states:
        filtered_df = filtered_df.filter(pl.col('NM_UF').is_in(selected_states))

    st.write(f"**Mostrando {len(filtered_df)} registros de {len(df)}**")

    # Allow search by municipality or community
    query = st.text_input("Pesquisar município ou comunidade (parte do nome)")
    if query:
        q = query.lower()
        filtered_df = filtered_df.filter(
            pl.col('NM_MUN').str.to_lowercase().str.contains(q) |
            pl.col('NM_FCU').str.to_lowercase().str.contains(q)
        )

    st.dataframe(filtered_df.to_pandas(), width='stretch', hide_index=True)

    # Download option
    csv = filtered_df.write_csv()
    st.download_button(
        label="Download dados filtrados como CSV",
        data=csv,
        file_name="favelas_comunidades_2022_filtered.csv",
        mime="text/csv"
    )


def main() -> None:
    """Main dashboard application."""
    st.set_page_config(
        page_title="Favelas e Comunidades Urbanas 2022",
        page_icon="🏘️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🏘️ Favelas e Comunidades Urbanas 2022 - Dashboard de Análise")
    st.markdown("""
    Este dashboard fornece uma análise interativa das favelas e comunidades urbanas mapeadas no Brasil em 2022.
    Os dados incluem informações sobre setores censitários, favelas/comunidades e distribuição geográfica por municípios e estados.
    """)

    # Load data
    df = load_data("favelas_comunidades_2022_cleaned.parquet")

    # Sidebar navigation
    st.sidebar.title("Navegação")
    page = st.sidebar.radio(
        "Selecione a Análise:",
        [
            "📊 Visão Geral",
            "🗺️ Distribuição Geográfica",
            "🏙️ Análise Municipal",
            "🏘️ Análise de Comunidades",
            "📈 Desigualdade Social (Gini)",
            "🔍 Explorador de Dados"
        ]
    )

    # Display selected page
    if page == "📊 Visão Geral":
        show_overview(df)
    elif page == "🗺️ Distribuição Geográfica":
        show_geographic_distribution(df)
    elif page == "🏙️ Análise Municipal":
        show_municipal_analysis(df)
    elif page == "🏘️ Análise de Comunidades":
        show_community_analysis(df)
    elif page == "📈 Desigualdade Social (Gini)":
        show_inequality_analysis(df)
    elif page == "🔍 Explorador de Dados":
        show_data_explorer(df)

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **Fonte de Dados:** IBGE - Favelas e Comunidades Urbanas (2022)

    **Sobre:** Este dashboard visualiza a distribuição de favelas e comunidades urbanas no Brasil.
    """)


if __name__ == "__main__":
    main()
