from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

st.set_page_config(
    page_title="Electronic House — Teste A/B Checkout",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

COUNTRY_NAMES = {
    "AUS": "AUS — Austrália",
    "BRA": "BRA — Brasil",
    "CAN": "CAN — Canadá",
    "DEU": "DEU — Alemanha",
    "ESP": "ESP — Espanha",
    "FRA": "FRA — França",
    "GBR": "GBR — Reino Unido",
    "MEX": "MEX — México",
    "TUR": "TUR — Turquia",
    "USA": "USA — Estados Unidos",
}

COLOR_MAP = {
    "GRP B (Manual)": "#2B5C8F",
    "GRP A (Automático)": "#E07A5F",
}


@st.cache_data
def load_data() -> pd.DataFrame:
    data_path = Path(__file__).resolve().parent / "ab_testing.csv"
    df = pd.read_csv(data_path)
    df = df.dropna(subset=["group", "spent", "purchases"]).copy()
    df = df[(df["spent"] >= 0) & (df["purchases"] >= 0)].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df["group_label"] = df["group"].map(
        {"GRP A": "GRP A (Automático)", "GRP B": "GRP B (Manual)"}
    ).fillna(df["group"])
    df["device_label"] = df["device"].map(
        {"I": "I (Site)", "A": "A (App)"}
    ).fillna(df["device"])
    df["country_full"] = df["country"].map(COUNTRY_NAMES).fillna(df["country"])
    df["avg_ticket"] = np.where(
        df["purchases"] > 0, df["spent"] / df["purchases"], 0.0
    )
    return df


@st.cache_data
def run_bootstrap(a_spent: np.ndarray, b_spent: np.ndarray, n_boot: int = 2000):
    rng = np.random.default_rng(42)
    boot_diffs = np.empty(n_boot)
    boot_lifts = np.empty(n_boot)
    n_a, n_b = len(a_spent), len(b_spent)
    for i in range(n_boot):
        ma = rng.choice(a_spent, size=n_a, replace=True).mean()
        mb = rng.choice(b_spent, size=n_b, replace=True).mean()
        boot_diffs[i] = ma - mb
        boot_lifts[i] = ((ma - mb) / mb) * 100 if mb != 0 else 0.0
    return boot_diffs, boot_lifts


df = load_data()

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙️ Filtros da Análise")
st.sidebar.caption("Explore o experimento global ou isole mercados e canais específicos.")

all_countries = sorted(df["country"].unique())
selected_countries = st.sidebar.multiselect(
    "Países (country)",
    options=all_countries,
    default=all_countries,
    format_func=lambda c: COUNTRY_NAMES.get(c, c),
)

all_devices = sorted(df["device_label"].unique())
selected_devices = st.sidebar.multiselect(
    "Canal de Compra (device)",
    options=all_devices,
    default=all_devices,
)

all_genders = sorted(df["gender"].unique())
selected_genders = st.sidebar.multiselect(
    "Gênero (gender)",
    options=all_genders,
    default=all_genders,
)

min_date = df["date"].min().date()
max_date = df["date"].max().date()
date_range = st.sidebar.date_input(
    "Período do Experimento",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

alpha = st.sidebar.select_slider(
    "Nível de Significância (α)",
    options=[0.01, 0.05, 0.10],
    value=0.05,
    format_func=lambda x: f"{x:.2f} (Confiança {int((1 - x) * 100)}%)",
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Legenda Oficial:**\n"
    "- `GRP A`: Preenchimento **Automático** (Tratamento)\n"
    "- `GRP B`: Preenchimento **Manual** (Controle)\n"
    "- `I`: Compra pelo **Site**\n"
    "- `A`: Compra pelo **App**"
)

# Aplicação dos filtros
mask = (
    df["country"].isin(selected_countries)
    & df["device_label"].isin(selected_devices)
    & df["gender"].isin(selected_genders)
)
if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start_dt, end_dt = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    mask &= (df["date"] >= start_dt) & (df["date"] <= end_dt)

df_f = df[mask].copy()

grp_a = df_f[df_f["group"] == "GRP A"]
grp_b = df_f[df_f["group"] == "GRP B"]

# ---------------- HEADER ----------------
st.title("💳 Electronic House — Teste A/B de Checkout")
st.markdown(
    "Avaliação estatística da efetividade do novo formulário de cartão de crédito com "
    "**90% de preenchimento automático (`GRP A`)** frente ao **preenchimento manual (`GRP B`)**."
)

if len(grp_a) < 5 or len(grp_b) < 5:
    st.warning("Selecione filtros com pelo menos 5 clientes em cada grupo para executar os testes estatísticos.")
    st.stop()

# Métricas principais do recorte filtrado
a_spent = grp_a["spent"].to_numpy()
b_spent = grp_b["spent"].to_numpy()
mean_a, mean_b = float(a_spent.mean()), float(b_spent.mean())
diff_spent = mean_a - mean_b
lift_spent = (diff_spent / mean_b) * 100 if mean_b != 0 else 0.0
_, p_welch = stats.ttest_ind(a_spent, b_spent, equal_var=False)
_, p_mwu = stats.mannwhitneyu(a_spent, b_spent, alternative="two-sided")

# Banner de Veredito Dinâmico
if p_welch < alpha:
    if diff_spent > 0:
        st.success(
            f"**Veredito no recorte selecionado (α = {alpha:.2f}):** Diferença estatisticamente significativa "
            f"**a favor do Preenchimento Automático (`GRP A`)** — Lift de **{lift_spent:+.2f}%** no gasto médio "
            f"(Welch *p-valor* = `{p_welch:.4f}`, Mann-Whitney *p-valor* = `{p_mwu:.4f}`)."
        )
    else:
        st.error(
            f"**Veredito no recorte selecionado (α = {alpha:.2f}):** Diferença estatisticamente significativa "
            f"**desfavorável ao Preenchimento Automático (`GRP A`)** — Variação de **{lift_spent:+.2f}%** frente ao Manual "
            f"(Welch *p-valor* = `{p_welch:.4f}`, Mann-Whitney *p-valor* = `{p_mwu:.4f}`). Recomenda-se investigar atrito técnico."
        )
else:
    st.info(
        f"**Veredito no recorte selecionado (α = {alpha:.2f}):** **Não há evidência estatística de diferença** entre "
        f"Preenchimento Automático (`GRP A`) e Manual (`GRP B`) — Variação observada de **{lift_spent:+.2f}%** "
        f"(Welch *p-valor* = `{p_welch:.4f}` ≥ `{alpha:.2f}`; Não Rejeita H₀)."
    )

# KPIs
k1, k2, k3, k4 = st.columns(4)
k1.metric(
    "Clientes Avaliados",
    f"{len(df_f):,}",
    f"A: {len(grp_a):,} | B: {len(grp_b):,}",
    delta_color="off",
)
k2.metric(
    "Gasto Médio — Manual (GRP B)",
    f"$ {mean_b:,.2f}",
    f"Mediana: $ {np.median(b_spent):,.0f}",
    delta_color="off",
)
k3.metric(
    "Gasto Médio — Automático (GRP A)",
    f"$ {mean_a:,.2f}",
    f"{lift_spent:+.2f}% vs Manual ({diff_spent:+.2f})",
)
k4.metric(
    "p-valor (Teste t de Welch)",
    f"{p_welch:.4f}",
    "Rejeita H₀ (Significativo)" if p_welch < alpha else "Não Rejeita H₀",
    delta_color="normal" if (p_welch < alpha and diff_spent > 0) else "off",
)

# ---------------- TABS ----------------
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📊 Visão Executiva & Temporal",
        "🧪 Testes de Hipóteses & Bootstrap",
        "🌍 Países, Site vs. App & Gênero",
        "📈 Distribuições & Dados",
    ]
)

# --- ABA 1: VISÃO EXECUTIVA & TEMPORAL ---
with tab1:
    col_l, col_r = st.columns([1, 1.5])

    with col_l:
        ci_b = 1.96 * b_spent.std(ddof=1) / np.sqrt(len(b_spent))
        ci_a = 1.96 * a_spent.std(ddof=1) / np.sqrt(len(a_spent))
        summary_bar = pd.DataFrame(
            {
                "Grupo": ["GRP B (Manual)", "GRP A (Automático)"],
                "Gasto Médio": [mean_b, mean_a],
                "IC_95": [ci_b, ci_a],
            }
        )
        fig_bar = px.bar(
            summary_bar,
            x="Grupo",
            y="Gasto Médio",
            color="Grupo",
            error_y="IC_95",
            text=summary_bar["Gasto Médio"].map(lambda v: f"$ {v:,.2f}"),
            color_discrete_map=COLOR_MAP,
            title="Gasto Médio por Cliente com Intervalo de Confiança (95%)",
        )
        fig_bar.update_traces(textposition="outside")
        fig_bar.update_layout(
            showlegend=False,
            yaxis_range=[0, max(mean_a, mean_b) * 1.18],
            height=420,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_r:
        monthly = (
            df_f.dropna(subset=["date"])
            .groupby([pd.Grouper(key="date", freq="MS"), "group_label"])
            .agg(spent_mean=("spent", "mean"), spent_sum=("spent", "sum"))
            .reset_index()
            .sort_values("date")
        )
        fig_time = px.line(
            monthly,
            x="date",
            y="spent_mean",
            color="group_label",
            markers=True,
            color_discrete_map=COLOR_MAP,
            labels={
                "date": "Mês",
                "spent_mean": "Gasto Médio ($)",
                "group_label": "Grupo",
            },
            title="Evolução Mensal do Gasto Médio por Cliente",
        )
        fig_time.update_layout(height=420, legend_title_text="")
        st.plotly_chart(fig_time, use_container_width=True)

    monthly["cum_rev_millions"] = (
        monthly.groupby("group_label")["spent_sum"].cumsum() / 1e6
    )
    fig_cum = px.line(
        monthly,
        x="date",
        y="cum_rev_millions",
        color="group_label",
        color_discrete_map=COLOR_MAP,
        labels={
            "date": "Mês",
            "cum_rev_millions": "Receita Acumulada ($ Milhões)",
            "group_label": "Grupo",
        },
        title="Receita Acumulada ao Longo do Experimento (em Milhões)",
    )
    fig_cum.update_layout(height=360, legend_title_text="")
    st.plotly_chart(fig_cum, use_container_width=True)

# --- ABA 2: TESTES DE HIPÓTESES & BOOTSTRAP ---
with tab2:
    st.subheader("1. Validação do Desenho Experimental (SRM e Poder Amostral)")
    n_total = len(grp_a) + len(grp_b)
    srm_stat, srm_pval = stats.chisquare(
        f_obs=[len(grp_a), len(grp_b)], f_exp=[n_total / 2, n_total / 2]
    )
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(0.80)
    pooled_std = df_f["spent"].std(ddof=1)
    n_min_2pct = int(
        np.ceil(2 * (((z_alpha + z_beta) * pooled_std) / (mean_b * 0.02)) ** 2)
    )

    c_s1, c_s2, c_s3 = st.columns(3)
    c_s1.metric(
        "Proporção GRP A / GRP B",
        f"{len(grp_a)/n_total*100:.2f}% / {len(grp_b)/n_total*100:.2f}%",
        f"SRM p-valor: {srm_pval:.4f}",
        delta_color="off",
    )
    c_s2.metric(
        "Amostra Mínima p/ Lift de 2% (Poder 80%)",
        f"{n_min_2pct:,} por grupo",
        "Atendido" if min(len(grp_a), len(grp_b)) >= n_min_2pct else "Amostra parcial no filtro",
        delta_color="normal" if min(len(grp_a), len(grp_b)) >= n_min_2pct else "off",
    )
    _, p_levene = stats.levene(a_spent, b_spent)
    c_s3.metric(
        "Homocedasticidade (Levene)",
        f"p = {p_levene:.4f}",
        "Variâncias Homogêneas" if p_levene >= alpha else "Variâncias Desiguais",
        delta_color="off",
    )

    st.subheader("2. Testes Paramétricos e Não-Paramétricos")
    hyp_rows = []
    for col_m, lbl_m in [
        ("spent", "Gasto por Cliente (spent)"),
        ("purchases", "Compras por Cliente (purchases)"),
        ("avg_ticket", "Ticket Médio por Compra (avg_ticket)"),
    ]:
        va = grp_a[col_m].to_numpy()
        vb = grp_b[col_m].to_numpy()
        ma, mb = va.mean(), vb.mean()
        d = ma - mb
        lft = (d / mb) * 100 if mb != 0 else 0.0
        _, pw = stats.ttest_ind(va, vb, equal_var=False)
        _, ps = stats.ttest_ind(va, vb, equal_var=True)
        _, pm = stats.mannwhitneyu(va, vb, alternative="two-sided")
        hyp_rows.append(
            {
                "Métrica": lbl_m,
                "Média Manual (GRP B)": round(mb, 4),
                "Média Automático (GRP A)": round(ma, 4),
                "Diferença (A − B)": round(d, 4),
                "Lift (%)": round(lft, 3),
                "p-valor (Welch t)": round(pw, 6),
                "p-valor (Student t)": round(ps, 6),
                "p-valor (Mann-Whitney U)": round(pm, 6),
                f"Decisão (α={alpha:.2f})": (
                    "Rejeita H₀ (Significativo)"
                    if pw < alpha
                    else "Não Rejeita H₀ (Sem Diferença)"
                ),
            }
        )
    st.dataframe(pd.DataFrame(hyp_rows), use_container_width=True, hide_index=True)

    st.subheader("3. Intervalo de Confiança via Bootstrapping (2.000 reamostragens)")
    boot_diffs, boot_lifts = run_bootstrap(a_spent, b_spent, n_boot=2000)
    low_pct = (alpha / 2) * 100
    high_pct = (1 - alpha / 2) * 100
    ci_low, ci_high = np.percentile(boot_diffs, [low_pct, high_pct])
    lift_low, lift_high = np.percentile(boot_lifts, [low_pct, high_pct])

    st.caption(
        f"Diferença observada (`GRP A − GRP B`): **{diff_spent:+.2f}** | "
        f"IC {int((1-alpha)*100)}% Bootstrap: **[{ci_low:+.2f}, {ci_high:+.2f}]** "
        f"(Lift: **[{lift_low:+.2f}%, {lift_high:+.2f}%]**)."
    )

    fig_boot = go.Figure()
    fig_boot.add_trace(
        go.Histogram(
            x=boot_diffs,
            nbinsx=45,
            marker_color="#3D5A80",
            opacity=0.82,
            name="Diferença Bootstrap",
        )
    )
    fig_boot.add_vline(
        x=0,
        line_dash="dash",
        line_color="#D62828",
        line_width=2.5,
        annotation_text="Efeito Nulo (0)",
        annotation_position="top right",
    )
    fig_boot.add_vline(
        x=ci_low,
        line_dash="dot",
        line_color="#2A9D8F",
        line_width=2,
        annotation_text=f"IC Inf: {ci_low:+.1f}",
    )
    fig_boot.add_vline(
        x=ci_high,
        line_dash="dot",
        line_color="#2A9D8F",
        line_width=2,
        annotation_text=f"IC Sup: {ci_high:+.1f}",
    )
    fig_boot.update_layout(
        title=f"Distribuição Bootstrap da Diferença de Gasto Médio (GRP A − GRP B) — IC {int((1-alpha)*100)}%",
        xaxis_title="Diferença de Gasto Médio ($)",
        yaxis_title="Frequência",
        height=400,
    )
    st.plotly_chart(fig_boot, use_container_width=True)

# --- ABA 3: PAÍSES, SITE VS APP E GÊNERO ---
with tab3:
    countries_in_slice = sorted(df_f["country"].unique())
    bonf_alpha = alpha / max(len(countries_in_slice), 1)
    c_rows = []
    for c in countries_in_slice:
        sa = df_f[(df_f["country"] == c) & (df_f["group"] == "GRP A")]["spent"]
        sb = df_f[(df_f["country"] == c) & (df_f["group"] == "GRP B")]["spent"]
        if len(sa) > 1 and len(sb) > 1:
            ma, mb = sa.mean(), sb.mean()
            lft = ((ma - mb) / mb) * 100 if mb != 0 else 0.0
            _, pt = stats.ttest_ind(sa, sb, equal_var=False)
            _, pm = stats.mannwhitneyu(sa, sb, alternative="two-sided")
            c_rows.append(
                {
                    "País": COUNTRY_NAMES.get(c, c),
                    "Código": c,
                    "N Manual (B)": len(sb),
                    "N Automático (A)": len(sa),
                    "Média Manual (B)": round(mb, 2),
                    "Média Automático (A)": round(ma, 2),
                    "Diferença (A − B)": round(ma - mb, 2),
                    "Lift Automático (%)": round(lft, 2),
                    "p-valor (Welch t)": round(pt, 4),
                    "p-valor (Mann-Whitney)": round(pm, 4),
                    f"Sig. (p < {alpha:.2f})": "Sim" if pt < alpha else "Não",
                    f"Sig. Bonferroni (p < {bonf_alpha:.4f})": "Sim" if pt < bonf_alpha else "Não",
                }
            )
    country_df = pd.DataFrame(c_rows).sort_values("Lift Automático (%)", ascending=True)

    if not country_df.empty:
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            sig_col = f"Sig. (p < {alpha:.2f})"
            fig_lift = px.bar(
                country_df,
                x="Lift Automático (%)",
                y="País",
                orientation="h",
                color=sig_col,
                color_discrete_map={"Sim": "#E07A5F", "Não": "#8D99AE"},
                text=country_df["Lift Automático (%)"].map(lambda v: f"{v:+.2f}%"),
                title=f"Lift (%) do Preenchimento Automático por País (α = {alpha:.2f})",
            )
            fig_lift.add_vline(x=0, line_dash="dash", line_color="black")
            fig_lift.update_layout(height=450)
            st.plotly_chart(fig_lift, use_container_width=True)

        with col_c2:
            melted_c = country_df.melt(
                id_vars=["País"],
                value_vars=["Média Manual (B)", "Média Automático (A)"],
                var_name="Grupo",
                value_name="Gasto Médio",
            )
            melted_c["Grupo"] = melted_c["Grupo"].replace(
                {
                    "Média Manual (B)": "GRP B (Manual)",
                    "Média Automático (A)": "GRP A (Automático)",
                }
            )
            fig_cbar = px.bar(
                melted_c,
                x="País",
                y="Gasto Médio",
                color="Grupo",
                barmode="group",
                color_discrete_map=COLOR_MAP,
                title="Gasto Médio por País e Método de Pagamento",
            )
            fig_cbar.update_layout(height=450, xaxis_tickangle=-25, legend_title_text="")
            st.plotly_chart(fig_cbar, use_container_width=True)

        st.markdown(
            f"**Tabela Estatística por País** *(Limiar ajustado de Bonferroni para {len(country_df)} países: `α = {bonf_alpha:.4f}`)*"
        )
        st.dataframe(
            country_df.sort_values("Lift Automático (%)", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Desempenho por Canal (Site vs. App) e Gênero")
    seg_rows = []
    for seg_col, dim_lbl in [("device_label", "Canal (Device)"), ("gender", "Gênero")]:
        for val in sorted(df_f[seg_col].unique()):
            sa = df_f[(df_f[seg_col] == val) & (df_f["group"] == "GRP A")]["spent"]
            sb = df_f[(df_f[seg_col] == val) & (df_f["group"] == "GRP B")]["spent"]
            if len(sa) > 1 and len(sb) > 1:
                ma, mb = sa.mean(), sb.mean()
                _, pt = stats.ttest_ind(sa, sb, equal_var=False)
                _, pm = stats.mannwhitneyu(sa, sb, alternative="two-sided")
                seg_rows.append(
                    {
                        "Dimensão": dim_lbl,
                        "Segmento": val,
                        "N Manual (B)": len(sb),
                        "N Automático (A)": len(sa),
                        "Média Manual (B)": round(mb, 2),
                        "Média Automático (A)": round(ma, 2),
                        "Lift Automático (%)": round(((ma - mb) / mb) * 100, 2),
                        "p-valor (Welch t)": round(pt, 4),
                        "p-valor (Mann-Whitney)": round(pm, 4),
                        f"Sig. (p < {alpha:.2f})": "Sim" if pt < alpha else "Não",
                    }
                )
    if seg_rows:
        st.dataframe(pd.DataFrame(seg_rows), use_container_width=True, hide_index=True)

# --- ABA 4: DISTRIBUIÇÕES & DADOS ---
with tab4:
    d_col1, d_col2 = st.columns(2)
    with d_col1:
        fig_hist = px.histogram(
            df_f,
            x="spent",
            color="group_label",
            barmode="overlay",
            nbins=40,
            opacity=0.6,
            color_discrete_map=COLOR_MAP,
            labels={"spent": "Valor Gasto ($)", "group_label": "Grupo"},
            title="Distribuição de Valor Gasto por Cliente (spent)",
        )
        fig_hist.update_layout(height=420, legend_title_text="")
        st.plotly_chart(fig_hist, use_container_width=True)

    with d_col2:
        fig_box = px.box(
            df_f,
            x="group_label",
            y="spent",
            color="group_label",
            color_discrete_map=COLOR_MAP,
            labels={"spent": "Valor Gasto ($)", "group_label": "Grupo"},
            title="Boxplot e Outliers de Gasto por Grupo",
        )
        fig_box.update_layout(height=420, showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)

    st.subheader("Amostra dos Registros Filtrados")
    st.dataframe(df_f.head(200), use_container_width=True, hide_index=True)

