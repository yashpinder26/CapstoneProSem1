# app.py — Interactive OOP Dashboard (visuals polished)
import os
import re
from typing import List, Optional

import pandas as pd
import plotly.express as px
import streamlit as st

# ---------- CONFIG ----------
st.set_page_config(page_title="Out-of-Pocket Costs Dashboard", layout="wide")

DATA_DIR = "."

FILE_TABLE8 = "Table8_GP_Out_of_Pocket_Clean_YearFixed (1).csv"
FILE_TABLE9 = "Table9_Remoteness_Out_of_Pocket_Clean (1) (1).csv"
FILE_STATES = "Out_of_pocket_costs_by_states&territories_2003_2023.csv"

# ---------- helpers ----------
def find_one(patterns: List[str], columns: List[str], required=True, default: Optional[str]=None) -> Optional[str]:
    for pat in patterns:
        rx = re.compile(pat, re.I)
        for c in columns:
            if rx.search(str(c)):
                return c
    if required:
        raise KeyError(f"Could not find any column matching: {patterns} in {list(columns)}")
    return default

def yearify(series: pd.Series) -> pd.Series:
    return (
        pd.to_numeric(
            series.astype(str).str.extract(r"(\d{4})", expand=False),
            errors="coerce",
        ).astype("Int64")
    )

@st.cache_data(show_spinner=False)
def load_csv(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df

def value_col_choice(df: pd.DataFrame, basis: str) -> str:
    cols = list(df.columns)
    if basis == "Actual":
        return find_one(
            [r"actual.*(cost|price)", r"\bactual\b", r"oop.*actual", r"value|amount|price"],
            cols, required=True
        )
    else:  # Inflation adjusted
        return find_one(
            [r"inflation.*adjust", r"adjust(ed|ment)", r"\bcpi\b", r"inflation.*price"],
            cols, required=True
        )

def seifa_standardize_label(s: str) -> str:
    t = str(s).strip().lower()
    mapping = {"quintile 1": "Q1", "quintile 2": "Q2", "quintile 3": "Q3", "quintile 4": "Q4", "quintile 5": "Q5"}
    for k, v in mapping.items():
        if k in t:
            return v
    m = re.search(r"\b([1-5])\b", t)
    return f"Q{m.group(1)}" if m else s

def area_standardize_label(s: str) -> str:
    t = str(s).strip().lower()
    t = t.replace("majorcities", "major cities")
    t = t.replace("innerregional", "inner regional")
    t = t.replace("outerregional", "outer regional")
    t = t.replace("veryremote", "very remote")
    return t.title()

# --------- visualization helpers (polished & consistent) ----------
def _is_dark():
    try:
        return st.get_option("theme.base") == "dark"
    except Exception:
        return True

def _template():
    return "plotly_dark" if _is_dark() else "plotly_white"

# Consistent categorical palettes (fixed order)
COLOR_SEIFA = {
    "Q1": "#5DA5DA",  # blue
    "Q2": "#60BD68",  # green
    "Q3": "#F17CB0",  # pink
    "Q4": "#B2912F",  # brown/gold
    "Q5": "#F15854",  # red
}
COLOR_AREA = {
    "Major Cities":  "#5DA5DA",
    "Inner Regional":"#60BD68",
    "Outer Regional":"#B276B2",
    "Remote":        "#FAA43A",
    "Very Remote":   "#F15854",
}

def currency_axis():
    # $ prefix + sensible ticks; clamp to zero for easier comparison
    return dict(title="", tickprefix="$", tickformat=",.2f", rangemode="tozero")

def style_time_series(fig, title, subtitle=None):
    fig.update_traces(mode="lines+markers",
                      marker=dict(size=6, line=dict(width=0)),
                      line=dict(width=2.2))
    fig.update_layout(
        template=_template(),
        title=title,
        title_x=0.02,
        margin=dict(l=40, r=40, t=60, b=40),
        hovermode="x unified",
        xaxis_title="Year",
        yaxis=currency_axis(),
        xaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.15)"),
        yaxis_showgrid=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="left", x=0.0, title=None),
    )
    # optional subtitle
    if subtitle:
        fig.add_annotation(
            x=0, y=1.08, xref="paper", yref="paper",
            showarrow=False, align="left",
            text=f"<span style='font-size:0.9em; opacity:0.8;'>{subtitle}</span>"
        )
    return fig

def style_bar(fig, title, subtitle=None):
    fig.update_traces(marker_line_width=0, opacity=0.95)
    fig.update_layout(
        template=_template(),
        title=title,
        title_x=0.02,
        margin=dict(l=40, r=40, t=60, b=40),
        xaxis_title="Cost ($)",
        yaxis_title="",
        xaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.15)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="left", x=0.0, title=None),
    )
    if subtitle:
        fig.add_annotation(
            x=0, y=1.08, xref="paper", yref="paper",
            showarrow=False, align="left",
            text=f"<span style='font-size:0.9em; opacity:0.8;'>{subtitle}</span>"
        )
    fig.update_traces(hovertemplate="<b>%{y}</b><br>$%{x:,.2f} per GP service<extra></extra>")
    return fig

def style_heatmap(fig, title, subtitle=None):
    fig.update_layout(
        template=_template(),
        title=title,
        title_x=0.02,
        margin=dict(l=40, r=40, t=60, b=40),
        coloraxis_colorbar=dict(title="Cost ($)"),
    )
    if subtitle:
        fig.add_annotation(
            x=0, y=1.08, xref="paper", yref="paper",
            showarrow=False, align="left",
            text=f"<span style='font-size:0.9em; opacity:0.8;'>{subtitle}</span>"
        )
    return fig

# ---------- micro-helpers for plain-English explanations ----------
def seifa_explainer():
    st.markdown(
        """
**What is SEIFA?**  
SEIFA = *Socio-Economic Indexes for Areas* (ABS). It ranks small areas in Australia by relative advantage/disadvantage.

**How to read the quintiles:**
- **Q1 – Most disadvantaged**: more unemployment, lower median incomes, more rental stress.  
  *Think:* outer-suburban fringes or smaller towns with fewer local services.
- **Q2 – Below average**
- **Q3 – Middle**
- **Q4 – Above average**
- **Q5 – Least disadvantaged**: higher incomes, more tertiary education, better access to services.  
  *Think:* inner-city / well-resourced suburbs.

**Why we show it:** to see if people in more disadvantaged areas pay more out-of-pocket (OOP) than people in advantaged areas.
"""
    )


def remoteness_explainer():
    st.markdown(
        """
**What are remoteness areas?**  
They come from the ABS ASGS classification and describe how far a place is from services based on road distance to population centres.

**Categories (from most to least accessible):**
- **Major Cities** – metropolitan areas with dense services and many GPs.
- **Inner Regional** – large regional centres (generally <2–3 hours from a capital).
- **Outer Regional** – smaller regional towns with fewer specialists.
- **Remote** – long travel to larger centres; limited providers locally.
- **Very Remote** – very long travel distances; very limited local services.

**Why we show it:** to see how distance/access to services links to OOP costs.
"""
    )

# helpers to keep category order consistent everywhere
SEIFA_ORDER = ["Q1", "Q2", "Q3", "Q4", "Q5"]
AREA_ORDER  = ["Major Cities", "Inner Regional", "Outer Regional", "Remote", "Very Remote"]

def order_seifa(df, col="SEIFA"):
    if col in df.columns:
        df[col] = pd.Categorical(df[col], categories=SEIFA_ORDER, ordered=True)
    return df

def order_area(df, col="Area"):
    if col in df.columns:
        df[col] = pd.Categorical(df[col], categories=AREA_ORDER, ordered=True)
    return df

# ---------- load data ----------
def safe_path(name): return os.path.join(DATA_DIR, name)

table8 = load_csv(safe_path(FILE_TABLE8))
table9 = load_csv(safe_path(FILE_TABLE9))
states = load_csv(safe_path(FILE_STATES))

# ---------- normalize schemas ----------
def prep_table8(df: pd.DataFrame) -> pd.DataFrame:
    y = find_one([r"^year$", r"service[_\s]*year", r"\bdate\b"], df.columns)
    s = find_one([r"^state$", r"jurisdiction"], df.columns)
    q = find_one([r"seifa.*quintile", r"\bquintile\b"], df.columns)
    df = df.copy()
    df["Year"] = yearify(df[y])
    df["State"] = df[s].astype(str)
    df["SEIFA"] = df[q].map(seifa_standardize_label)
    return df

def prep_table9(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    y = find_one([r"^year$", r"service[_\s]*year"], df.columns, required=False)
    if not y and "service_year" in df.columns: y = "service_year"
    s = find_one([r"^state$", r"jurisdiction"], df.columns, required=False) or "state"
    a = find_one([r"remoteness|aria|ra\s*category|area"], df.columns)
    df["Year"] = yearify(df[y]) if y else pd.NA
    df["State"] = df[s].astype(str)
    df["Area"] = df[a].map(area_standardize_label)
    return df

def prep_states(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    y = find_one([r"^year$"], df.columns)
    r = find_one([r"^region$|^state$|jurisdiction"], df.columns)
    df["Year"] = yearify(df[y])
    actual_cols = [c for c in df.columns if re.fullmatch(r"actual[1-5]", c.lower())]
    if not actual_cols:
        actual_cols = [find_one([r"actual.*(price|cost)|\bactual\b"], df.columns)]
    df["_actual_mean"] = df[actual_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    return df.rename(columns={r: "Region"})

t8 = prep_table8(table8)
t9 = prep_table9(table9)
st_wide = prep_states(states)

# ---------- sidebar (global controls) ----------
st.sidebar.title("Filters")
year_min = int(pd.concat([t8["Year"], t9["Year"], st_wide["Year"]], ignore_index=True).dropna().min())
year_max = int(pd.concat([t8["Year"], t9["Year"], st_wide["Year"]], ignore_index=True).dropna().max())
year_range = st.sidebar.slider("Year range", min_value=year_min, max_value=year_max,
                               value=(max(year_min, 2003), year_max), step=1)

# 👇 changed wording: "Adjusted" -> "Inflation adjusted"
basis = st.sidebar.radio(
    "Price basis",
    ["Actual", "Inflation adjusted"],
    index=0,
    horizontal=True,
    help="Actual = prices in the year they were paid. Inflation adjusted = constant dollars for cross-year comparison."
)

state_options = sorted(set(t8["State"].dropna().unique()).union(set(t9["State"].dropna().unique())))
state_pick = st.sidebar.multiselect("State(s)/Territories", options=state_options, default=[])

# ---------- page routing ----------
page = st.sidebar.radio("Page", ["Overview", "SEIFA equity", "Remoteness", "States & Territories"])

# ---------- derive filtered frames ----------
def filter_by_year(df):
    if "Year" not in df.columns: return df.copy()
    return df[(df["Year"].notna()) & (df["Year"].between(year_range[0], year_range[1]))].copy()

def apply_state_filter(df):
    if not state_pick or "State" not in df.columns: return df
    return df[df["State"].isin(state_pick)] if "State" in df.columns else df

# ---------- Overview ----------
if page == "Overview":
    st.title("Overview")
    st.markdown(
        """
**What you’re seeing:**  
• National trend of average out-of-pocket (OOP) cost per GP service.  
• Latest year, year-over-year change, and equity gap (SEIFA Q5 − Q1).  
• Latest-year comparison by state/territory.

**How to use:**  
• Adjust the year range and price basis in the sidebar.  
• “Inflation adjusted” converts all years to constant dollars for fair comparison.
        """
    )

    col_val = value_col_choice(t8, basis)
    base = filter_by_year(t8)
    aus = base[base["State"].str.fullmatch("Aus", case=False, na=False)]
    if aus.empty:
        aus = base.groupby("Year", as_index=False)[col_val].mean()

    if not aus.empty:
        latest_row = aus.sort_values("Year").tail(1)
        latest_year = int(latest_row["Year"].iloc[0])
        latest_val = float(latest_row[col_val].iloc[0])
        prev = aus[aus["Year"] == latest_year-1]
        yoy = (latest_val - float(prev[col_val].iloc[0]))/float(prev[col_val].iloc[0])*100 if not prev.empty else 0.0

        t8_gap = filter_by_year(t8)
        t8_gap = t8_gap[t8_gap["State"].str.fullmatch("Aus", case=False, na=False)]
        try:
            pvt = t8_gap.pivot_table(index="Year", columns="SEIFA", values=col_val, aggfunc="mean")
            gap_latest = float(pvt.loc[latest_year, "Q5"] - pvt.loc[latest_year, "Q1"])
        except Exception:
            gap_latest = float("nan")

        c1, c2, c3 = st.columns(3)
        c1.metric(f"Latest OOP ({basis})", f"${latest_val:,.2f}", f"{yoy:+.1f}% vs {latest_year-1}")
        c2.metric("Latest year", f"{latest_year}")
        c3.metric("Equity gap (Q5 − Q1)", f"${gap_latest:,.2f}" if pd.notna(gap_latest) else "N/A")

        # --- National time series (polished) ---
        series = aus.sort_values("Year").reset_index(drop=True)
        fig = px.line(series, x="Year", y=col_val, markers=True)
        fig.update_traces(
            hovertemplate="<b>%{x}</b><br>$%{y:.2f} per GP service<br>(Price basis: " + basis + ")<extra></extra>"
        )
        fig = style_time_series(fig, f"Australia — OOP per service ({basis})")

        # highlight latest year
        fig.add_scatter(
            x=[int(series['Year'].iloc[-1])],
            y=[float(series[col_val].iloc[-1])],
            mode="markers",
            marker=dict(size=11, line=dict(width=1), symbol="circle-open"),
            name="Latest year",
            hovertemplate="<b>%{x}</b><br>$%{y:.2f} per GP service (latest)<extra></extra>",
        )

        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Each dot shows the average out-of-pocket (OOP) cost per GP service for that year on the selected price basis."
        )

    # --- Latest year bar by state/territory ---
    st.subheader("Latest-year OOP by state/territory")
    sw = filter_by_year(st_wide)
    if not sw.empty:
        ly = int(sw["Year"].max())
        latest = sw[sw["Year"] == ly].groupby("Region", as_index=False)["_actual_mean"].mean().sort_values("_actual_mean")
        fig2 = px.bar(latest, x="_actual_mean", y="Region", orientation="h",
                      labels={"_actual_mean":"Cost ($)", "Region":"State/Territory"})
        fig2 = style_bar(fig2, f"OOP by state/territory — {ly} (Actual)")
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("Bars show the mean OOP per GP service in each state/territory for the latest year.")
    else:
        st.info("No state data in selected year range.")

# ---------- SEIFA ----------
elif page == "SEIFA equity":
    st.title("SEIFA equity")
    seifa_explainer()
    st.markdown(
        """
**What you’re seeing:**  
• Out-of-pocket (OOP) cost per GP service by **SEIFA quintile** (Q1 = most disadvantaged, Q5 = least).  
• Optional state filter from the sidebar.  
• Equity gap (Q5 − Q1) tracked over time.

**What is SEIFA?**  
SEIFA stands for **Socio-Economic Indexes for Areas**, created by the ABS (Australian Bureau of Statistics).  
It ranks areas in Australia by levels of relative disadvantage:  
- **Q1:** Most disadvantaged areas (lower income, higher unemployment, fewer resources)  
- **Q5:** Least disadvantaged areas (higher income, better resources and opportunities)

This lets us see whether people in more disadvantaged areas are paying more out-of-pocket compared to those in wealthier areas.
        """
    )

    df = order_seifa(apply_state_filter(filter_by_year(t8)))
    col_val = value_col_choice(t8, basis)
    fig = px.line(
        df.sort_values(["SEIFA","Year"]),
        x="Year", y=col_val, color="SEIFA", line_group="SEIFA",
        color_discrete_map=COLOR_SEIFA,
        category_orders={"SEIFA": SEIFA_ORDER},
)


    if df.empty:
        st.info("No rows for current filters.")
    else:
        fig = px.line(df.sort_values(["SEIFA","Year"]), x="Year", y=col_val, color="SEIFA", line_group="SEIFA")
        fig.update_traces(hovertemplate="<b>%{x}</b><br>$%{y:.2f} per GP service<br>SEIFA: %{legendgroup}<extra></extra>")
        fig = style_time_series(fig, f"OOP by SEIFA quintile ({basis})")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Each dot is the yearly average OOP per GP service for that SEIFA quintile.")

        # Gap (Q5 - Q1)
        pvt = df.pivot_table(index="Year", columns="SEIFA", values=col_val, aggfunc="mean")
        if set(["Q1","Q5"]).issubset(pvt.columns):
            pvt["Gap_Q5_minus_Q1"] = pvt["Q5"] - pvt["Q1"]
            fig2 = px.line(pvt.reset_index(), x="Year", y="Gap_Q5_minus_Q1")
            fig2.update_traces(hovertemplate="<b>%{x}</b><br>Gap: $%{y:.2f} (Q5 − Q1)<extra></extra>")
            fig2 = style_time_series(fig2, f"Gap in OOP (Q5 − Q1), {basis}")
            st.subheader("Equity gap (Q5 − Q1) over time")
            st.plotly_chart(fig2, use_container_width=True)
            st.caption("Shows the difference in OOP between Q5 (least disadvantage) and Q1 (most disadvantage) each year.")
        else:
            st.info("Need both Q1 and Q5 to compute gap.")


# ---------- Remoteness ----------
elif page == "Remoteness":
    st.title("Remoteness")
    remoteness_explainer()
    st.markdown(
        """
**What you’re seeing:**  
• OOP per service by remoteness area (Major Cities → Very Remote) over time.  
• Latest-year comparison of remoteness areas.
        """
    )

    df = apply_state_filter(filter_by_year(t9))
    try:
        col_val = value_col_choice(t9, basis)
    except KeyError:
        col_val = "oop_actual" if basis == "Actual" else "oop_inflation_adjusted"
        if col_val not in t9.columns:
            st.error("Could not find an appropriate value column for Table 9.")
            st.stop()

    if df.empty:
        st.info("No rows for current filters.")
    else:
        fig = px.line(df.sort_values(["Area","Year"]), x="Year", y=col_val, color="Area")
        fig.update_traces(hovertemplate="<b>%{x}</b><br>$%{y:.2f} per GP service<br>Area: %{legendgroup}<extra></extra>")
        fig = style_time_series(fig, f"OOP by remoteness area ({basis})")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Each dot is the yearly average OOP per GP service for that remoteness area.")

        ly = int(df["Year"].max())
        latest = df[df["Year"] == ly].groupby("Area", as_index=False)[col_val].mean().sort_values(col_val)
        st.subheader(f"Latest-year by remoteness — {ly}")
        fig2 = px.bar(latest, x=col_val, y="Area", orientation="h",
                      labels={col_val:"Cost ($)", "Area":"Remoteness"})
        fig2 = style_bar(fig2, "Latest-year remoteness")
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("Bars show the mean OOP per GP service for each remoteness area in the latest year.")

# ---------- States ----------
elif page == "States & Territories":
    st.title("States & Territories")
    st.markdown(
        """
**What you’re seeing:**  
• Heatmap of OOP (Actual) by state/territory across years (darker = higher).  
• Optional line comparison for selected states.
        """
    )
    df = filter_by_year(st_wide)
    if df.empty:
        st.info("No state data in current range.")
    else:
        pvt = df.pivot_table(index="Region", columns="Year", values="_actual_mean", aggfunc="mean")
        st.subheader("Heatmap (Actual)")
        fig = px.imshow(pvt, aspect="auto", color_continuous_scale="YlOrRd",
                        labels=dict(color="Cost ($)"))
        fig = style_heatmap(fig, "OOP (mean of quintiles) by State/Territory × Year")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Cells show the mean OOP per GP service for each state × year (Actual prices).")

        opts = sorted(pvt.index.tolist())
        pick = st.multiselect("Compare states/territories", options=opts, default=opts[:3])
        if pick:
            dfc = df[df["Region"].isin(pick)]
            fig2 = px.line(dfc.sort_values(["Region","Year"]), x="Year", y="_actual_mean", color="Region")
            fig2.update_traces(hovertemplate="<b>%{x}</b><br>$%{y:.2f} per GP service<br>Region: %{legendgroup}<extra></extra>")
            fig2 = style_time_series(fig2, "Comparison — OOP (Actual)")
            st.plotly_chart(fig2, use_container_width=True)
            st.caption("Each dot is the yearly mean OOP per GP service for the selected state/territory.")

# ---------- footer ----------
st.caption("Notes: “Actual” = prices in the year paid. “Inflation adjusted” = constant dollars to compare across years. "
           "Data sources: cleaned CSVs from AIHW MBS bulk-billing summary (Table 8 & 9), and state-wide file.")


