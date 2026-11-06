"""Intelligent System to Visualize Pollutant Trends Across Monitoring Stations
Run:  streamlit run app.py
"""
import os
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from data_generator import generate
from warehouse import DB, build_warehouse, connect, dims, get_hourly

st.set_page_config(page_title="Pollutant Trend Dashboard", page_icon="🌫️", layout="wide")

# ---------- Data warehouse bootstrap ----------
@st.cache_resource
def get_con():
    if not os.path.exists(DB):
        return build_warehouse(generate())
    return connect()

with st.sidebar:
    st.title("🌫️ Controls")
    up = st.file_uploader("Upload your own CSV (optional)", type="csv",
                          help="Columns: timestamp, station_name, city, lat, lon, pollutant, value")
    if up is not None and st.button("Load into warehouse"):
        build_warehouse(pd.read_csv(up)); st.cache_resource.clear(); st.cache_data.clear(); st.rerun()

con = get_con()
dim_st, dim_po = dims(con)
dmin, dmax = pd.read_sql("SELECT MIN(date) a, MAX(date) b FROM fact_readings", con).iloc[0]

with st.sidebar:
    cities = st.multiselect("City", sorted(dim_st.city.unique()), default=sorted(dim_st.city.unique()))
    stn_opts = dim_st[dim_st.city.isin(cities)].station_name.tolist()
    stations = st.multiselect("Monitoring stations", stn_opts, default=stn_opts)
    pollutants = st.multiselect("Pollutants", dim_po.pollutant.tolist(), default=["PM2.5", "PM10"])
    main_pol = st.selectbox("Primary pollutant", pollutants or dim_po.pollutant.tolist())
    dr = st.date_input("Date range", (pd.to_datetime(dmax) - pd.Timedelta(days=180), pd.to_datetime(dmax)),
                       min_value=pd.to_datetime(dmin), max_value=pd.to_datetime(dmax))
    freq = st.radio("Aggregation", ["Hourly", "Daily", "Weekly", "Monthly"], index=1, horizontal=True)
    roll = st.slider("Smoothing window (periods)", 1, 30, 7)

if not stations or not pollutants or len(dr) != 2:
    st.info("Select at least one station, one pollutant and a full date range."); st.stop()

@st.cache_data(show_spinner="Querying warehouse…")
def load(stations, pollutants, s, e):
    return get_hourly(con, list(stations), list(pollutants), s, e)

df = load(tuple(stations), tuple(pollutants), dr[0], dr[1])
if df.empty:
    st.warning("No data for the selected filters."); st.stop()

RULE = {"Hourly": "h", "Daily": "D", "Weekly": "W", "Monthly": "MS"}[freq]
def agg(d):
    return (d.set_index("timestamp").groupby(["station_name", "pollutant"])["value"]
            .resample(RULE).mean().reset_index())

d_main = df[df.pollutant == main_pol]
unit = d_main.unit.iloc[0]; limit = d_main.who_limit.iloc[0]
ts = agg(d_main)
ts["smooth"] = ts.groupby("station_name")["value"].transform(lambda x: x.rolling(roll, min_periods=1).mean())

# ---------- Header + KPIs ----------
st.title("An Intelligent System to Visualize Pollutant Trends Across Monitoring Stations")
st.caption("Big Data Analytics • Data Warehousing • Environmental Monitoring")
by_stn = d_main.groupby("station_name")["value"].mean().sort_values()
exceed = (d_main.groupby(["station_name", "date"])["value"].mean() > limit).mean() * 100 if pd.notna(limit) else np.nan
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric(f"Avg {main_pol}", f"{d_main.value.mean():.1f} {unit}")
c2.metric("Peak (hourly)", f"{d_main.value.max():.1f}")
c3.metric("Cleanest station", by_stn.index[0], f"{by_stn.iloc[0]:.1f}", delta_color="off")
c4.metric("Most polluted", by_stn.index[-1], f"{by_stn.iloc[-1]:.1f}", delta_color="off")
c5.metric("Days above WHO limit", f"{exceed:.0f}%" if pd.notna(exceed) else "n/a")
st.caption(f"{len(df):,} readings loaded from the warehouse")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["📈 Trends", "🏭 Station comparison", "🔥 Heatmaps", "🗺️ Map", "🧠 Insights", "📋 Data"])

# ---------- Trends ----------
with tab1:
    fig = px.line(ts, x="timestamp", y="smooth", color="station_name",
                  labels={"smooth": f"{main_pol} ({unit})", "timestamp": ""}, title=f"{main_pol} trend by station ({freq.lower()}, {roll}-period smoothing)")
    if pd.notna(limit):
        fig.add_hline(y=limit, line_dash="dash", line_color="red", annotation_text=f"WHO guideline {limit}")
    st.plotly_chart(fig, use_container_width=True)
    if len(pollutants) > 1:
        multi = agg(df.groupby(["timestamp", "pollutant"], as_index=False)["value"].mean().assign(station_name="All"))
        multi["norm"] = multi.groupby("pollutant")["value"].transform(lambda x: x / x.max() * 100)
        st.plotly_chart(px.line(multi, x="timestamp", y="norm", color="pollutant",
                                title="Pollutants compared (normalised to each pollutant's max = 100)"),
                        use_container_width=True)

# ---------- Comparison ----------
with tab2:
    a, b = st.columns(2)
    a.plotly_chart(px.bar(by_stn.reset_index(), x="value", y="station_name", orientation="h", color="value",
                          color_continuous_scale="YlOrRd", title=f"Mean {main_pol} by station",
                          labels={"value": unit, "station_name": ""}), use_container_width=True)
    b.plotly_chart(px.box(d_main, x="station_name", y="value", color="station_name",
                          title="Distribution & outliers", labels={"value": unit, "station_name": ""}).update_layout(showlegend=False),
                   use_container_width=True)
    mon = d_main.groupby(["month", "station_name"])["value"].mean().reset_index()
    st.plotly_chart(px.line(mon, x="month", y="value", color="station_name", markers=True,
                            title="Seasonal pattern (average by calendar month)"), use_container_width=True)

# ---------- Heatmaps ----------
with tab3:
    a, b = st.columns(2)
    p1 = d_main.pivot_table(index="station_name", columns="month", values="value", aggfunc="mean")
    a.plotly_chart(px.imshow(p1, color_continuous_scale="YlOrRd", aspect="auto", title="Station × Month",
                             labels=dict(color=unit)), use_container_width=True)
    d2 = d_main.assign(weekday=d_main.timestamp.dt.day_name())
    p2 = d2.pivot_table(index="weekday", columns="hour", values="value", aggfunc="mean").reindex(
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    b.plotly_chart(px.imshow(p2, color_continuous_scale="YlOrRd", aspect="auto", title="Weekday × Hour of day",
                             labels=dict(color=unit)), use_container_width=True)
    if len(pollutants) > 1:
        corr = df.pivot_table(index=["timestamp", "station_name"], columns="pollutant", values="value").corr()
        st.plotly_chart(px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                                  title="Pollutant correlation matrix"), use_container_width=True)

# ---------- Map ----------
with tab4:
    m = d_main.groupby(["station_name", "city", "lat", "lon"])["value"].mean().reset_index()
    map_kwargs = dict(lat="lat", lon="lon", size="value", color="value", hover_name="station_name",
                      hover_data={"city": True, "value": ":.1f", "lat": False, "lon": False},
                      color_continuous_scale="YlOrRd", size_max=35, zoom=4, height=600,
                      title=f"Average {main_pol} ({unit}) at each station")
    if hasattr(px, "scatter_map"):            # Plotly 5.24+ / 6.x
        fig = px.scatter_map(m, **map_kwargs)
        fig.update_layout(map_style="open-street-map")
    else:                                     # older Plotly
        fig = px.scatter_mapbox(m, **map_kwargs)
        fig.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig, use_container_width=True)

# ---------- Insights (analytics) ----------
with tab5:
    st.subheader("Trend direction per station")
    rows = []
    for s_, g in agg(d_main.assign(timestamp=d_main.timestamp)).groupby("station_name"):
        g = g.dropna()
        if len(g) > 2:
            slope = np.polyfit(np.arange(len(g)), g.value.values, 1)[0]
            pct = slope * len(g) / g.value.mean() * 100
            rows.append({"Station": s_, "Mean": round(g.value.mean(), 1), "Change over period (%)": round(pct, 1),
                         "Direction": "🔺 Worsening" if pct > 5 else "🔻 Improving" if pct < -5 else "➖ Stable"})
    st.dataframe(pd.DataFrame(rows).sort_values("Change over period (%)", ascending=False), use_container_width=True, hide_index=True)

    st.subheader("Anomaly detection (daily mean, rolling z-score > 2.5)")
    dly = d_main.set_index("timestamp").groupby("station_name")["value"].resample("D").mean().reset_index()
    g = dly.groupby("station_name")["value"]
    dly["z"] = (dly["value"] - g.transform(lambda x: x.rolling(30, min_periods=10).mean())) / \
               g.transform(lambda x: x.rolling(30, min_periods=10).std())
    an = dly[dly.z > 2.5]
    fig = px.line(dly, x="timestamp", y="value", color="station_name", title="Daily mean with anomalies marked")
    fig.add_scatter(x=an.timestamp, y=an.value, mode="markers", marker=dict(color="red", size=9, symbol="x"), name="Anomaly")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"{len(an)} anomalous days detected.")

    if pd.notna(limit):
        st.subheader("Exceedance of WHO guideline")
        ex = (d_main.groupby(["station_name", "date"])["value"].mean().gt(limit)
              .groupby("station_name").mean().mul(100).round(1).reset_index(name="% days above limit"))
        st.plotly_chart(px.bar(ex, x="station_name", y="% days above limit", color="% days above limit",
                               color_continuous_scale="Reds"), use_container_width=True)

    if main_pol == "PM2.5":   # CPCB (India) PM2.5 categories, daily mean
        st.subheader("Air-quality category share (PM2.5, CPCB bands)")
        dd = d_main.groupby(["station_name", "date"])["value"].mean().reset_index()
        bins = [0, 30, 60, 90, 120, 250, 1e9]
        labels = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
        dd["Category"] = pd.cut(dd.value, bins, labels=labels)
        share = dd.groupby(["station_name", "Category"], observed=False).size().reset_index(name="days")
        st.plotly_chart(px.bar(share, x="station_name", y="days", color="Category", barmode="stack",
                               color_discrete_sequence=["#2ecc71", "#a3d977", "#f1c40f", "#e67e22", "#e74c3c", "#7b241c"]),
                        use_container_width=True)

# ---------- Data ----------
with tab6:
    st.dataframe(ts.rename(columns={"value": f"{main_pol} ({unit})"}).round(2), use_container_width=True)
    st.download_button("⬇️ Download aggregated data (CSV)", ts.to_csv(index=False), "pollutant_trends.csv", "text/csv")
    with st.expander("Warehouse schema"):
        st.code("dim_station(station_id, station_name, city, lat, lon)\n"
                "dim_pollutant(pollutant_id, pollutant, unit, who_limit)\n"
                "fact_readings(timestamp, date, year, month, hour, station_id, pollutant_id, value)", language="sql")
