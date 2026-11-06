"""Mini data warehouse (star schema) on SQLite."""
import sqlite3
import pandas as pd

DB = "pollution_dw.db"
LIMITS = {"PM2.5": (15, "µg/m³"), "PM10": (45, "µg/m³"), "NO2": (25, "µg/m³"),
          "SO2": (40, "µg/m³"), "CO": (4, "mg/m³"), "O3": (100, "µg/m³")}


def connect(db=DB):
    return sqlite3.connect(db, check_same_thread=False)


def build_warehouse(df, db=DB):
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.dropna(subset=["value"])
    df = df[df["value"] >= 0]
    st = df[["station_name", "city", "lat", "lon"]].drop_duplicates().reset_index(drop=True)
    st["station_id"] = st.index + 1
    po = pd.DataFrame({"pollutant": sorted(df["pollutant"].unique())})
    po["pollutant_id"] = po.index + 1
    po["unit"] = po["pollutant"].map(lambda p: LIMITS.get(p, (None, "µg/m³"))[1])
    po["who_limit"] = po["pollutant"].map(lambda p: LIMITS.get(p, (None, None))[0])
    f = df.merge(st, on=["station_name", "city", "lat", "lon"]).merge(po[["pollutant", "pollutant_id"]], on="pollutant")
    fact = pd.DataFrame({"timestamp": f["timestamp"].astype(str), "date": f["timestamp"].dt.strftime("%Y-%m-%d"),
                         "year": f["timestamp"].dt.year, "month": f["timestamp"].dt.month,
                         "hour": f["timestamp"].dt.hour, "station_id": f["station_id"],
                         "pollutant_id": f["pollutant_id"], "value": f["value"]})
    con = connect(db)
    st.to_sql("dim_station", con, if_exists="replace", index=False)
    po.to_sql("dim_pollutant", con, if_exists="replace", index=False)
    fact.to_sql("fact_readings", con, if_exists="replace", index=False)
    con.execute("CREATE INDEX IF NOT EXISTS ix_fact ON fact_readings(date, station_id, pollutant_id)")
    con.commit()
    return con


def dims(con):
    return pd.read_sql("SELECT * FROM dim_station", con), pd.read_sql("SELECT * FROM dim_pollutant", con)


def get_hourly(con, stations, pollutants, start, end):
    qs = ",".join("?" * len(stations)); qp = ",".join("?" * len(pollutants))
    sql = f"""SELECT f.timestamp, f.date, f.month, f.hour, s.station_name, s.city, s.lat, s.lon,
                     p.pollutant, p.unit, p.who_limit, f.value
              FROM fact_readings f
              JOIN dim_station s ON f.station_id = s.station_id
              JOIN dim_pollutant p ON f.pollutant_id = p.pollutant_id
              WHERE s.station_name IN ({qs}) AND p.pollutant IN ({qp}) AND f.date BETWEEN ? AND ?"""
    df = pd.read_sql(sql, con, params=[*stations, *pollutants, str(start), str(end)])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df