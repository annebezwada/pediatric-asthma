import streamlit as st
from streamlit_folium import st_folium
import requests
import folium
from dataclasses import dataclass
from typing import List, Tuple, Optional
from statistics import mean
from urllib.parse import quote_plus
import datetime as dt

#  CONFIG: API KEYS 
# =============================================================================

GEOAPIFY_KEY = "f379915c3e174f519467d3b78a8586ca"
AIRNOW_KEY = "4288390E-A717-4DB1-9B4C-6BC8A983236D"

# =============================================================================
#  CORE HELPERS: GEOAPIFY + AIRNOW
# =============================================================================

def geoapify_geocode(place: str) -> Tuple[float, float, str]:
    """Return (lat, lon, label) for a place name using Geoapify."""
    url = "https://api.geoapify.com/v1/geocode/search"
    params = {"text": place, "apiKey": GEOAPIFY_KEY}
    r = requests.get(url, params=params, timeout=20)
    data = r.json()
    features = data.get("features", [])
    if not features:
        raise ValueError(f"No geocoding result for '{place}'. Response: {data}")
    feat = features[0]
    lon, lat = feat["geometry"]["coordinates"]
    label = feat["properties"].get("formatted", place)
    return float(lat), float(lon), label


def geoapify_route(
    start: Tuple[float, float],
    end: Tuple[float, float],
    route_type: str = "balanced",
    avoid: Optional[str] = None,
) -> dict:
    """
    Get a single driving route between two points using Geoapify Routing API (GeoJSON).
    start/end: (lat, lon)
    route_type: 'balanced', 'short', 'less_maneuvers'
    avoid: 'highways', 'tolls', etc. (optional)
    """
    url = "https://api.geoapify.com/v1/routing"
    waypoints = f"{start[0]},{start[1]}|{end[0]},{end[1]}"
    params = {
        "waypoints": waypoints,
        "mode": "drive",
        "type": route_type,
        "format": "geojson",
        "apiKey": GEOAPIFY_KEY,
    }
    if avoid:
        params["avoid"] = avoid

    r = requests.get(url, params=params, timeout=30)
    data = r.json()
    features = data.get("features", [])
    if not features:
        raise ValueError(f"No routes returned by Geoapify. Response: {data}")
    feature = features[0]
    props = feature.get("properties", {})
    distance_m = float(props.get("distance", 0.0))
    duration_s = float(props.get("time", 0.0))

    # GeoJSON MultiLineString -> flatten to list of (lat, lon)
    geom = feature.get("geometry", {})
    coords = geom.get("coordinates", [])
    latlon_points: List[Tuple[float, float]] = []
    for line in coords:
        for lon, lat in line:
            latlon_points.append((float(lat), float(lon)))

    if not latlon_points:
        raise ValueError(f"Route geometry missing. Response: {data}")

    return {
        "distance_m": distance_m,
        "duration_s": duration_s,
        "coords": latlon_points,
    }


def airnow_aqi(lat: float, lon: float, distance_miles: int = 25) -> Optional[int]:
    """Get current AQI near a point from AirNow. Returns max AQI or None."""
    url = "https://www.airnowapi.org/aq/observation/latLong/current"
    params = {
        "format": "application/json",
        "latitude": f"{lat:.4f}",
        "longitude": f"{lon:.4f}",
        "distance": str(distance_miles),
        "API_KEY": AIRNOW_KEY,
    }
    r = requests.get(url, params=params, timeout=20)
    try:
        data = r.json()
    except Exception:
        return None

    if not data or (isinstance(data, dict) and data.get("Message")):
        return None

    try:
        return max(int(item.get("AQI", 0)) for item in data)
    except Exception:
        return None

# =============================================================================
#  ROUTE SAMPLING + SCORING
# =============================================================================

@dataclass
class RouteScore:
    name: str
    distance_km: float
    duration_min: float
    avg_aqi: float
    max_aqi: int
    num_samples: int
    maps_url: str
    coords: List[Tuple[float, float]]  # for map


def sample_points(coords: List[Tuple[float, float]], max_samples: int = 10) -> List[Tuple[float, float]]:
    """Downsample route coordinates so we don't spam AirNow."""
    if len(coords) <= max_samples:
        return coords
    step = max(1, len(coords) // max_samples)
    return [coords[i] for i in range(0, len(coords), step)][:max_samples]


def score_route(
    coords: List[Tuple[float, float]],
    origin_label: str,
    dest_label: str,
    name: str,
    distance_m: float,
    duration_s: float,
) -> Optional[RouteScore]:
    pts = sample_points(coords, max_samples=10)
    aqis = []
    for lat, lon in pts:
        aqi = airnow_aqi(lat, lon)
        if aqi is not None:
            aqis.append(aqi)

    if not aqis:
        return None

    avg_aqi = mean(aqis)
    max_aqi = max(aqis)
    maps_url = f"https://www.google.com/maps/dir/{quote_plus(origin_label)}/{quote_plus(dest_label)}"

    return RouteScore(
        name=name,
        distance_km=distance_m / 1000.0,
        duration_min=duration_s / 60.0,
        avg_aqi=avg_aqi,
        max_aqi=max_aqi,
        num_samples=len(aqis),
        maps_url=maps_url,
        coords=coords,
    )


def plan_clean_routes_geoapify(origin: str, destination: str) -> List[RouteScore]:
    """
    1. Geocode origin/destination.
    2. Get multiple driving routes (short, balanced, avoid highways).
    3. Sample AQI along each route.
    4. Rank routes by pollution exposure.
    """
    o_lat, o_lon, o_label = geoapify_geocode(origin)
    d_lat, d_lon, d_label = geoapify_geocode(destination)
    start = (o_lat, o_lon)
    end = (d_lat, d_lon)

    configs = [
        ("Shortest", {"route_type": "short", "avoid": None}),
        ("Balanced", {"route_type": "balanced", "avoid": None}),
        ("Avoid highways", {"route_type": "balanced", "avoid": "highways"}),
    ]

    scores: List[RouteScore] = []

    for name, cfg in configs:
        try:
            r = geoapify_route(start, end, **cfg)
        except Exception:
            continue

        s = score_route(
            coords=r["coords"],
            origin_label=o_label,
            dest_label=d_label,
            name=name,
            distance_m=r["distance_m"],
            duration_s=r["duration_s"],
        )
        if s:
            scores.append(s)

    if not scores:
        raise ValueError("Could not score any routes (routing or AQI failed).")

    scores.sort(key=lambda s: (s.avg_aqi, s.max_aqi))
    return scores

# =============================================================================
#  FORECAST HELPERS (WHEN TO TRAVEL)
# =============================================================================

def airnow_forecast_zip(zip_code: str, distance_miles: int = 25):
    """
    Get AirNow AQI forecast for a ZIP code for the next few days.
    Returns list of {date, aqi, category, pollutant}.
    """
    url = "https://www.airnowapi.org/aq/forecast/zipCode/"
    params = {
        "format": "application/json",
        "zipCode": zip_code,
        "distance": str(distance_miles),
        "API_KEY": AIRNOW_KEY,
    }
    r = requests.get(url, params=params, timeout=20)
    try:
        data = r.json()
    except Exception:
        raise ValueError(f"Could not decode AirNow forecast response:\n{r.text[:300]}")

    if not data or (isinstance(data, dict) and data.get("Message")):
        raise ValueError(f"No forecast data returned for ZIP {zip_code}. Response: {data}")

    forecast_days = []
    for item in data:
        try:
            date_str = item.get("DateForecast") or item.get("DateIssue")
            date_obj = dt.datetime.strptime(date_str.split("T")[0], "%Y-%m-%d").date()
        except Exception:
            date_obj = dt.date.today()

        forecast_days.append({
            "date": date_obj,
            "aqi": int(item.get("AQI", 0)),
            "category": item.get("Category", {}).get("Name", "Unknown"),
            "pollutant": item.get("ParameterName", "Unknown"),
        })

    # Merge pollutants by date (keep worst AQI per day)
    by_date = {}
    for f in forecast_days:
        d = f["date"]
        if d not in by_date or f["aqi"] > by_date[d]["aqi"]:
            by_date[d] = f

    return [by_date[d] for d in sorted(by_date.keys())]


def suggest_best_travel_day(zip_code: str, look_ahead_days: int = 3):
    """Suggest best day in next N days (min AQI)."""
    all_fc = airnow_forecast_zip(zip_code)
    today = dt.date.today()
    cutoff = today + dt.timedelta(days=look_ahead_days)

    fc = [f for f in all_fc if today <= f["date"] <= cutoff]
    if not fc:
        raise ValueError("No forecast entries in the desired look-ahead window.")

    best = min(fc, key=lambda x: x["aqi"])
    return {"candidates": fc, "best": best}

# =============================================================================
#  MAP VISUALIZATION
# =============================================================================

def aqi_color(aqi: float) -> str:
    """Return a color string based on AQI level."""
    if aqi <= 50:
        return "green"
    if aqi <= 100:
        return "yellow"
    if aqi <= 150:
        return "orange"
    if aqi <= 200:
        return "red"
    return "purple"


def show_routes_map(routes: List[RouteScore]) -> folium.Map:
    """
    Show all candidate routes on an interactive map.
    Color based on AQI; cleanest route drawn thicker.
    """
    all_coords = [pt for r in routes for pt in r.coords]
    center_lat = sum(lat for lat, _ in all_coords) / len(all_coords)
    center_lon = sum(lon for _, lon in all_coords) / len(all_coords)

    m = folium.Map(location=[center_lat, center_lon], zoom_start=11)

    for idx, r in enumerate(routes):
        color = aqi_color(r.avg_aqi)
        tooltip = (
            f"{r.name}: avg AQI {r.avg_aqi:.1f}, max {r.max_aqi}, "
            f"{r.distance_km:.1f} km / {r.duration_min:.0f} min"
        )

        folium.PolyLine(
            locations=[(lat, lon) for lat, lon in r.coords],
            color=color,
            weight=6 if idx == 0 else 4,
            opacity=0.9 if idx == 0 else 0.6,
            tooltip=tooltip,
        ).add_to(m)

    first = routes[0]
    start_lat, start_lon = first.coords[0]
    end_lat, end_lon = first.coords[-1]

    folium.Marker([start_lat, start_lon], popup="Start").add_to(m)
    folium.Marker([end_lat, end_lon], popup="Destination").add_to(m)

    return m

# =============================================================================
#  DISPLAY HELPERS (SO RESULTS DON'T DISAPPEAR)
# =============================================================================

def render_plan(plan: dict):
    """Render forecast + routes from a cached plan dict."""
    candidates = plan["forecast_candidates"]
    best_day = plan["forecast_best_day"]
    routes = plan["routes"]
    best_route = plan["best_route"]

    # WHEN SECTION
    st.subheader("When should you go?")

    col1, col2 = st.columns([2, 1])
    with col1:
        rows = [
            {
                "Date": f["date"].strftime("%a %b %d"),
                "AQI": f["aqi"],
                "Category": f["category"],
                "Pollutant": f["pollutant"],
            }
            for f in candidates
        ]
        st.write("Forecast for the next few days (worst pollutant per day):")
        st.table(rows)

    with col2:
        st.metric(
            label="Suggested day for non-urgent trips",
            value=best_day["date"].strftime("%a %b %d"),
            delta=f"AQI {best_day['aqi']} ({best_day['category']})",
        )

    # HOW SECTION
    st.subheader("How should you go?")

    route_rows = []
    for r in routes:
        route_rows.append({
            "Route": r.name,
            "Distance (km)": f"{r.distance_km:.1f}",
            "Time (min)": f"{r.duration_min:.0f}",
            "Avg AQI": f"{r.avg_aqi:.1f}",
            "Max AQI": r.max_aqi,
        })
    st.write("Routes ranked from **cleanest** (top) to **dirtiest**:")
    st.table(route_rows)

    st.success(
        f"Recommended route: **{best_route.name}** "
        f"({best_route.distance_km:.1f} km / {best_route.duration_min:.0f} min, "
        f"avg AQI ≈ {best_route.avg_aqi:.1f}, max AQI {best_route.max_aqi})."
    )
    st.markdown(
        f"[Open in Google Maps]({best_route.maps_url})  "
        "(route shape is approximate but works for navigation)."
    )

    # MAP SECTION
    st.subheader("Route map (color-coded by average AQI)")
    fmap = show_routes_map(routes)
    st_folium(fmap, width=900, height=500)

# =============================================================================
#  STREAMLIT APP UI (PRETTY VERSION)
# =============================================================================

st.set_page_config(
    page_title="Asthma Guardian – Clean Route Planner",
    page_icon="💙",
    layout="wide"
)

# --- Custom CSS for colors, cards, and background ---
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #fdfbfb 0%, #e3f2fd 100%);
    }
    .main > div {
        padding-top: 1rem;
    }
    .big-title {
        font-size: 2.4rem;
        font-weight: 700;
        color: #264653;
        margin-bottom: 0.25rem;
    }
    .subtitle {
        font-size: 1rem;
        color: #555;
        margin-bottom: 0.5rem;
    }
    .pill {
        background-color: #ffe0f0;
        color: #ad1457;
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 0.8rem;
        display: inline-block;
        margin-bottom: 0.75rem;
    }
    .card {
        background-color: #ffffff;
        padding: 1.1rem 1rem;
        border-radius: 0.9rem;
        box-shadow: 0 4px 16px rgba(0,0,0,0.04);
        margin-bottom: 1rem;
        border: 1px solid #f1f1f1;
    }
    .section-title {
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 0.4rem;
        color: #1f3b4d;
    }
    .small-label {
        font-size: 0.85rem;
        color: #666;
    }
    .sidebar-title {
        font-weight: 600;
        font-size: 1.1rem;
        margin-bottom: 0.3rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Top header with optional mascot image ---
header_col1, header_col2 = st.columns([3, 1])

with header_col1:
    st.markdown('<div class="pill">Agentic AI · Environmental Health</div>', unsafe_allow_html=True)
    st.markdown('<div class="big-title">Asthma Guardian</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Helping expecting parents find cleaner-air routes '
        'and safer travel days for baby lungs. 💙</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        "_For educational use only – not medical advice. Always consult your clinician for health decisions._"
    )
with header_col2:
    # If you add an image file like assets/guardian_owl.png, it will show here.
    # Otherwise, this just shows a placeholder emoji.
    try:
        st.image("assets/guardian_owl.png", width=120)
    except Exception:
        st.markdown("🦉", unsafe_allow_html=True)
        st.caption("Asthma Guardian")

with st.expander("💡 What this app does", expanded=False):
    st.write(
        "- Uses **Geoapify** for driving routes\n"
        "- Uses **AirNow** for air quality and forecasts\n"
        "- Compares multiple route types (shortest, balanced, avoid highways)\n"
        "- Samples AQI along each route and ranks them by exposure\n"
        "- Suggests a **cleaner route** and **better day** for non-urgent trips\n"
    )

# --- Sidebar settings ---
st.sidebar.markdown('<div class="sidebar-title">🌎 Travel Settings</div>', unsafe_allow_html=True)
home_zip = st.sidebar.text_input("Home ZIP code", value="20874")
look_ahead_days = st.sidebar.slider("Days ahead to check air quality", min_value=1, max_value=7, value=3)
st.sidebar.markdown("---")
st.sidebar.markdown("**Tip:** Try your home → hospital route and see how air quality changes.")

# --- Main content: trip planner + results ---
st.markdown("### ✨ Plan a cleaner-air trip")

with st.container():
    with st.form("trip_form"):
        col_a, col_b = st.columns(2)
        with col_a:
            origin = st.text_input("Starting address", "Germantown, MD")
        with col_b:
            destination = st.text_input("Destination", "Children's National Hospital, Washington, DC")

        st.markdown(
            '<span class="small-label">The app will compare multiple routes and pick the one '
            'with the lowest air pollution exposure.</span>',
            unsafe_allow_html=True
        )
        submitted = st.form_submit_button("🌈 Plan cleanest trip")

# Show last successful result so it doesn't disappear on reruns
if "last_plan" in st.session_state:
    st.markdown("### 📊 Your latest Asthma Guardian plan")
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        render_plan(st.session_state["last_plan"])
        st.markdown('</div>', unsafe_allow_html=True)

# Handle new submission
if submitted:
    try:
        with st.spinner("Analyzing forecast and routes… breathing-friendly planning in progress 🌬️"):
            forecast_plan = suggest_best_travel_day(home_zip, look_ahead_days)
            routes = plan_clean_routes_geoapify(origin, destination)

        plan = {
            "forecast_candidates": forecast_plan["candidates"],
            "forecast_best_day": forecast_plan["best"],
            "routes": routes,
            "best_route": routes[0],
        }

        st.session_state["last_plan"] = plan  # cache results so they persist
        st.rerun()

    except Exception as e:
        st.error(f"Something went wrong: {e}")



