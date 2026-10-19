"""LogiRoute: capacity-constrained routing, optional OSM roads & PyTorch demo."""
from pathlib import Path
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from src.io_utils import parse_orders
from src.routing import distance_matrix, heuristic, ortools_solve, route_distance, validate

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title='LogiRoute', page_icon='🚚', layout='wide')
st.title('🚚 LogiRoute — Vehicle Routing & Dispatch')
st.caption('Compare capacity-constrained routing algorithms using Euclidean coordinates or OpenStreetMap drive-network distances.')
with st.sidebar:
    st.header('Scenario')
    mode = st.radio('Distance source', ['Planar example', 'OpenStreetMap roads'])
    example = 'sample_orders.csv' if mode == 'Planar example' else 'sample_geo_orders.csv'
    upload = st.file_uploader('Upload orders CSV', type='csv')
    vehicles = st.number_input('Available vehicles', 1, 30, 4)
    capacity = st.number_input('Vehicle capacity', 1, 100000, 30)
    limit = st.slider('OR-Tools limit (seconds)', 1, 20, 3)
    st.download_button('Download example CSV', (ROOT/'data'/example).read_bytes(), example, 'text/csv')
    st.caption('First row is depot (demand 0). Planar: id,x,y,demand. OSM: id,lat,lon,demand.')

@st.cache_data(ttl=3600, show_spinner='Downloading OSM road network and computing shortest paths...')
def cached_roads(csv_text):
    from src.road_network import road_distance_matrix
    from io import StringIO
    geo = pd.read_csv(StringIO(csv_text))
    return road_distance_matrix(geo)

try:
    raw = pd.read_csv(upload) if upload else pd.read_csv(ROOT/'data'/example)
    if mode == 'Planar example':
        orders = parse_orders(raw)
        distances = distance_matrix(orders[['x','y']].to_numpy())
        road_polylines = None
        metric = 'Planar distance (units)'
    else:
        cols = ['id','lat','lon','demand']
        if not set(cols).issubset(raw.columns):
            raise ValueError('OSM mode requires id,lat,lon,demand')
        # Reuse the strict demand / ID validator.
        check = parse_orders(raw.rename(columns={'lon':'x','lat':'y'}))
        orders = raw[cols].copy().reset_index(drop=True)
        orders['demand'] = check['demand']
        orders[['lat','lon']] = check[['y','x']]
        distances, road_polylines = cached_roads(orders.to_csv(index=False))
        metric = 'Road distance (km)'
        st.info('OSM mode requires internet on first use. Routes follow downloaded drivable streets; distances are road lengths, not live traffic times.')
    demands = orders['demand'].tolist()
    st.subheader('Orders')
    st.dataframe(orders, width='stretch', height=190)
    if sum(demands) > vehicles * capacity or max(demands) > capacity:
        st.error('Fleet capacity insufficient; increase vehicles or capacity.')
        st.stop()

    costs = distances
    cost_metric = metric
    if mode == 'OpenStreetMap roads':
        with st.expander('Experimental PyTorch trip-time model (optional)', expanded=False):
            st.warning('The built-in demonstration trains on SYNTHETIC trip durations. It is not a trained real-world traffic forecasting model. For empirical use, upload observed trip data.')
            enable_ml = st.checkbox('Use PyTorch-estimated trip duration as routing cost')
            trip_upload = st.file_uploader('Optional observed trip records: distance_km,departure_hour,observed_minutes', type=['csv'], key='trip_upload')
            departure_hour = st.slider('Assumed departure hour', 0, 23, 8)
            if enable_ml:
                try:
                    from src.traffic_ml import evaluate_duration_model, simulated_observations, predict_duration_matrix
                    records = pd.read_csv(trip_upload) if trip_upload else simulated_observations()
                    with st.spinner('Training a small PyTorch MLP...'):
                        model, evaluation = evaluate_duration_model(records)
                        costs = predict_duration_matrix(model, distances, departure_hour)
                    cost_metric = 'Estimated travel time (minutes)'
                    st.caption(f'Model trained on {evaluation["sizes"]["train"]} observations; validation {evaluation["sizes"]["validation"]}, test {evaluation["sizes"]["test"]}. Data: {"UPLOADED" if trip_upload else "SYNTHETIC"}.')
                    eval_rows = [{'Split': split.title(), 'PyTorch MAE (min)': round(evaluation[split]['model_mae'], 3), 'Constant-speed baseline MAE (min)': round(evaluation[split]['baseline_mae'], 3)} for split in ('train', 'validation', 'test')]
                    st.dataframe(pd.DataFrame(eval_rows), hide_index=True, width='stretch')
                    st.caption(evaluation['split_note'])
                    if trip_upload:
                        st.warning('Uploaded records are randomly split for this demonstration. If the observations form a time series, use chronological validation before claiming future forecasting performance.')
                except ImportError:
                    st.error('PyTorch not installed. Install: pip install -r requirements-advanced.txt')
                    st.stop()
    results = {}
    for label, fn in [
        ('Nearest neighbour', lambda: heuristic(costs, demands, capacity, vehicles, False)),
        ('Nearest neighbour + 2-opt', lambda: heuristic(costs, demands, capacity, vehicles, True)),
        ('OR-Tools CVRP', lambda: ortools_solve(costs, demands, capacity, vehicles, limit)),
    ]:
        try:
            results[label] = fn()
        except (ValueError, ImportError) as exc:
            st.warning(f'{label}: {exc}')
    if not results:
        st.error('No feasible solution returned. Try more vehicles, larger capacities, or a different scenario.')
        st.stop()
    table = []
    for name, sol in results.items():
        row = {'Method': name, cost_metric: round(sol.total_distance, 2), 'Runtime (s)': round(sol.runtime_seconds, 4), 'Vehicles used': len(sol.routes)}
        if mode == 'OpenStreetMap roads' and costs is not distances:
            row['Physical road distance (km)'] = round(sum(route_distance(r, distances) for r in sol.routes), 2)
        table.append(row)
    st.subheader('Algorithm comparison')
    st.dataframe(pd.DataFrame(table), hide_index=True, width='stretch')
    chosen = st.selectbox('Visualise solution', list(results))
    sol = results[chosen]
    palette = ['#0072B2','#D55E00','#009E73','#CC79A7','#E69F00','#56B4E9','#F0E442']
    fig = go.Figure()
    if mode == 'Planar example':
        for i, route in enumerate(sol.routes):
            fig.add_trace(go.Scatter(x=[orders.loc[j,'x'] for j in route], y=[orders.loc[j,'y'] for j in route], mode='lines+markers', name=f'Vehicle {i+1}', line=dict(color=palette[i%len(palette)], width=3), text=[str(orders.loc[j,'id']) for j in route], hovertemplate='%{text}<extra></extra>'))
        fig.update_layout(xaxis_title='X (planar units)', yaxis_title='Y (planar units)', height=560)
        fig.update_yaxes(scaleanchor='x', scaleratio=1)
        st.plotly_chart(fig, width='stretch')
    else:
        for i, route in enumerate(sol.routes):
            lat, lon = [], []
            for a,b in zip(route[:-1],route[1:]):
                poly = road_polylines[(a,b)] if a != b else []
                lat.extend([p[0] for p in poly]+[None]);lon.extend([p[1] for p in poly]+[None])
            fig.add_trace(go.Scattermap(lat=lat, lon=lon, mode='lines', line=dict(width=4, color=palette[i%len(palette)]), name=f'Vehicle {i+1}'))
        fig.add_trace(go.Scattermap(lat=orders['lat'], lon=orders['lon'], mode='markers+text', text=orders['id'], textposition='top right', marker=dict(size=10), name='Stops'))
        fig.update_layout(map=dict(style='open-street-map', center=dict(lat=orders['lat'].mean(), lon=orders['lon'].mean()), zoom=12), height=580, margin=dict(t=0,l=0,r=0,b=0))
        st.plotly_chart(fig, width='stretch')
    rows = [{'vehicle':i+1, 'route':' → '.join(str(orders.loc[j,'id']) for j in r), 'load':sum(demands[j] for j in r[1:-1]), cost_metric:round(route_distance(r,costs),2)} for i,r in enumerate(sol.routes)]
    st.subheader('Vehicle assignments')
    st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)
    st.download_button('Download vehicle assignments', pd.DataFrame(rows).to_csv(index=False), 'routes.csv', 'text/csv')
    with st.expander('Solution feasibility checks', expanded=False):
        try:
            validate(sol.routes, demands, capacity, vehicles)
            st.success(f'Validated: {len(demands)-1} unique customers served, fleet size and capacities respected.')
        except ValueError as exc:
            st.error(f'Route validation failed: {exc}')
    st.divider()
    st.subheader('Multi-scenario benchmark')
    st.caption('Independent reproducible SYNTHETIC planar CVRP instances. This section is not an evaluation of the Ningbo road-network scenario above.')
    from experiments.benchmark import run_benchmark
    bench_path = ROOT / 'experiments' / 'benchmark_results.csv'
    if st.button('Run 5-scenario benchmark', type='secondary', help='Runs three solvers on five synthetic CVRP scenarios; OR-Tools may take several seconds.'):
        with st.spinner('Running benchmark...'):
            st.session_state['benchmark_results'] = run_benchmark()
    if 'benchmark_results' in st.session_state:
        bench = st.session_state['benchmark_results']
    elif bench_path.exists():
        bench = pd.read_csv(bench_path)
    else:
        bench = None
    if bench is None:
        st.info('Click Run 5-scenario benchmark to produce results, or run python -m experiments.benchmark from the terminal.')
    else:
        from src.reporting import summarize_benchmark
        summary = summarize_benchmark(bench)
        st.dataframe(summary.round(3), hide_index=True, width='stretch')
        left, right = st.columns(2)
        with left:
            st.markdown('**Mean route length (planar units)**')
            st.bar_chart(summary.set_index('method')['mean_distance'])
        with right:
            st.markdown('**Mean solver runtime (seconds)**')
            st.bar_chart(summary.set_index('method')['mean_runtime_s'])
        st.caption('Lower distance is better; runtime is a trade-off. Methods may use different numbers of vehicles. Feasibility is checked per instance.')
        st.download_button('Download benchmark CSV', bench.to_csv(index=False), 'benchmark_results.csv', 'text/csv')
except Exception as exc:
    st.error(f'Scenario error: {exc}')
