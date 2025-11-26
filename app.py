import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from utils import generate_load_profile, generate_solar_profile, generate_wind_profile

st.set_page_config(page_title="Aggregated Procurement Tool", layout="wide")

st.title("Aggregated Procurement of Firm Power (ERCOT)")
st.markdown("### Individual Portfolios vs. Aggregated Pool")

# --- Sidebar Inputs ---
st.sidebar.header("1. Define Participants")

# Initialize session state for companies if not exists
if 'companies' not in st.session_state:
    st.session_state.companies = [
        {'name': 'Tech Corp', 'demand': 150000, 'profile': 'Data Center', 'solar_mw': 40.0, 'wind_mw': 10.0},
        {'name': 'City Hospital', 'demand': 40000, 'profile': 'Health Care', 'solar_mw': 5.0, 'wind_mw': 5.0},
        {'name': 'Mega Mall', 'demand': 25000, 'profile': 'Retail', 'solar_mw': 10.0, 'wind_mw': 0.0},
        {'name': 'Heavy Industry', 'demand': 200000, 'profile': 'Industrial', 'solar_mw': 20.0, 'wind_mw': 60.0}
    ]

def add_company():
    st.session_state.companies.append({
        'name': f'New Co {len(st.session_state.companies)+1}', 
        'demand': 10000, 
        'profile': 'Business',
        'solar_mw': 0.0,
        'wind_mw': 0.0
    })

def remove_company():
    if len(st.session_state.companies) > 0:
        st.session_state.companies.pop()

col1, col2 = st.sidebar.columns(2)
col1.button("Add Participant", on_click=add_company)
col2.button("Remove Last", on_click=remove_company)

# Display Company Inputs
updated_companies = []
for i, comp in enumerate(st.session_state.companies):
    with st.sidebar.expander(f"{comp['name']}", expanded=False):
        name = st.text_input(f"Name", value=comp['name'], key=f"name_{i}")
        c1, c2 = st.columns(2)
        demand = c1.number_input(f"Load (MWh)", value=comp['demand'], step=1000, key=f"demand_{i}")
        
        profile_options = ['Business', 'Industrial', 'Residential', 'Data Center', 'Health Care', 'Retail']
        profile = c2.selectbox(f"Profile", profile_options, index=profile_options.index(comp['profile']), key=f"profile_{i}")
        
        st.markdown("**PPA Assets**")
        c3, c4 = st.columns(2)
        solar_mw = c3.number_input(f"Solar (MW)", value=comp['solar_mw'], step=1.0, key=f"solar_{i}")
        wind_mw = c4.number_input(f"Wind (MW)", value=comp['wind_mw'], step=1.0, key=f"wind_{i}")
        
        updated_companies.append({
            'name': name, 'demand': demand, 'profile': profile,
            'solar_mw': solar_mw, 'wind_mw': wind_mw
        })

st.session_state.companies = updated_companies

# --- Simulation Logic ---

results = []
aggregated_load = None
aggregated_solar = None
aggregated_wind = None

for comp in st.session_state.companies:
    # 1. Load
    load = generate_load_profile(comp['demand'], comp['profile'])
    
    # 2. Generation
    solar = generate_solar_profile(comp['solar_mw'])
    wind = generate_wind_profile(comp['wind_mw'])
    total_re = solar + wind
    
    # 3. Metrics
    total_demand = load.sum()
    total_gen = total_re.sum()
    
    # Match: Min(Load, Gen) at each hour
    matched_energy = pd.Series(index=load.index, data=[min(l, g) for l, g in zip(load, total_re)])
    match_pct = (matched_energy.sum() / total_demand) * 100 if total_demand > 0 else 0
    
    # REC Value
    rec_value = total_gen * 5 # $5/MWh
    
    results.append({
        'name': comp['name'],
        'load': load,
        'solar': solar,
        'wind': wind,
        'total_re': total_re,
        'matched_energy': matched_energy,
        'match_pct': match_pct,
        'rec_value': rec_value,
        'total_demand': total_demand,
        'total_gen': total_gen
    })
    
    # Aggregation
    if aggregated_load is None:
        aggregated_load = load.copy()
        aggregated_solar = solar.copy()
        aggregated_wind = wind.copy()
    else:
        aggregated_load += load
        aggregated_solar += solar
        aggregated_wind += wind

# Aggregated Calculations
agg_total_re = aggregated_solar + aggregated_wind
agg_matched = pd.Series(index=aggregated_load.index, data=[min(l, g) for l, g in zip(aggregated_load, agg_total_re)])
agg_match_pct = (agg_matched.sum() / aggregated_load.sum()) * 100 if aggregated_load.sum() > 0 else 0
agg_rec_value = agg_total_re.sum() * 5

# --- Dashboard ---

# 1. Summary Metrics (Aggregated)
st.markdown("### Aggregated Pool Performance")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Demand", f"{aggregated_load.sum()/1000:,.1f} GWh")
m2.metric("Total Generation", f"{agg_total_re.sum()/1000:,.1f} GWh")
m3.metric("Pool Match %", f"{agg_match_pct:.1f}%")
m4.metric("Total REC Value ($5/MWh)", f"${agg_rec_value:,.0f}")

# 2. Comparison Chart (Individual vs Pool)
st.subheader("Match % Comparison: Individual vs Aggregated")
comparison_data = []
for r in results:
    comparison_data.append({'Name': r['name'], 'Match %': r['match_pct'], 'Type': 'Individual'})
comparison_data.append({'Name': 'Aggregated Pool', 'Match %': agg_match_pct, 'Type': 'Pool'})

df_comp = pd.DataFrame(comparison_data)
fig_comp = px.bar(df_comp, x='Name', y='Match %', color='Type', text_auto='.1f', color_discrete_map={'Individual': 'gray', 'Pool': 'green'})
fig_comp.add_hline(y=agg_match_pct, line_dash="dot", annotation_text="Pool Level", annotation_position="top right")
st.plotly_chart(fig_comp, use_container_width=True)

# 3. Detailed View
st.markdown("---")
st.subheader("Detailed Analysis")

view_option = st.selectbox("Select View", ["Aggregated Pool"] + [c['name'] for c in st.session_state.companies])

if view_option == "Aggregated Pool":
    target_load = aggregated_load
    target_solar = aggregated_solar
    target_wind = aggregated_wind
    target_re = agg_total_re
    title = "Aggregated Pool"
else:
    # Find the specific company result
    res = next(r for r in results if r['name'] == view_option)
    target_load = res['load']
    target_solar = res['solar']
    target_wind = res['wind']
    target_re = res['total_re']
    title = res['name']

# Time Series Plot
df_plot = pd.DataFrame({
    'Load': target_load,
    'Solar': target_solar,
    'Wind': target_wind,
    'Total RE': target_re
})

week_start = st.date_input("Select Start Date", value=pd.to_datetime("2024-06-01"), key="date_picker")
start_idx = pd.Timestamp(week_start)
end_idx = start_idx + pd.Timedelta(days=7)
subset = df_plot[start_idx:end_idx]

fig_ts = go.Figure()
fig_ts.add_trace(go.Scatter(x=subset.index, y=subset['Load'], name='Load', line=dict(color='black', width=3)))
fig_ts.add_trace(go.Scatter(x=subset.index, y=subset['Solar'], name='Solar', fill='tozeroy', line=dict(color='orange', width=0)))
fig_ts.add_trace(go.Scatter(x=subset.index, y=subset['Wind'], name='Wind', fill='tonexty', line=dict(color='cyan', width=0)))
fig_ts.update_layout(title=f"{title} - Hourly Dispatch (First Week)", xaxis_title="Time", yaxis_title="MW", height=400)
st.plotly_chart(fig_ts, use_container_width=True)

# Monthly View
monthly = df_plot.resample('M').sum() / 1000 # GWh
monthly.index = monthly.index.strftime('%b')
fig_bar = px.bar(monthly, x=monthly.index, y=['Solar', 'Wind'], title=f"{title} - Monthly Energy Balance")
fig_bar.add_trace(go.Scatter(x=monthly.index, y=monthly['Load'], name='Load', line=dict(color='black', width=3)))
st.plotly_chart(fig_bar, use_container_width=True)
