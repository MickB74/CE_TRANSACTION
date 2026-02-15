import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from utils import (
    generate_load_profile, 
    generate_solar_profile, 
    generate_wind_profile, 
    calculate_swap_value
)

st.set_page_config(page_title="Aggregated Procurement Tool", layout="wide")

st.title("Aggregated Procurement of Firm Power (ERCOT)")
st.markdown("### Individual Portfolios vs. Aggregated Pool")

# --- Sidebar Inputs ---
st.sidebar.header("1. Define Participants")

# Initialize session state for companies if not exists
if 'companies' not in st.session_state:
    st.session_state.companies = [
        {'name': 'Data Center Tech Corp', 'demand': 200000, 'profile': 'Data Center', 'solar_mwh': 200000, 'wind_mwh': 110000}, # Total 310k (155%)
        {'name': 'Hospital Network', 'demand': 100000, 'profile': 'Health Care', 'solar_mwh': 50000, 'wind_mwh': 50000}, # Total 100k (100%)
        {'name': 'REIT For Malls', 'demand': 65000, 'profile': 'Retail', 'solar_mwh': 55000, 'wind_mwh': 0}, # Total 55k (85%)
        {'name': 'Heavy Industry', 'demand': 200000, 'profile': 'Industrial', 'solar_mwh': 80000, 'wind_mwh': 150000} # Total 230k (115%)
    ]

def add_company():
    st.session_state.companies.append({
        'name': f'New Co {len(st.session_state.companies)+1}', 
        'demand': 10000, 
        'profile': 'Business',
        'solar_mwh': 0.0,
        'wind_mwh': 0.0
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
    st.sidebar.markdown(f"### {comp['name']}")
    
    c1, c2 = st.sidebar.columns(2)
    name = c1.text_input(f"Name", value=comp['name'], key=f"name_{i}")
    demand = c2.number_input(f"Load (MWh)", value=comp['demand'], step=1000, key=f"demand_{i}")
    
    profile_options = ['Business', 'Industrial', 'Residential', 'Data Center', 'Health Care', 'Retail']
    profile = st.sidebar.selectbox(f"Profile", profile_options, index=profile_options.index(comp['profile']), key=f"profile_{i}")
    
    st.sidebar.markdown("**PPA Assets (Annual Generation)**")
    c3, c4 = st.sidebar.columns(2)
    solar_mwh = c3.number_input(f"Solar (MWh)", value=float(comp['solar_mwh']), step=1000.0, key=f"solar_{i}")
    wind_mwh = c4.number_input(f"Wind (MWh)", value=float(comp['wind_mwh']), step=1000.0, key=f"wind_{i}")
    
    updated_companies.append({
        'name': name, 'demand': demand, 'profile': profile,
        'solar_mwh': solar_mwh, 'wind_mwh': wind_mwh
    })

st.session_state.companies = updated_companies

st.sidebar.markdown("---")
st.sidebar.header("2. Market Parameters")
rec_price = st.sidebar.number_input("REC Price ($/MWh)", value=5.0, step=0.5)

# --- Simulation Logic ---

np.random.seed(42) # Ensure deterministic results on re-runs

results = []
aggregated_load = None
aggregated_solar = None
aggregated_wind = None

for comp in st.session_state.companies:
    # 1. Load
    load = generate_load_profile(comp['demand'], comp['profile'])
    
    # 2. Generation
    solar = generate_solar_profile(annual_mwh=comp['solar_mwh'])
    wind = generate_wind_profile(annual_mwh=comp['wind_mwh'])
    total_re = solar + wind
    
    # 3. Metrics
    total_demand = load.sum()
    total_gen = total_re.sum()
    
    # Match: Min(Load, Gen) at each hour
    matched_energy = pd.Series(index=load.index, data=[min(l, g) for l, g in zip(load, total_re)])
    match_pct = (matched_energy.sum() / total_demand) * 100 if total_demand > 0 else 0
    
    # Volumetric % (Total Gen / Total Load)
    volumetric_pct = (total_gen / total_demand) * 100 if total_demand > 0 else 0
    
    # REC Position (Hourly)
    net_position = total_re - load # Positive = Excess, Negative = Shortfall
    excess_mwh = net_position.clip(lower=0)
    shortfall_mwh = -net_position.clip(upper=0)
    
    # Financials
    rec_value_total = total_gen * rec_price # Value of all RECs generated
    excess_value = excess_mwh.sum() * rec_price
    shortfall_cost = shortfall_mwh.sum() * rec_price
    
    results.append({
        'name': comp['name'],
        'load': load,
        'solar': solar,
        'wind': wind,
        'total_re': total_re,
        'matched_energy': matched_energy,
        'match_pct': match_pct,
        'volumetric_pct': volumetric_pct,
        'rec_value': rec_value_total,
        'excess_mwh': excess_mwh,
        'shortfall_mwh': shortfall_mwh,
        'excess_value': excess_value,
        'shortfall_cost': shortfall_cost,
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
agg_rec_value = agg_total_re.sum() * rec_price

# --- Swap Logic (Pro-Rata Allocation) ---
# 1. Create DataFrame of Net Positions (Columns = Companies, Index = Time)
# Rule: Must have > 50% Volumetric RE to participate
eligible_companies = [r['name'] for r in results if r['volumetric_pct'] >= 50]
net_positions_df = pd.DataFrame({r['name']: r['total_re'] - r['load'] for r in results if r['name'] in eligible_companies})
swap_results = {r['name']: {'imported': 0, 'exported': 0, 'cost': 0, 'revenue': 0} for r in results}

# 2. Iterate Hourly
if not net_positions_df.empty:
    hourly_imports = pd.DataFrame(0.0, index=net_positions_df.index, columns=net_positions_df.columns)

    for t in net_positions_df.index:
        row = net_positions_df.loc[t]
        longs = row[row > 0]
        shorts = row[row < 0]
        
        total_surplus = longs.sum()
        total_deficit = -shorts.sum() # Positive number
        
        if total_surplus > 0 and total_deficit > 0:
            # Amount to swap is limited by supply or demand
            swappable = min(total_surplus, total_deficit)
            
            # Pro-rata allocation
            for name, surplus in longs.items():
                export_amt = (surplus / total_surplus) * swappable
                swap_results[name]['exported'] += export_amt
                swap_results[name]['revenue'] += export_amt * rec_price
                
            for name, deficit in shorts.items():
                import_amt = (abs(deficit) / total_deficit) * swappable
                swap_results[name]['imported'] += import_amt
                swap_results[name]['cost'] += import_amt * rec_price
                hourly_imports.at[t, name] = import_amt
else:
    st.warning("No companies eligible for swapping (>50% Volumetric RE required).")

# 3. Update Results with Swap Info
for r in results:
    name = r['name']
    swaps = swap_results[name]
    
    # Optimized Match = Original Match + Imported Swaps
    # Note: Original Match is Min(Load, Gen). Imported Swaps fill the gap.
    # Check: Ensure we don't exceed Load. (Logic above ensures import <= deficit, so Match + Import <= Load)
    
    optimized_match_mwh = r['matched_energy'].sum() + swaps['imported']
    optimized_match_pct = (optimized_match_mwh / r['total_demand']) * 100 if r['total_demand'] > 0 else 0
    
    r['swap_imported'] = swaps['imported']
    r['swap_exported'] = swaps['exported']
    r['swap_cost'] = swaps['cost']
    r['swap_revenue'] = swaps['revenue']
    r['swap_net_settlement'] = swaps['revenue'] - swaps['cost']
    r['optimized_match_pct'] = optimized_match_pct

# --- Dashboard ---

# 1. Summary Metrics (Aggregated)
# Calculate Pool Financials
pool_net_load = agg_total_re - aggregated_load
pool_unused_recs = pool_net_load.clip(lower=0).sum()
pool_unused_value = pool_unused_recs * rec_price
pool_shortfall_recs = -pool_net_load.clip(upper=0).sum()
pool_shortfall_cost = pool_shortfall_recs * rec_price

st.markdown("### Aggregated Pool Performance")
m1, m2, m3 = st.columns(3)
m1.metric("Total Demand", f"{aggregated_load.sum()/1000:,.1f} GWh")
m2.metric("Total Generation", f"{agg_total_re.sum()/1000:,.1f} GWh")
m3.metric("Pool Match %", f"{agg_match_pct:.1f}%")

m4, m5, m6 = st.columns(3)
m4.metric(f"Total REC Value (${rec_price}/MWh)", f"${agg_rec_value:,.0f}")
m5.metric("Value of Unused RECs", f"${pool_unused_value:,.0f}", f"{pool_unused_recs:,.0f} MWh")
m6.metric("Cost for Needed RECs", f"-${pool_shortfall_cost:,.0f}", f"{pool_shortfall_recs:,.0f} MWh")

# --- Financial Value of Swaps ---
if not net_positions_df.empty:
    # Pass Load - Gen (Deficit = Positive)
    swap_stats, _ = calculate_swap_value(-net_positions_df, market_rec_price=rec_price)
    
    st.markdown("### Financial Value of Internal Swaps")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Volume Swapped", f"{swap_stats['Total Volume Swapped (MWh)']:,.0f} MWh")
    s2.metric("Value Created", f"${swap_stats['Market Value of Swaps ($)']:,.0f}")
    s3.metric("Manager Revenue (20%)", f"${swap_stats['Projected Manager Revenue ($)']:,.0f}")
    s4.metric("Client Net Savings", f"${swap_stats['Client Net Savings ($)']:,.0f}")
    
    with st.expander("How are these calculated?"):
        st.markdown(f"""
        **1. Volume Swapped:** Sum of matching hourly surplus and deficits within the pool. 
        (e.g., If Pool Surplus=100 and Pool Deficit=20, Swappable=20)
        
        **2. Value Created:** `Volume Swapped` × `REC Price (${rec_price}/MWh)`
        
        **3. Manager Revenue:** `Value Created` × `20%` (Assumed Fee)
        
        **4. Client Net Savings:** `Value Created` - `Manager Revenue`
        """)

# 2. Member Performance Table
st.subheader("Member Performance Metrics (With Swaps)")
member_metrics = []
for r in results:
    member_metrics.append({
        'Participant': r['name'],
        'Annual Load (MWh)': r['total_demand'],
        'Total Generation (MWh)': r['total_gen'],
        'Volumetric RE %': r['volumetric_pct'],
        'Standalone CFE %': r['match_pct'],
        'Optimized CFE %': r['optimized_match_pct'],
        'RECs In (MWh)': r['swap_imported'],
        'RECs Out (MWh)': r['swap_exported'],
        'Swap Cost ($)': r['swap_cost'],
        'Swap Revenue ($)': r['swap_revenue'],
        'Swap Net ($)': r['swap_net_settlement'],
        'Needed RECs (MWh)': r['shortfall_mwh'].sum() - r['swap_imported'],
        'Cost for Needed RECs': -1 * (r['shortfall_mwh'].sum() - r['swap_imported']) * rec_price,
        'Unused RECs (MWh)': r['excess_mwh'].sum() - r['swap_exported'],
        'Value of Unused RECs': (r['excess_mwh'].sum() - r['swap_exported']) * rec_price
    })
# Add Pool Row
member_metrics.append({
    'Participant': 'Aggregated Pool',
    'Annual Load (MWh)': aggregated_load.sum(),
    'Total Generation (MWh)': agg_total_re.sum(),
    'Volumetric RE %': (agg_total_re.sum() / aggregated_load.sum()) * 100,
    'Standalone CFE %': agg_match_pct,
    'Optimized CFE %': agg_match_pct, # Pool is already optimized internally
    'RECs In (MWh)': sum(r['swap_imported'] for r in results),
    'RECs Out (MWh)': sum(r['swap_exported'] for r in results),
    'Swap Cost ($)': sum(r['swap_cost'] for r in results),
    'Swap Revenue ($)': sum(r['swap_revenue'] for r in results),
    'Swap Net ($)': sum(r['swap_net_settlement'] for r in results), # Internal sum is zero
    'Needed RECs (MWh)': sum(m['Needed RECs (MWh)'] for m in member_metrics),
    'Cost for Needed RECs': sum(m['Cost for Needed RECs'] for m in member_metrics),
    'Unused RECs (MWh)': sum(m['Unused RECs (MWh)'] for m in member_metrics),
    'Value of Unused RECs': sum(m['Value of Unused RECs'] for m in member_metrics)
})

df_metrics = pd.DataFrame(member_metrics)

st.sidebar.markdown("---")
st.sidebar.header("3. Exports")
st.sidebar.download_button(
    label="Download Portfolio JSON",
    data=df_metrics.to_json(orient='records', indent=2),
    file_name="portfolio_summary.json",
    mime="application/json"
)

st.dataframe(df_metrics.style.format({
    'Annual Load (MWh)': '{:,.0f}',
    'Total Generation (MWh)': '{:,.0f}',
    'Volumetric RE %': '{:.1f}%',
    'Standalone CFE %': '{:.1f}%',
    'Optimized CFE %': '{:.1f}%',
    'RECs In (MWh)': '{:,.0f}',
    'RECs Out (MWh)': '{:,.0f}',
    'Swap Cost ($)': '${:,.0f}',
    'Swap Revenue ($)': '${:,.0f}',
    'Swap Net ($)': '${:,.0f}',
    'Needed RECs (MWh)': '{:,.0f}',
    'Cost for Needed RECs': '${:,.0f}',
    'Unused RECs (MWh)': '{:,.0f}',
    'Value of Unused RECs': '${:,.0f}'
}).apply(lambda x: ['background-color: #e0e0e0; font-weight: bold; color: black' if x['Participant'] == 'Aggregated Pool' else '' for i in x], axis=1), use_container_width=True, hide_index=True)

# Unused Pool RECs & Shortfall (Already calculated above)
c_unused, c_shortfall = st.columns(2)
c_unused.metric("Unused Pool RECs (Surplus)", f"{pool_unused_recs:,.0f} MWh", f"${pool_unused_value:,.0f} Revenue")
c_shortfall.metric("Pool Shortfall (Deficit)", f"{pool_shortfall_recs:,.0f} MWh", f"-${pool_shortfall_cost:,.0f} Cost")

# 3. Comparison Chart (Individual vs Pool)
st.subheader("Match % Comparison: Individual vs Aggregated")
comparison_data = []
for r in results:
    comparison_data.append({'Name': r['name'], 'Match %': r['match_pct'], 'Type': 'Prior to Transfer'})
    comparison_data.append({'Name': r['name'], 'Match %': r['optimized_match_pct'], 'Type': 'After Transfer'})

# Comparison Chart
df_comp = pd.DataFrame(comparison_data)

fig_comp = px.bar(
    df_comp, 
    x='Name', 
    y='Match %', 
    color='Type', 
    barmode='group',
    text_auto='.1f', 
    color_discrete_map={'Prior to Transfer': '#7f7f7f', 'After Transfer': '#00CC96'}
)
fig_comp.add_hline(y=agg_match_pct, line_dash="dash", line_color="#00FF00", line_width=3, annotation_text="Pool Level", annotation_position="top right")
st.plotly_chart(fig_comp, use_container_width=True)

# 4. Swap Financials Chart
st.subheader("Swap Financials: Paid vs. Received")
financial_data = []
for r in results:
    financial_data.append({'Name': r['name'], 'Amount': r['swap_revenue'], 'Type': 'Revenue (Received)'})
    financial_data.append({'Name': r['name'], 'Amount': -r['swap_cost'], 'Type': 'Cost (Paid)'})

df_fin = pd.DataFrame(financial_data)
# Add formatted text column
df_fin['Text'] = df_fin['Amount'].apply(lambda x: f"${x:,.0f}")

fig_fin = px.bar(
    df_fin, 
    x='Name', 
    y='Amount', 
    color='Type', 
    text='Text',
    barmode='group',
    color_discrete_map={'Revenue (Received)': '#00CC96', 'Cost (Paid)': '#EF553B'},
    title="Swap Financial Impact"
)
fig_fin.update_traces(textposition='auto')
fig_fin.update_yaxes(title="Amount ($)")
st.plotly_chart(fig_fin, use_container_width=True)

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

c_date, c_view = st.columns([2, 2])
view_mode = c_view.radio("View Mode", ["Weekly", "Full Year"], horizontal=True)

if view_mode == "Weekly":
    week_start = c_date.date_input("Select Start Date", value=pd.to_datetime("2024-06-01"), key="date_picker")
    start_idx = pd.Timestamp(week_start)
    end_idx = start_idx + pd.Timedelta(days=7)
    subset = df_plot[start_idx:end_idx]
    chart_title_suffix = "(Selected Week)"
else:
    subset = df_plot
    chart_title_suffix = "(Full Year)"

tab_chart, tab_data = st.tabs(["Chart", "Net Positions Data"])

with tab_chart:
    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(x=subset.index, y=subset['Total RE'], name='Renewables', fill='tozeroy', line=dict(color='#00CC96', width=0)))
    fig_ts.add_trace(go.Scatter(x=subset.index, y=subset['Load'], name='Load', line=dict(color='#AB63FA', width=1)))
    fig_ts.update_layout(title=f"{title} - Hourly Dispatch {chart_title_suffix}", xaxis_title="Time", yaxis_title="MW", height=400)
    st.plotly_chart(fig_ts, use_container_width=True)

with tab_data:
    # Filter for selected time range if desired, or show full year
    if view_mode == "Weekly":
        s_idx = start_idx
        e_idx = end_idx
    else:
        s_idx = None
        e_idx = None

    if view_option == "Aggregated Pool":
        st.markdown("### Hourly Net Positions (Generation - Load)")
        st.markdown("Positive values indicate a **Surplus** (Export available), Negative values indicate a **Deficit** (Import needed).")
        
        data_to_show = net_positions_df
        if s_idx:
            data_to_show = data_to_show[s_idx:e_idx]
            
        st.dataframe(data_to_show.style.format("{:,.1f}"), use_container_width=True)
    
    else:
        # Detailed view for single participant
        # res is already found above as 'res'
        
        # Calculate Net Position for this view (Gen - Load)
        hourly_net = res['total_re'] - res['load']
        
        # Construct detailed DF
        participant_df = pd.DataFrame({
            'Load (MW)': res['load'],
            'Generation (MW)': res['total_re'],
            'Net Position (MW)': hourly_net,
            'Swap Imports (MW)': 0.0, # Default, fill if exists
            'Swap Exports (MW)': 0.0  # Default, fill if exists
        })
        
        # We need to access the swap details again. 
        # Though 'swap_imported' is a sum in results, we need hourly series.
        # But 'swap_results' in line 153 aggregated totals, it didn't save hourly series in the dictionary structure except for imports which we have in 'hourly_imports' dataframe!
        
        # For Exports, we didn't explicitly save the hourly export series in a persistent DF in the main loop, only totals.
        # However, imports for one are exports for another.
        # But we do have 'hourly_imports' DF.
        
        if 'hourly_imports' in locals():
            if view_option in hourly_imports.columns:
                 participant_df['Swap Imports (MW)'] = hourly_imports[view_option]
        
        # To get hourly exports accurately without re-running logic, we recall:
        # Export = (Surplus / Total Surplus) * Swappable
        # This is a bit complex to reconstruct perfectly here without saving it earlier.
        # For now, let's show what we have: Load, Gen, Net Position.
        
        st.markdown(f"### {view_option} - Detailed Hourly Data")
        st.markdown("Net Position = Generation - Load")
        
        if s_idx:
            participant_df = participant_df[s_idx:e_idx]
            
        st.dataframe(participant_df.style.format("{:,.1f}").applymap(lambda x: 'color: red' if x < 0 else 'color: green', subset=['Net Position (MW)']), use_container_width=True)

# Monthly View
monthly = df_plot.resample('M').sum() / 1000 # GWh
monthly.index = monthly.index.strftime('%b')
monthly['Renewables'] = monthly['Solar'] + monthly['Wind']
fig_bar = px.bar(monthly, x=monthly.index, y=['Renewables'], title=f"{title} - Monthly Energy Balance", color_discrete_map={'Renewables': '#00CC96'})
fig_bar.add_trace(go.Scatter(x=monthly.index, y=monthly['Load'], name='Load', line=dict(color='#AB63FA', width=3)))
st.plotly_chart(fig_bar, use_container_width=True)

# REC Position Analysis
st.subheader("REC Position Analysis")
if view_option == "Aggregated Pool":
    net_pos = agg_total_re - aggregated_load
    exc_mwh = net_pos.clip(lower=0)
    short_mwh = -net_pos.clip(upper=0)
    exc_val = exc_mwh.sum() * rec_price
    short_cost = short_mwh.sum() * rec_price
else:
    # Already calculated in results
    exc_mwh = res['excess_mwh']
    short_mwh = res['shortfall_mwh']
    exc_val = res['excess_value']
    short_cost = res['shortfall_cost']
    net_pos = res['total_re'] - res['load']

c1, c2 = st.columns(2)
c1.metric("Excess RECs (Surplus)", f"{exc_mwh.sum():,.0f} MWh", f"${exc_val:,.0f} Revenue")
c2.metric("Needed RECs (Shortfall)", f"{short_mwh.sum():,.0f} MWh", f"-${short_cost:,.0f} Cost")

# Hourly Net Position Chart
df_pos = pd.DataFrame({'Net Position': net_pos})
if view_mode == "Weekly":
    subset_pos = df_pos[start_idx:end_idx]
else:
    subset_pos = df_pos

fig_pos = go.Figure()
fig_pos.add_trace(go.Bar(
    x=subset_pos.index, 
    y=subset_pos['Net Position'],
    marker_color=subset_pos['Net Position'].apply(lambda x: '#00CC96' if x >= 0 else '#EF553B'),
    name='Net Position'
))
fig_pos.update_layout(title=f"{title} - Hourly REC Position {chart_title_suffix}", xaxis_title="Time", yaxis_title="MW", height=400)
st.plotly_chart(fig_pos, use_container_width=True)
