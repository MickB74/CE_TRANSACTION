import numpy as np
import pandas as pd
import streamlit as st

def generate_hourly_index(year=2024):
    """Generates a datetime index for a given year."""
    start_date = f'{year}-01-01'
    end_date = f'{year}-12-31 23:00'
    return pd.date_range(start=start_date, end=end_date, freq='H')



@st.cache_data
def generate_load_profile(annual_consumption_mwh, profile_type='Business', year=2024):
    """
    Generates an hourly load profile based on annual consumption and a profile type.
    
    Args:
        annual_consumption_mwh (float): Total annual consumption in MWh.
        profile_type (str): 'Business', 'Industrial', 'Residential', 'Data Center', 'Health Care', 'Retail'.
        year (int): Year for the profile.
        
    Returns:
        pd.Series: Hourly load in MW.
    """
    index = generate_hourly_index(year)
    hours = index.hour
    day_of_week = index.dayofweek
    
    # Base shape
    if profile_type == 'Business':
        # Peak 9am-5pm, low weekends
        daily_shape = np.where((hours >= 9) & (hours <= 17), 1.0, 0.2)
        weekend_factor = np.where(day_of_week >= 5, 0.3, 1.0)
        shape = daily_shape * weekend_factor
    elif profile_type == 'Industrial':
        # Flat load with minor fluctuations
        shape = np.ones(len(index)) + np.random.normal(0, 0.05, len(index))
    elif profile_type == 'Residential':
        # Morning and Evening peaks
        daily_shape = np.where((hours >= 7) & (hours <= 9), 0.8, 
                               np.where((hours >= 17) & (hours <= 21), 1.0, 0.3))
        shape = daily_shape
    elif profile_type == 'Data Center':
        # Extremely flat, high load factor
        shape = np.ones(len(index)) + np.random.normal(0, 0.02, len(index))
    elif profile_type == 'Health Care':
        # 24/7 operation but higher during day
        daily_shape = np.where((hours >= 7) & (hours <= 19), 1.0, 0.7)
        shape = daily_shape
    elif profile_type == 'Retail':
        # Broader peak than business (e.g. 9am - 9pm)
        daily_shape = np.where((hours >= 9) & (hours <= 21), 1.0, 0.1)
        weekend_factor = np.where(day_of_week >= 5, 1.2, 1.0) # Higher on weekends
        shape = daily_shape * weekend_factor
    else:
        shape = np.ones(len(index))

    # Add some random noise and seasonality (higher in summer/winter)
    day_of_year = index.dayofyear.to_numpy()
    
    if profile_type == 'Data Center':
        # Very low seasonality and noise for Data Centers
        seasonality = 1 + 0.02 * np.sin(2 * np.pi * (day_of_year - 150) / 365)
        noise_level = 0.01
    else:
        seasonality = 1 + 0.2 * np.sin(2 * np.pi * (day_of_year - 150) / 365) # Peak in summer
        noise_level = 0.1
    
    final_shape = shape * seasonality * (1 + np.random.normal(0, noise_level, len(index)))
    final_shape = np.maximum(final_shape, 0) # Ensure non-negative
    
    # Normalize to match annual consumption
    total_shape_sum = final_shape.sum()
    scaling_factor = annual_consumption_mwh / total_shape_sum
    
    return pd.Series(final_shape * scaling_factor, index=index, name='Load_MW')



@st.cache_data
def generate_solar_profile(capacity_mw=None, annual_mwh=None, year=2024):
    """Generates a synthetic solar profile. Can specify either capacity or annual MWh."""
    index = generate_hourly_index(year)
    hours = index.hour.to_numpy()
    day_of_year = index.dayofyear.to_numpy()
    
    # Solar elevation proxy
    declination = 23.45 * np.sin(2 * np.pi * (284 + day_of_year) / 365)
    hour_angle = 15 * (hours - 12)
    elevation = np.maximum(0, np.sin(np.radians(declination)) * np.sin(np.radians(30)) + \
                np.cos(np.radians(declination)) * np.cos(np.radians(30)) * np.cos(np.radians(hour_angle)))
    
    irradiance = elevation * 1000 # W/m2 approx
    
    # Add cloud cover noise
    cloud_cover = np.random.beta(2, 5, len(index)) 
    final_output = irradiance * (1 - cloud_cover * 0.8)
    
    # Determine generation profile
    if annual_mwh is not None:
        # Normalize to sum to annual_mwh
        total_raw = np.sum(final_output)
        if total_raw > 0:
            generation = (final_output / total_raw) * annual_mwh
        else:
            generation = np.zeros(len(index))
    elif capacity_mw is not None:
        # Normalize to capacity (assuming 1000 W/m2 is STC)
        generation = (final_output / 1000) * capacity_mw
    else:
        generation = np.zeros(len(index))
        
    generation = np.maximum(generation, 0)
    
    return pd.Series(generation, index=index, name='Solar_MW')



@st.cache_data
def generate_wind_profile(capacity_mw=None, annual_mwh=None, year=2024):
    """Generates a synthetic wind profile. Can specify either capacity or annual MWh."""
    index = generate_hourly_index(year)
    
    # Weibull distribution for wind speed
    wind_speed = np.random.weibull(2, len(index)) * 7 # Scale factor
    
    # Power curve approximation (cut-in 3, rated 12, cut-out 25)
    generation = np.zeros(len(index))
    
    # Use a temporary capacity of 1.0 to get the shape
    temp_capacity = 1.0
    
    mask_ramp = (wind_speed >= 3) & (wind_speed < 12)
    generation[mask_ramp] = ((wind_speed[mask_ramp] - 3) / 9) ** 3 * temp_capacity
    
    mask_rated = (wind_speed >= 12) & (wind_speed < 25)
    generation[mask_rated] = temp_capacity
    
    # Add some autocorrelation to make it look like wind
    # Simple smoothing
    series = pd.Series(generation, index=index)
    smoothed = series.rolling(window=3, center=True).mean().fillna(0)
    
    if annual_mwh is not None:
        # Normalize to sum to annual_mwh
        total_raw = smoothed.sum()
        final_generation = (smoothed / total_raw) * annual_mwh
    elif capacity_mw is not None:
        # Scale by capacity
        final_generation = smoothed * capacity_mw
    else:
        final_generation = np.zeros(len(index))
    
    return final_generation.rename('Wind_MW')

class BatteryStorage:
    def __init__(self, power_mw, energy_mwh, efficiency=0.9):
        self.power_mw = power_mw
        self.energy_mwh = energy_mwh
        self.efficiency = efficiency
        self.soc = 0 # State of Charge in MWh
        
    def dispatch(self, net_load):
        """
        Simple dispatch: 
        - If Net Load > 0 (Deficit), Discharge.
        - If Net Load < 0 (Surplus), Charge.
        """
        dispatch_mw = []
        soc_profile = []
        
        current_soc = self.energy_mwh * 0.5 # Start at 50%
        
        for load in net_load:
            action = 0 # + is discharge, - is charge
            
            if load > 0: # Deficit, try to discharge
                max_discharge = min(self.power_mw, current_soc)
                action = min(load, max_discharge)
                current_soc -= action
            elif load < 0: # Surplus, try to charge
                surplus = -load
                max_charge = min(self.power_mw, (self.energy_mwh - current_soc) / self.efficiency)
                charge_amount = min(surplus, max_charge)
                action = -charge_amount
                current_soc += charge_amount * self.efficiency
            
            dispatch_mw.append(action)
            soc_profile.append(current_soc)
            
        return pd.Series(dispatch_mw, index=net_load.index, name='Battery_Dispatch_MW'), \
               pd.Series(soc_profile, index=net_load.index, name='Battery_SOC_MWh')

def calculate_swap_value(hourly_data, market_rec_price=5.00, manager_fee_pct=0.20):
    """
    Calculates the financial value of the internal REC swaps.
    
    Parameters:
    - hourly_data (pd.DataFrame): DataFrame with columns for each participant's hourly Net Position 
                                  (Load - Generation). 
                                  Positive = Deficit (Need Power), Negative = Surplus (Exporting).
    - market_rec_price (float): The cost ($/MWh) to buy a REC on the open market.
    - manager_fee_pct (float): The percentage of the savings you keep as revenue.
    
    Returns:
    - dict: Contains total savings, volume swapped, and your potential revenue.
    """
    
    # 1. Identify Longs (Surplus) and Shorts (Deficit) for every hour
    # 'clip(lower=0)' gives us only the deficits (Shorts)
    # 'clip(upper=0).abs()' gives us only the surpluses (Longs)
    total_market_deficits = hourly_data.clip(lower=0).sum(axis=1)
    total_market_surpluses = hourly_data.clip(upper=0).abs().sum(axis=1)
    
    # 2. Calculate the "Swappable Volume"
    # In any given hour, the amount we can swap is limited by the smaller of the two sides.
    # If the group needs 100 MW but only has 20 MW surplus, we swap 20 MW.
    # If the group has 100 MW surplus but only needs 20 MW, we swap 20 MW.
    hourly_swapped_volume = np.minimum(total_market_deficits, total_market_surpluses)
    
    # 3. Calculate Financials
    total_swapped_mwh = hourly_swapped_volume.sum()
    
    # Value Created = (Volume Swapped) * (Price they would have paid on the market)
    total_value_created = total_swapped_mwh * market_rec_price
    
    # Your Revenue
    manager_revenue = total_value_created * manager_fee_pct
    
    # 4. Formatted Summary
    results = {
        "Total Volume Swapped (MWh)": round(total_swapped_mwh, 2),
        "Market Value of Swaps ($)": round(total_value_created, 2),
        "Projected Manager Revenue ($)": round(manager_revenue, 2),
        "Client Net Savings ($)": round(total_value_created - manager_revenue, 2)
    }
    
    return results, hourly_swapped_volume
