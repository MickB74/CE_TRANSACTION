# Aggregated Procurement Tool (ERCOT)

A Streamlit application designed to simulate and analyze the aggregated procurement of firm power in the ERCOT market. This tool demonstrates the benefits of pooling diverse load profiles and renewable energy assets to maximize Carbon Free Energy (CFE) matching and optimize financial performance.

## Overview

The application simulates a scenario where multiple entities (e.g., Data Centers, Hospitals, Retailers) pool their electricity demand and renewable energy Power Purchase Agreements (PPAs). It calculates individual and aggregated performance metrics, visualizing how acting as a single pool can smooth out variability and improve matching.

It also includes an internal specific mechanism to "swap" renewable attributes (RECs) between participants who have excess generation and those who have a deficit, further optimizing individual scores.

## Key Features

- **Dynamic Participant Management**:
    - Add or remove participants.
    - Customize load profiles (Business, Industrial, Residential, Data Center, Health Care, Retail).
    - Assign specific annual Solar and Wind generation volumes.

- **Detailed Simulation**:
    - Generates synthetic hourly load profiles based on typical behaviors and seasonality.
    - Generates hourly Solar and Wind generation profiles based on physical models and randomized weather patterns (Cloud cover, Wind speed).
    - Runs a full-year hourly simulation (8,760 hours).

- **Performance Metrics**:
    - **Volumetric RE %**: Total Renewable Generation / Total Load.
    - **Standalone CFE %**: Percent of load matched by renewables in real-time (hourly).
    - **Optimized CFE %**: CFE score after accounting for internal REC swaps.
    - **Financials**: Estimates REC revenues, shortage costs, and net settlement values.

- **Interactive Visualizations**:
    - **Hourly Dispatch**: Zoomable time-series charts showing Load vs. Solar/Wind generation.
    - **Monthly Balance**: Aggregated monthly views of energy and load.
    - **Net Position**: specific analysis of hourly long/short positions.
    - **Comparison Charts**: Side-by-side comparison of Individual vs. Pooled Match percentages.

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/MickB74/CE_TRANSACTION.git
    cd CE_TRANSACTION
    ```

2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your default web browser (usually at `http://localhost:8501`).

## Project Structure

-   `app.py`: The main entry point for the Streamlit application. Contains the UI layout, simulation loop, aggregation logic, and visualization code.
-   `utils.py`: Contains utility functions and classes:
    -   `generate_load_profile`: Creates hourly load shapes for different sectors.
    -   `generate_solar_profile`: Simulates solar irradiance and generation.
    -   `generate_wind_profile`: Simulates wind speeds and power curves.
    -   `BatteryStorage`: A class for modeling battery dispatch (currently available in utils but not fully exposed in the main UI).
-   `requirements.txt`: List of Python packages required to run the tool.

## Methodology

-   **Load Profiles**: Generated using normalized daily shapes with added seasonality and random noise.
-   **Solar Model**: Based on solar variations (elevation/azimuth) and randomized cloud cover factors.
-   **Wind Model**: Based on a Weibull wind speed distribution and a standard power curve.
-   **Optimization**: The internal swap logic uses a pro-rata allocation method. Participants with excess renewables "export" to the pool, and those with deficits "import" from the pool, constrained by the total available surplus/deficit in that hour.
