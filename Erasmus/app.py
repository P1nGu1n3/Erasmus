import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --- CONFIGURARE PAGINĂ ---
st.set_page_config(page_title="Industry 5.0 Digital Twin", page_icon="🏭", layout="wide")
st.title("🏭 Industry 5.0: Unified Stochastic Framework")
st.markdown("### Erasmus+ 2026 | Shumen Regional Hub Simulation")

# --- SIDEBAR (Global Parameters) ---
st.sidebar.header("⚙️ Global Simulation Settings")
n_sim = st.sidebar.slider("Number of Monte Carlo Iterations", min_value=1000, max_value=100000, value=10000, step=1000)
random_seed = st.sidebar.number_input("Random Seed (for reproducibility)", value=2026)
np.random.seed(random_seed)

st.sidebar.markdown("---")
st.sidebar.info("This dashboard simulates Optimization, Logistics, and Reliability under uncertainty.")

# --- TABS ---
tab1, tab2, tab3, tab4 = st.tabs(
    ["⚡ Pillar 1: Energy", "📦 Pillar 2: Logistics", "⚙️ Pillar 3: Reliability", "📊 Unified Dashboard"])

# ==========================================
# TAB 1: PILLAR 1 - ENERGY OPTIMIZATION
# ==========================================
with tab1:
    st.header("⚡ Pillar 1: Optimization under Uncertainty")
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Control Variables")
        speed_setpoint = st.slider("Machine Speed (m/s)", 1.5, 3.0, 2.5, 0.1)
        grid_limit = st.number_input("Grid Overload Limit (kWh)", value=100.0)
        run_p1 = st.button("Run Energy Simulation", type="primary")

    if run_p1:
        with col2:
            # Stochastic variables
            hardness = np.random.triangular(0.8, 1.0, 1.5, size=n_sim)
            energy_price = np.random.lognormal(mean=np.log(0.15), sigma=0.05, size=n_sim)

            # Equation: E = 12.5 * v^2 * mu
            energy_consumed = 12.5 * (speed_setpoint ** 2) * hardness
            total_cost = energy_consumed * energy_price

            # Metrics
            risk_overload = np.mean(energy_consumed > grid_limit) * 100
            expected_cost = np.mean(total_cost)

            # Layout Metrics
            m1, m2 = st.columns(2)
            m1.metric("Expected Total Cost (€)", f"€{expected_cost:.2f}")
            m2.metric("Risk of Grid Overload (%)", f"{risk_overload:.1f}%",
                      delta_color="inverse", delta=f"{risk_overload - 2.0:.1f}% vs Target")

            # Plotly Histogram
            fig = px.histogram(x=total_cost, nbins=50, title="Probability Density of Total Energy Cost",
                               labels={'x': 'Total Cost (€)', 'y': 'Frequency'},
                               color_discrete_sequence=['#1f77b4'])
            fig.add_vline(x=expected_cost, line_dash="dash", line_color="red", annotation_text="Expected Cost")
            st.plotly_chart(fig, use_container_width=True)

# ==========================================
# TAB 2: PILLAR 2 - LOGISTICS (SHUMEN HUB)
# ==========================================
with tab2:
    st.header("📦 Pillar 2: Regional Logistics Risk")
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Inventory Policy")
        safety_stock = st.slider("Safety Stock Level (Units)", 100, 500, 200, 10)
        daily_demand_mean = st.number_input("Average Daily Demand", value=50)
        run_p2 = st.button("Run Logistics Simulation", type="primary")

    if run_p2:
        with col2:
            # Stochastic variables
            daily_demand = np.random.poisson(lam=daily_demand_mean, size=n_sim)
            lead_time_days = np.random.lognormal(mean=np.log(3), sigma=0.4, size=n_sim)
            regional_shock = np.random.choice([0, 2], p=[0.95, 0.05], size=n_sim)

            # Logic
            total_lead_time = lead_time_days + regional_shock
            demand_during_lt = daily_demand * total_lead_time
            stockout_volume = np.maximum(0, demand_during_lt - safety_stock)
            financial_loss = stockout_volume * 150  # €150 per missing unit

            # Metrics
            var_95 = np.percentile(financial_loss, 95)
            service_level = np.mean(stockout_volume == 0) * 100

            m1, m2 = st.columns(2)
            m1.metric("Service Level (%)", f"{service_level:.1f}%")
            m2.metric("95% Value at Risk (VaR)", f"€{var_95:,.2f}")

            # Plotly CDF
            fig = px.ecdf(x=financial_loss, title="Cumulative Distribution of Financial Loss (Stockouts)",
                          labels={'x': 'Financial Loss (€)', 'y': 'Cumulative Probability'})
            fig.add_vline(x=var_95, line_dash="dash", line_color="red", annotation_text="95% VaR")
            st.plotly_chart(fig, use_container_width=True)

# ==========================================
# TAB 3: PILLAR 3 - RELIABILITY (ROBOTIC ARM)
# ==========================================
with tab3:
    st.header("⚙️ Pillar 3: Predictive Maintenance")
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Weibull Parameters")
        beta = st.slider("Shape Parameter (β) - Wear out", 1.0, 5.0, 3.5, 0.1)
        eta = st.number_input("Characteristic Life (η) hours", value=10000)
        run_p3 = st.button("Run Reliability Simulation", type="primary")

    if run_p3:
        with col2:
            # Stochastic Logic
            ttf = eta * np.random.weibull(beta, n_sim)

            # Metrics
            pm_trigger = np.percentile(ttf, 10)  # 10% failure mark
            early_fail_prob = np.mean(ttf < 9500) * 100

            m1, m2 = st.columns(2)
            m1.metric("Recommended Maintenance Window", f"{pm_trigger:.0f} hrs")
            m2.metric("Risk of Early Failure (<9500h)", f"{early_fail_prob:.1f}%")

            # Plotly Survival Curve
            fig = px.histogram(x=ttf, nbins=100, title="Time-To-Failure (TTF) Distribution",
                               labels={'x': 'Hours to Failure', 'y': 'Component Count'},
                               color_discrete_sequence=['#2ca02c'])
            fig.add_vline(x=pm_trigger, line_dash="dash", line_color="red", annotation_text="Inspect Here")
            st.plotly_chart(fig, use_container_width=True)

# ==========================================
# TAB 4: UNIFIED DASHBOARD
# ==========================================
with tab4:
    st.header("📊 Unified System Health Dashboard")
    st.info(
        "Run the simulations in the previous tabs to update global metrics (In a full app, this syncs automatically using st.session_state).")

    # Static mockup for the presentation visual effect
    st.markdown("### 🌐 Real-Time Industry 5.0 Global Status")

    colA, colB, colC = st.columns(3)

    figA = go.Figure(go.Indicator(
        mode="gauge+number",
        value=98.8,  # Mock value for Target speed 2.2
        title={'text': "Energy Safety Score"},
        gauge={'axis': {'range': [None, 100]}, 'bar': {'color': "green"}}
    ))
    colA.plotly_chart(figA, use_container_width=True)

    figB = go.Figure(go.Indicator(
        mode="gauge+number",
        value=88.0,  # Mock service level
        title={'text': "Logistics Service Level"},
        gauge={'axis': {'range': [None, 100]}, 'bar': {'color': "orange"}}
    ))
    colB.plotly_chart(figB, use_container_width=True)

    figC = go.Figure(go.Indicator(
        mode="gauge+number",
        value=95.0,  # Mock reliability
        title={'text': "Equipment Reliability"},
        gauge={'axis': {'range': [None, 100]}, 'bar': {'color': "green"}}
    ))
    colC.plotly_chart(figC, use_container_width=True)