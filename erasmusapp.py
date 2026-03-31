import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

# Configurare pagină
st.set_page_config(
    page_title="Industry 5.0 Stochastic Simulator",
    page_icon="🏭",
    layout="wide"
)

st.title("🏭 Industry 5.0 Unified Stochastic Framework")
st.markdown("**Erasmus+ Project 2026 | Monte Carlo Simulation for Optimization, Logistics & Reliability**")

# Inițializare session state
if 'results' not in st.session_state:
    st.session_state.results = {
        'p1': None, 'p2': None, 'p3': None,
        'p1_risk': None, 'p2_service': None, 'p3_reliability': None
    }

# SIDEBAR - Setări globale
st.sidebar.header("⚙️ Global Simulation Settings")
n_sim = st.sidebar.slider(
    "Monte Carlo Iterations",
    min_value=1000, max_value=100000, value=10000, step=1000,
    help="More iterations improve precision (Law of Large Numbers) but increase computation time."
)
random_seed = st.sidebar.number_input("Random Seed (reproducibility)", value=2026,
                                      help="Fix seed for exact reproduction of results in the paper.")
np.random.seed(random_seed)

col_btn1, col_btn2 = st.sidebar.columns(2)
with col_btn1:
    if st.button("🚀 Run All Pillars", type="primary"):
        st.session_state.results = {k: None for k in st.session_state.results}
        st.rerun()

with col_btn2:
    if st.button("🔄 Reset All", type="secondary"):
        st.session_state.results = {k: None for k in st.session_state.results}
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.info("Run individual pillars or all at once to update the Unified Dashboard.")

# Tabs (adăugat Convergence Check)
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["⚡ Pillar 1: Energy", "📦 Pillar 2: Logistics", "⚙️ Pillar 3: Reliability", "📊 Unified Dashboard", "🔍 Convergence Check"]
)

# ==========================================
# TAB 1: PILLAR 1 - ENERGY OPTIMIZATION
# ==========================================
with tab1:
    st.header("⚡ Pillar 1: Optimization under Uncertainty")
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Control Variables")
        speed_setpoint = st.slider("Machine Speed (m/s)", 1.5, 3.0, 2.5, 0.1,
                                   help="Conveyor / equipment speed setpoint.")
        grid_limit = st.number_input("Grid Overload Limit (kWh)", value=100.0,
                                     help="Power grid threshold before overload/trip.")
        run_p1 = st.button("Run Pillar 1", type="primary", key="run_p1")

    if run_p1 or st.session_state.results['p1'] is None:
        with st.spinner("Running Pillar 1 Monte Carlo simulation..."):
            hardness = np.random.triangular(0.8, 1.0, 1.5, size=n_sim)
            energy_price = np.random.lognormal(mean=np.log(0.15), sigma=0.05, size=n_sim)
            energy_consumed = 12.5 * (speed_setpoint ** 2) * hardness
            total_cost = energy_consumed * energy_price

            risk_overload = np.mean(energy_consumed > grid_limit) * 100
            expected_cost = np.mean(total_cost)

            st.session_state.results['p1'] = {
                'cost': total_cost, 'risk': risk_overload, 'exp_cost': expected_cost,
                'hardness': hardness, 'price': energy_price
            }
            st.session_state.results['p1_risk'] = 100 - risk_overload

        st.success("Pillar 1 completed!")

    if st.session_state.results['p1']:
        with col2:
            r = st.session_state.results['p1']
            m1, m2 = st.columns(2)
            m1.metric("Expected Total Cost (€)", f"€{r['exp_cost']:.2f}",
                      help="Mean total cost across all simulations")
            if r['risk'] > 10:
                m2.error(f"⚠️ Overload Risk: {r['risk']:.1f}%")
            else:
                m2.success(f"✅ Overload Risk: {r['risk']:.1f}%")

            fig = px.histogram(x=r['cost'], nbins=50,
                               title="Probability Density of Total Energy Cost",
                               labels={'x': 'Total Cost (€)', 'y': 'Frequency'},
                               color_discrete_sequence=['#1f77b4'])
            fig.add_vline(x=r['exp_cost'], line_dash="dash", line_color="red",
                          annotation_text="Expected Cost")
            st.plotly_chart(fig, use_container_width=True)

            # 3D Scatter
            st.markdown("#### 🌐 3D Stochastic Risk Map")
            sample_size = min(n_sim, 500)
            fig_3d = px.scatter_3d(
                x=r['hardness'][:sample_size], y=r['price'][:sample_size], z=r['cost'][:sample_size],
                color=r['cost'][:sample_size], color_continuous_scale='Turbo',
                labels={'x': 'Material Hardness', 'y': 'Energy Price (€)', 'z': 'Total Cost (€)'}
            )
            fig_3d.update_traces(marker=dict(size=4, opacity=0.8))
            fig_3d.update_layout(margin=dict(l=0, r=0, b=0, t=0), height=500)
            st.plotly_chart(fig_3d, use_container_width=True)

            # Export CSV
            df_p1 = pd.DataFrame({'Hardness': r['hardness'], 'Price': r['price'], 'Total_Cost': r['cost']})
            st.download_button("📥 Export Energy Data (CSV)", df_p1.to_csv(index=False),
                               file_name="pillar1_energy.csv", mime="text/csv")

# ==========================================
# TAB 2: PILLAR 2 - LOGISTICS
# ==========================================
with tab2:
    st.header("📦 Pillar 2: Regional Logistics Risk (Shumen Hub)")
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Inventory Policy")
        safety_stock = st.slider("Safety Stock Level (Units)", 100, 500, 200, 10,
                                 help="Buffer stock to prevent production stoppages.")
        daily_demand_mean = st.number_input("Average Daily Demand (units)", value=50)
        run_p2 = st.button("Run Pillar 2", type="primary", key="run_p2")

    if run_p2 or st.session_state.results['p2'] is None:
        with st.spinner("Running Pillar 2 Monte Carlo simulation..."):
            daily_demand = np.random.poisson(lam=daily_demand_mean, size=n_sim)
            lead_time_days = np.random.lognormal(mean=np.log(3), sigma=0.4, size=n_sim)
            regional_shock = np.random.choice([0, 2], p=[0.95, 0.05], size=n_sim)

            total_lead_time = lead_time_days + regional_shock
            demand_during_lt = daily_demand * total_lead_time
            stockout_volume = np.maximum(0, demand_during_lt - safety_stock)
            financial_loss = stockout_volume * 150

            var_95 = np.percentile(financial_loss, 95)
            service_level = np.mean(stockout_volume == 0) * 100

            st.session_state.results['p2'] = {
                'loss': financial_loss, 'var95': var_95, 'service': service_level
            }
            st.session_state.results['p2_service'] = service_level

        st.success("Pillar 2 completed!")

    if st.session_state.results['p2']:
        with col2:
            r = st.session_state.results['p2']
            m1, m2 = st.columns(2)
            if r['service'] >= 95:
                m1.success(f"✅ Service Level: {r['service']:.1f}%")
            else:
                m1.warning(f"⚠️ Service Level: {r['service']:.1f}% (Low)")

            m2.metric("95% Value at Risk (VaR)", f"€{r['var95']:,.2f}",
                      help="Worst-case financial loss at 95% confidence level")

            fig = px.ecdf(x=r['loss'], title="Cumulative Distribution of Financial Loss (Stockouts)",
                          labels={'x': 'Financial Loss (€)', 'y': 'Cumulative Probability'})
            fig.add_vline(x=r['var95'], line_dash="dash", line_color="red", annotation_text="95% VaR")
            st.plotly_chart(fig, use_container_width=True)

            df_p2 = pd.DataFrame({'Financial Loss (€)': r['loss']})
            st.download_button("📥 Export Loss Data (CSV)", df_p2.to_csv(index=False),
                               file_name="pillar2_financial_loss.csv", mime="text/csv")

# ==========================================
# TAB 3: PILLAR 3 - RELIABILITY
# ==========================================
with tab3:
    st.header("⚙️ Pillar 3: Predictive Maintenance")
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Weibull Parameters")
        beta = st.slider("Shape Parameter (β)", 1.0, 5.0, 3.5, 0.1,
                         help="β > 1 indicates wear-out phase.")
        eta = st.number_input("Characteristic Life (η hours)", value=10000,
                              help="Time at which 63.2% of components fail.")
        run_p3 = st.button("Run Pillar 3", type="primary", key="run_p3")

    if run_p3 or st.session_state.results['p3'] is None:
        with st.spinner("Running Pillar 3 Monte Carlo simulation..."):
            ttf = stats.weibull_min.rvs(c=beta, scale=eta, size=n_sim)
            pm_trigger = np.percentile(ttf, 10)
            early_fail_prob = np.mean(ttf < 9500) * 100

            st.session_state.results['p3'] = {
                'ttf': ttf, 'pm_trigger': pm_trigger, 'early_fail': early_fail_prob
            }
            st.session_state.results['p3_reliability'] = 100 - early_fail_prob

        st.success("Pillar 3 completed!")

    if st.session_state.results['p3']:
        with col2:
            r = st.session_state.results['p3']
            m1, m2 = st.columns(2)
            m1.metric("Recommended Maintenance Window", f"{r['pm_trigger']:.0f} hrs",
                      help="10th percentile – early warning point")
            m2.metric("Risk of Early Failure (<9500h)", f"{r['early_fail']:.1f}%")

            fig = px.histogram(x=r['ttf'], nbins=100, title="Time-To-Failure (TTF) Distribution",
                               labels={'x': 'Hours to Failure', 'y': 'Component Count'},
                               color_discrete_sequence=['#2ca02c'])
            fig.add_vline(x=r['pm_trigger'], line_dash="dash", line_color="red",
                          annotation_text="Inspect Here")
            st.plotly_chart(fig, use_container_width=True)

            df_p3 = pd.DataFrame({'Time to Failure (hours)': r['ttf']})
            st.download_button("📥 Export TTF Data (CSV)", df_p3.to_csv(index=False),
                               file_name="pillar3_ttf.csv", mime="text/csv")

# ==========================================
# TAB 4: UNIFIED DASHBOARD
# ==========================================
with tab4:
    st.header("📊 Unified Industry 5.0 Health Dashboard")
    st.info("Aggregated resilience scores from the three engineering pillars.")

    if all(v is not None for v in
           [st.session_state.results.get(k) for k in ['p1_risk', 'p2_service', 'p3_reliability']]):
        p1_score = st.session_state.results['p1_risk']
        p2_score = st.session_state.results['p2_service']
        p3_score = st.session_state.results['p3_reliability']

        health_score = (p1_score * 0.4 + p2_score * 0.3 + p3_score * 0.3)

        colA, colB, colC, colD = st.columns(4)
        colA.metric("Energy Safety Score", f"{p1_score:.1f}%")
        colB.metric("Logistics Service Level", f"{p2_score:.1f}%")
        colC.metric("Equipment Reliability", f"{p3_score:.1f}%")
        colD.metric("Overall Health Score", f"{health_score:.1f}%",
                    delta="Optimal" if health_score >= 90 else "Requires Attention",
                    delta_color="normal" if health_score >= 90 else "inverse")

        # Alerte condiționate
        if health_score < 80:
            st.error("⚠️ Critical System Health – Immediate Action Recommended!")
        elif health_score < 90:
            st.warning("⚠️ Moderate System Health – Review Parameters")
        else:
            st.success("✅ Optimal System Health")

        # Gauge dinamic
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=health_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Global Resilience Index"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#2ca02c" if health_score >= 90 else "#ffa500" if health_score >= 80 else "#ff4b4b"},
                'steps': [
                    {'range': [0, 75], 'color': "#ff4b4b"},
                    {'range': [75, 90], 'color': "#ffa500"},
                    {'range': [90, 100], 'color': "#2ca02c"}
                ]
            }
        ))
        st.plotly_chart(fig, use_container_width=True)

        # Export raport
        report = f"""Industry 5.0 Monte Carlo Framework Report
Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
Simulations: {n_sim}

Pillar 1 Energy Safety: {p1_score:.1f}%
Pillar 2 Logistics Service: {p2_score:.1f}%
Pillar 3 Reliability: {p3_score:.1f}%
Overall Health: {health_score:.1f}%
"""
        st.download_button("📄 Download Full Report (TXT)", report, "system_health_report.txt")

    else:
        st.warning("Run all pillar simulations to generate the global health score.")

# ==========================================
# TAB 5: CONVERGENCE CHECK
# ==========================================
with tab5:
    st.header("🔍 Convergence Analysis")
    st.info("Shows how increasing N reduces estimation error (example from Pillar 1 energy model)")

    if st.button("Run Convergence Test"):
        with st.spinner("Computing convergence..."):
            ns = [1000, 5000, 10000, 20000, 50000, 100000]
            means = []
            errors = []
            true_mean = 12.5 * (2.2 ** 2) * 1.0  # expected mean hardness = 1.0

            for n in ns:
                np.random.seed(2026)
                hardness = np.random.triangular(0.8, 1.0, 1.5, size=n)
                energy = 12.5 * (2.2 ** 2) * hardness
                mean_energy = np.mean(energy)
                rel_error = abs(mean_energy - true_mean) / true_mean * 100
                means.append(mean_energy)
                errors.append(rel_error)

            df_conv = pd.DataFrame({'N': ns, 'Mean Energy (kWh)': means, 'Relative Error (%)': errors})

            fig_line = px.line(df_conv, x='N', y='Relative Error (%)',
                               title="Relative Error vs Number of Simulations",
                               markers=True)
            st.plotly_chart(fig_line, use_container_width=True)

            st.dataframe(df_conv.style.format({
                'Mean Energy (kWh)': '{:.2f}',
                'Relative Error (%)': '{:.3f}'
            }))

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>"
    "Developed during Erasmus+ Mobility 2026 | "
    "Konstantin Preslavsky University of Shumen | "
    "Monte Carlo Framework for Industry 5.0 Resilience"
    "</p>",
    unsafe_allow_html=True
)