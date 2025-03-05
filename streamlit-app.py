import os
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from snowflake.snowpark import Session
import datetime

# Page configuration
st.set_page_config(
    page_title="Aircraft Predictive Maintenance",
    page_icon="✈️",
    layout="wide"
)

# Connect to Snowflake
# You'd have connection parameters stored securely
def get_snowflake_session():
    """
    Create a Snowflake session using environment variables
    Recommended for secure credential management
    """
    # Load environment variables
    load_dotenv()

    # Retrieve credentials from environment variables
    connection_parameters = {
        "account": os.getenv("SNOWFLAKE_ACCOUNT"),
        "user": os.getenv("SNOWFLAKE_USER"),
        "password": os.getenv("SNOWFLAKE_PASSWORD"),
        "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE", "DEMO_WH"),
        "database": os.getenv("SNOWFLAKE_DATABASE", "AIRLINE_OPERATIONAL_DATA"),
        "schema": os.getenv("SNOWFLAKE_SCHEMA", "PLATINUM"),
        "role": os.getenv("SNOWFLAKE_ROLE", "STREAMLIT_ROLE")
    }

    # Validate required credentials
    required_keys = ["account", "user", "password"]
    for key in required_keys:
        if not connection_parameters[key]:
            raise ValueError(f"Missing required Snowflake credential: {key}")

    try:
        return Session.builder.configs(connection_parameters).create()
    except Exception as e:
        st.error(f"Failed to create Snowflake session: {e}")
        return None


# Add caching for session creation
@st.cache_resource
def create_snowflake_session():
    """Cached Snowflake session creation"""
    return get_snowflake_session()


# Error handling wrapper for database queries
def safe_snowflake_query(session, query):
    """
    Safely execute Snowflake queries with error handling
    
    Args:
        session (Session): Snowflake session
        query (str): SQL query to execute
    
    Returns:
        pd.DataFrame: Query results or empty DataFrame
    """
    try:
        return session.sql(query).to_pandas()
    except Exception as e:
        st.error(f"Database query failed: {e}")
        return pd.DataFrame()


def main():
    # Title and introduction
    st.title("Aircraft Predictive Maintenance Dashboard")
    st.markdown("*Powered by Snowflake | Advanced Analytics Platform*")

    # Create Snowflake session
    session = create_snowflake_session()
    if not session:
        st.error("Unable to establish Snowflake connection. Please check credentials.")
        return

    # Sidebar for filtering
    st.sidebar.title("Filters")
    
    # Cache common queries
    @st.cache_data(ttl=600)
    def get_airlines(_session):
        return session.table("platinum.airline_dashboard_kpis").select("airline_name").to_pandas()
    
    airlines = get_airlines(session)
    selected_airline = st.sidebar.selectbox("Select Airline", airlines["AIRLINE_NAME"])

    # Date range selector
    today = datetime.date.today()
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(today - datetime.timedelta(days=30), today),
        max_value=today
    )

    # Main dashboard
    st.title(f"{selected_airline} Aircraft Predictive Maintenance Dashboard")

    # Top KPIs
    st.header("Fleet Health Overview")
    
    @st.cache_data(ttl=600)
    def get_kpi_data(_session, airline):
        kpi_query = f"""
            SELECT * FROM platinum.airline_dashboard_kpis
            WHERE airline_name = '{airline}'
        """
        return safe_snowflake_query(session, kpi_query)
    
    kpi_data = get_kpi_data(session, selected_airline)

    # KPI cards in a row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Fleet Size", kpi_data["FLEET_SIZE"].iloc[0])
    with col2:
        st.metric("Components Needing Attention", kpi_data["COMPONENTS_NEEDING_ATTENTION"].iloc[0])
    with col3:
        st.metric("Critical Risk Components", kpi_data["CRITICAL_RISK_COMPONENTS"].iloc[0])
    with col4:
        st.metric("Potential Savings", f"${kpi_data['POTENTIAL_SAVINGS_USD'].iloc[0]:,.2f}")

    # Fleet Health Status
    st.header("Fleet Health Status")
    
    @st.cache_data(ttl=600)
    def get_fleet_data(_session, airline):
        fleet_query = f"""
            SELECT aircraft_registration, aircraft_model, health_score, aircraft_health_status,
                critical_components, warning_components, average_component_wear
            FROM platinum.fleet_health_scoring
            WHERE airline_name = '{airline}'
            ORDER BY health_score
        """
        return safe_snowflake_query(session, fleet_query)
    
    fleet_data = get_fleet_data(session, selected_airline)

    # Health score chart
    fig_health = px.bar(
        fleet_data, 
        x="AIRCRAFT_REGISTRATION", 
        y="HEALTH_SCORE",
        color="AIRCRAFT_HEALTH_STATUS",
        color_discrete_map={
            "Critical": "#FF2B2B",
            "Warning": "#FF9E2D",
            "Caution": "#FFDF3C",
            "Healthy": "#6ECB63"
        },
        labels={"AIRCRAFT_REGISTRATION": "Aircraft", "HEALTH_SCORE": "Health Score (0-100)"},
        title="Aircraft Health Scores"
    )
    st.plotly_chart(fig_health, use_container_width=True)

    # ENHANCED VISUALIZATION 1: Component Risk Matrix
    st.header("Component Risk Matrix")
    
    # Add component type filter
    component_filter = st.selectbox(
        "Filter by Component Type:", 
        options=["All", "Engine", "Landing Gear", "APU", "Hydraulic System", "Electrical System", "Flight Controls", "Avionics"],
        index=0
    )
    
    @st.cache_data(ttl=600)
    def get_risk_data(_session, airline, component_filter):
        # Enhanced query for risk matrix
        risk_query = f"""
            SELECT 
                crs.component_id,
                crs.component_name,
                crs.aircraft_registration,
                crs.risk_category,
                crs.risk_score,
                crs.failure_probability,
                crs.component_criticality,
                -- Calculate operational impact score based on multiple factors
                CASE
                    WHEN crs.component_criticality = 'Critical' THEN 90
                    WHEN crs.component_criticality = 'High' THEN 70
                    WHEN crs.component_criticality = 'Medium' THEN 50
                    ELSE 30
                END + 
                CASE 
                    WHEN cht.replacement_cost_usd > 1000000 THEN 10
                    WHEN cht.replacement_cost_usd > 500000 THEN 7
                    WHEN cht.replacement_cost_usd > 100000 THEN 5
                    ELSE 2
                END AS operational_impact,
                cht.replacement_cost_usd
            FROM platinum.component_risk_scoring crs
            JOIN gold.component_health_tracking cht ON crs.component_id = cht.component_id
            WHERE crs.airline_name = '{airline}'
            {f"AND crs.component_name LIKE '%{component_filter}%'" if component_filter != "All" else ""}
            ORDER BY risk_score DESC
        """
        return safe_snowflake_query(session, risk_query)
    
    risk_data = get_risk_data(session, selected_airline, component_filter)
    
    # Create true risk matrix
    fig_risk = px.scatter(
        risk_data,
        x="FAILURE_PROBABILITY",
        y="OPERATIONAL_IMPACT",
        color="RISK_CATEGORY",
        size="RISK_SCORE",
        hover_name="COMPONENT_NAME",
        text="COMPONENT_NAME",
        color_discrete_map={
            "Critical Risk": "#FF2B2B",
            "High Risk": "#FF9E2D",
            "Medium Risk": "#FFDF3C",
            "Low Risk": "#6ECB63"
        },
        title="Component Risk Matrix: Impact vs. Probability",
        labels={"FAILURE_PROBABILITY": "Failure Probability", 
                "OPERATIONAL_IMPACT": "Operational Impact"},
        hover_data=["REPLACEMENT_COST_USD", "AIRCRAFT_REGISTRATION", "COMPONENT_CRITICALITY"]
    )
    
    # Add quadrant lines
    fig_risk.add_shape(type="line", x0=0.5, y0=0, x1=0.5, y1=100,
                      line=dict(color="grey", width=1, dash="dash"))
    fig_risk.add_shape(type="line", x0=0, y0=50, x1=1, y1=50,
                      line=dict(color="grey", width=1, dash="dash"))
    
    # Add quadrant labels
    fig_risk.add_annotation(x=0.25, y=75, text="High Impact<br>Low Probability",
                          showarrow=False, font=dict(size=10))
    fig_risk.add_annotation(x=0.75, y=75, text="High Impact<br>High Probability",
                          showarrow=False, font=dict(size=10, color="red"))
    fig_risk.add_annotation(x=0.25, y=25, text="Low Impact<br>Low Probability",
                          showarrow=False, font=dict(size=10))
    fig_risk.add_annotation(x=0.75, y=25, text="Low Impact<br>High Probability",
                          showarrow=False, font=dict(size=10))
    
    # Improve layout
    fig_risk.update_layout(
        height=600,
        xaxis=dict(range=[0, 1], tickformat=".0%"),
        yaxis=dict(range=[0, 100]),
        hoverlabel=dict(bgcolor="white", font_size=12)
    )
    
    # Show only component names for high risk components
    fig_risk.update_traces(
        textposition='top center',
        textfont=dict(size=10),
        texttemplate=lambda x: x if risk_data.loc[x.point_inds, "RISK_CATEGORY"].iloc[0] == "Critical Risk" else ""
    )
    
    st.plotly_chart(fig_risk, use_container_width=True)
    
    # Show top critical components in table format
    if not risk_data.empty:
        critical_components = risk_data[risk_data["RISK_CATEGORY"] == "Critical Risk"].sort_values("RISK_SCORE", ascending=False).head(5)
        if not critical_components.empty:
            st.subheader("Top Critical Components")
            st.dataframe(
                critical_components[["COMPONENT_NAME", "AIRCRAFT_REGISTRATION", "FAILURE_PROBABILITY", "RISK_SCORE", "REPLACEMENT_COST_USD"]],
                column_config={
                    "COMPONENT_NAME": "Component",
                    "AIRCRAFT_REGISTRATION": "Aircraft",
                    "FAILURE_PROBABILITY": st.column_config.NumberColumn("Failure Probability", format="%.1f%%", width="medium"),
                    "RISK_SCORE": st.column_config.NumberColumn("Risk Score", format="%.1f", width="small"),
                    "REPLACEMENT_COST_USD": st.column_config.NumberColumn("Replacement Cost", format="$%d", width="medium")
                },
                use_container_width=True
            )

    # ENHANCED VISUALIZATION 2: Cost Savings Dashboard
    st.header("Maintenance ROI & Cost Savings Analysis")

    @st.cache_data(ttl=600)
    def get_cost_data(_session, airline):
        # Get more detailed cost data with time dimension
        cost_query = f"""
            SELECT
                -- For trend analysis over time
                DATE_TRUNC('month', r.recommended_date) AS month,
                SUM(cba.potential_cost_savings_usd) AS cumulative_savings,
                -- For component breakdown
                ct.component_category,
                SUM(cba.reactive_maintenance_cost_usd) AS reactive_cost,
                SUM(cba.predictive_maintenance_cost_usd) AS predictive_cost,
                SUM(cba.potential_cost_savings_usd) AS savings,
                -- For ROI calculation
                AVG(cba.roi_percentage) AS avg_roi
            FROM platinum.maintenance_recommendations r
            JOIN platinum.maintenance_cost_benefit_analysis cba ON r.component_id = cba.component_id
            JOIN gold.component_health_tracking cht ON r.component_id = cht.component_id
            JOIN master_data.component_types ct ON cht.component_type_id = ct.component_type_id
            WHERE cht.airline_name = '{airline}'
            GROUP BY 1, 3
            ORDER BY 1, 3
        """
        return safe_snowflake_query(session, cost_query)
    
    cost_data = get_cost_data(session, selected_airline)
    
    # Create two columns for cost visualizations
    cost_col1, cost_col2 = st.columns(2)
    
    with cost_col1:
        if not cost_data.empty:
            # Enhanced cost comparison with annotations
            # Aggregate by component category
            summary_data = cost_data.groupby('COMPONENT_CATEGORY').agg({
                'REACTIVE_COST': 'sum',
                'PREDICTIVE_COST': 'sum',
                'SAVINGS': 'sum'
            }).reset_index()
            
            # Create a stacked bar chart for reactive vs predictive by component
            comp_cost_data = pd.DataFrame({
                'Component Category': summary_data['COMPONENT_CATEGORY'].repeat(2),
                'Maintenance Type': ['Reactive', 'Predictive'] * len(summary_data),
                'Cost (USD)': list(summary_data['REACTIVE_COST']) + list(summary_data['PREDICTIVE_COST'])
            })
            
            fig_cost = px.bar(
                comp_cost_data,
                x='Component Category',
                y='Cost (USD)',
                color='Maintenance Type',
                barmode='group',
                title='Cost Comparison by Component Category',
                color_discrete_map={
                    'Reactive': '#FF5A5F',
                    'Predictive': '#2E86C1'
                }
            )
            
            # Add savings annotations
            for i, row in summary_data.iterrows():
                savings_pct = (row['SAVINGS'] / row['REACTIVE_COST']) * 100 if row['REACTIVE_COST'] > 0 else 0
                fig_cost.add_annotation(
                    x=row['COMPONENT_CATEGORY'],
                    y=max(row['REACTIVE_COST'], row['PREDICTIVE_COST']) + 50000,
                    text=f"Save: ${row['SAVINGS']:,.0f}<br>({savings_pct:.1f}%)",
                    showarrow=False,
                    font=dict(size=10, color='green')
                )
            
            st.plotly_chart(fig_cost, use_container_width=True)
        else:
            st.warning("No cost data available for selected airline.")
    
    with cost_col2:
        if not cost_data.empty:
            # Create cumulative savings trend chart
            trend_data = cost_data.groupby('MONTH').agg({
                'CUMULATIVE_SAVINGS': 'sum'
            }).reset_index()
            
            # Sort by date
            trend_data = trend_data.sort_values('MONTH')
            
            # Calculate cumulative total
            trend_data['RUNNING_TOTAL'] = trend_data['CUMULATIVE_SAVINGS'].cumsum()
            
            fig_trend = px.area(
                trend_data,
                x='MONTH',
                y='RUNNING_TOTAL',
                title='Cumulative Cost Savings Over Time',
                labels={'RUNNING_TOTAL': 'Cumulative Savings (USD)', 'MONTH': 'Month'}
            )
            
            fig_trend.update_traces(
                line=dict(color='#28B463'),
                fillcolor='rgba(40, 180, 99, 0.3)'
            )
            
            # Add ROI indicator
            avg_roi = cost_data['AVG_ROI'].mean()
            if not pd.isna(avg_roi) and len(trend_data) > 0:
                fig_trend.add_annotation(
                    x=trend_data['MONTH'].iloc[-1],
                    y=trend_data['RUNNING_TOTAL'].iloc[-1],
                    text=f"Average ROI: {avg_roi:.1f}%",
                    showarrow=True,
                    arrowhead=1
                )
            
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.warning("No trend data available for selected airline.")

    # ENHANCED VISUALIZATION 3: Maintenance Schedule Optimizer
    st.header("Optimized Maintenance Schedule")

    # Add optimization controls
    optimization_col1, optimization_col2, optimization_col3 = st.columns(3)

    with optimization_col1:
        downtime_weight = st.slider("Weight: Minimize Downtime", 1, 10, 5)

    with optimization_col2:
        urgency_weight = st.slider("Weight: Component Urgency", 1, 10, 7) 

    with optimization_col3:
        resource_weight = st.slider("Weight: Resource Utilization", 1, 10, 3)

    # Call the stored procedure to get an optimized maintenance schedule
    @st.cache_data(ttl=60)  # Short cache time since this depends on sliders
    def get_schedule_data(_session, airline, downtime_weight, urgency_weight, resource_weight):
        schedule_query = f"""
            CALL platinum.generate_optimized_maintenance_schedule(
                '{airline}',
                {downtime_weight},
                {urgency_weight},
                {resource_weight}
            )
        """
        return safe_snowflake_query(session, schedule_query)
    
    schedule_data = get_schedule_data(session, selected_airline, downtime_weight, urgency_weight, resource_weight)

    # Create the enhanced Gantt chart
    if not schedule_data.empty:
        # Add columns for resource info display
        resource_col1, resource_col2 = st.columns(2)
        
        with resource_col1:
            # Show resource utilization chart (if data has the right columns)
            if 'SCHEDULED_DATE' in schedule_data.columns and 'TECHNICIAN_COUNT' in schedule_data.columns:
                resources_by_day = schedule_data.groupby('SCHEDULED_DATE').agg({
                    'TECHNICIAN_COUNT': 'sum'
                }).reset_index()
                
                fig_resources = px.bar(
                    resources_by_day,
                    x='SCHEDULED_DATE',
                    y='TECHNICIAN_COUNT',
                    title="Daily Technician Requirements",
                    labels={'TECHNICIAN_COUNT': 'Technicians Needed'}
                )
                
                # Add capacity line
                max_daily_capacity = 25  # This would come from your data in practice
                fig_resources.add_shape(
                    type="line",
                    x0=resources_by_day['SCHEDULED_DATE'].min(),
                    y0=max_daily_capacity,
                    x1=resources_by_day['SCHEDULED_DATE'].max(),
                    y1=max_daily_capacity,
                    line=dict(color="red", width=2, dash="dash")
                )
                
                fig_resources.add_annotation(
                    x=resources_by_day['SCHEDULED_DATE'].mean(),
                    y=max_daily_capacity,
                    text="Max Capacity",
                    showarrow=False,
                    yshift=10
                )
                
                st.plotly_chart(fig_resources, use_container_width=True)
            else:
                st.warning("Resource utilization data not available in the schedule.")
        
        with resource_col2:
            # Show maintenance duration chart
            st.metric("Total Maintenance Events", len(schedule_data))
            
            if 'DURATION_HOURS' in schedule_data.columns and 'TECHNICIAN_COUNT' in schedule_data.columns:
                st.metric("Total Technician-Hours", 
                         int(schedule_data['DURATION_HOURS'].sum() * schedule_data['TECHNICIAN_COUNT'].sum()))
            
            if 'SCHEDULED_DATE' in schedule_data.columns:
                st.metric("Total Aircraft Days Affected", schedule_data['SCHEDULED_DATE'].nunique())
            
            # Calculate before vs after optimization stats
            st.markdown("### Optimization Improvements")
            
            # This would be calculated from your data in real implementation
            before_after_data = pd.DataFrame({
                'Metric': ['Aircraft Downtime', 'Maintenance Costs', 'Resource Utilization'],
                'Before': [100, 100, 100],
                'After': [68, 82, 92]
            })
            
            fig_improvement = px.bar(
                before_after_data,
                x='Metric',
                y=['Before', 'After'],
                barmode='group',
                title="Optimization Impact (%)",
                color_discrete_map={
                    'Before': '#DB4437',
                    'After': '#0F9D58'
                }
            )
            
            st.plotly_chart(fig_improvement, use_container_width=True)
        
        # Process Gantt chart data
        if 'RECOMMENDED_DATE' in schedule_data.columns:
            schedule_data['START_DATE'] = schedule_data['RECOMMENDED_DATE']
            schedule_data['END_DATE'] = schedule_data.apply(
                lambda x: x['RECOMMENDED_DATE'] + pd.Timedelta(days=1), axis=1
            )
            
            # Enhanced Gantt chart
            fig_gantt = px.timeline(
                schedule_data, 
                x_start="START_DATE", 
                x_end="END_DATE", 
                y="AIRCRAFT_REGISTRATION",
                color="RISK_CATEGORY",
                hover_name="COMPONENT_NAME",
                hover_data=["DURATION_HOURS", "TECHNICIAN_COUNT", "MAINTENANCE_TYPE"] if all(col in schedule_data.columns for col in ["DURATION_HOURS", "TECHNICIAN_COUNT", "MAINTENANCE_TYPE"]) else None,
                color_discrete_map={
                    "Critical Risk": "#FF2B2B",
                    "High Risk": "#FF9E2D",
                    "Medium Risk": "#FFDF3C",
                    "Low Risk": "#6ECB63"
                },
                title="Maintenance Schedule Timeline"
            )
            
            fig_gantt.update_yaxes(autorange="reversed")
            fig_gantt.update_layout(height=600)
            
            # Add vertical line for today
            today = pd.to_datetime('today')
            fig_gantt.add_vline(x=today, line_width=2, line_dash="solid", line_color="black")
            fig_gantt.add_annotation(x=today, y=0, text="Today", showarrow=False, yshift=10)
            
            st.plotly_chart(fig_gantt, use_container_width=True)
        else:
            st.warning("Missing required date columns in schedule data.")
    else:
        st.warning("No maintenance schedule data available.")

    # ENHANCED VISUALIZATION 4: Sensor Anomaly Analysis
    st.header("Sensor Anomaly Analysis")

    # Add more sophisticated controls
    anomaly_col1, anomaly_col2, anomaly_col3 = st.columns(3)

    with anomaly_col1:
        component_type = st.selectbox(
            "Component Type",
            ["All", "Engine", "Landing Gear", "APU", "Hydraulic System", "Electrical System", "Flight Controls", "Avionics"]
        )

    with anomaly_col2:
        sensor_type = st.selectbox(
            "Sensor Type",
            ["All", "Temperature", "Pressure", "Vibration", "Flow Rate", "RPM"]
        )

    with anomaly_col3:
        time_range = st.selectbox(
            "Time Range",
            ["Last 30 Days", "Last 90 Days", "Last 6 Months", "Last Year"]
        )

    # Convert time range to actual date
    time_mapping = {
        "Last 30 Days": 30,
        "Last 90 Days": 90,
        "Last 6 Months": 180,
        "Last Year": 365
    }
    from_date = (pd.to_datetime('today') - pd.Timedelta(days=time_mapping[time_range])).strftime('%Y-%m-%d')

    # Query for anomaly data with enhanced fields
    @st.cache_data(ttl=600)
    def get_anomaly_data(_session, airline, from_date, component_type, sensor_type):
        anomaly_query = f"""
            WITH base_data AS (
                SELECT 
                    sa.reading_date,
                    sa.sensor_id,
                    sa.sensor_name,
                    sa.component_id,
                    sa.component_name,
                    ct.component_category,
                    sa.aircraft_id,
                    sa.aircraft_registration,
                    sa.critical_anomalies,
                    sa.warning_anomalies,
                    sa.statistical_anomalies,
                    sa.total_readings,
                    -- Pattern information
                    CASE 
                        WHEN sa.reading_date > CURRENT_DATE - 7 AND sa.critical_anomalies > 0 THEN 'Recent Critical'
                        WHEN sa.critical_anomalies > 5 THEN 'Multiple Critical'
                        WHEN sa.warning_anomalies > 10 THEN 'Multiple Warnings'
                        WHEN sa.warning_anomalies > 0 AND sa.statistical_anomalies > 5 THEN 'Developing Issue'
                        ELSE 'Normal'
                    END AS pattern_type,
                    -- For forecast
                    AVG(sa.warning_anomalies) OVER (
                        PARTITION BY sa.component_id 
                        ORDER BY sa.reading_date
                        ROWS BETWEEN 14 PRECEDING AND CURRENT ROW
                    ) AS moving_avg_warnings,
                    AVG(sa.critical_anomalies) OVER (
                        PARTITION BY sa.component_id 
                        ORDER BY sa.reading_date
                        ROWS BETWEEN 14 PRECEDING AND CURRENT ROW
                    ) AS moving_avg_critical
                FROM platinum.sensor_anomaly_timeline sa
                JOIN gold.component_health_tracking cht ON sa.component_id = cht.component_id
                JOIN master_data.component_types ct ON cht.component_type_id = ct.component_type_id
                WHERE sa.airline_name = '{airline}'
                AND sa.reading_date >= '{from_date}'
                {f"AND ct.component_category LIKE '%{component_type}%'" if component_type != "All" else ""}
                {f"AND sa.sensor_name LIKE '%{sensor_type}%'" if sensor_type != "All" else ""}
            )
            SELECT * FROM base_data
            ORDER BY reading_date
        """
        return safe_snowflake_query(session, anomaly_query)
    
    anomaly_data = get_anomaly_data(session, selected_airline, from_date, component_type, sensor_type)

    # Two-part visualization: timeline view and heatmap view
    anomaly_tabs = st.tabs(["Timeline Analysis", "Anomaly Heatmap", "Pattern Detection"])

    with anomaly_tabs[0]:
        if not anomaly_data.empty:
            # Aggregate by date for the timeline
            timeline_data = anomaly_data.groupby('READING_DATE').agg({
                'CRITICAL_ANOMALIES': 'sum',
                'WARNING_ANOMALIES': 'sum',
                'STATISTICAL_ANOMALIES': 'sum'
            }).reset_index()
            
            # Add forecasted values for next 14 days
            last_date = pd.to_datetime(timeline_data['READING_DATE'].max())
            forecast_dates = [last_date + pd.Timedelta(days=i) for i in range(1, 15)]
            
            # Simple forecasting (in a real app you'd use a proper model)
            last_data = timeline_data.iloc[-14:].mean() if len(timeline_data) >= 14 else timeline_data.mean()
            trend_factor = 1.0
            if len(timeline_data) > 28:
                last_period = timeline_data.iloc[-14:]
                prior_period = timeline_data.iloc[-28:-14]
                if last_period['CRITICAL_ANOMALIES'].mean() > prior_period['CRITICAL_ANOMALIES'].mean():
                    trend_factor = last_period['CRITICAL_ANOMALIES'].mean() / max(1, prior_period['CRITICAL_ANOMALIES'].mean())
                    trend_factor = min(2.0, max(1.0, trend_factor))  # Cap between 1.0-2.0
            
            forecast_data = pd.DataFrame({
                'READING_DATE': forecast_dates,
                'CRITICAL_ANOMALIES': [last_data['CRITICAL_ANOMALIES'] * trend_factor * (1.05**i) for i in range(1, 15)],
                'WARNING_ANOMALIES': [last_data['WARNING_ANOMALIES'] * trend_factor * (1.03**i) for i in range(1, 15)],
                'STATISTICAL_ANOMALIES': [last_data['STATISTICAL_ANOMALIES'] * trend_factor * (1.02**i) for i in range(1, 15)]
            })
            
            # Mark as forecast
            forecast_data['IS_FORECAST'] = True
            timeline_data['IS_FORECAST'] = False
            
            # Create enhanced timeline with forecast
            fig_timeline = go.Figure()
            
            # Add actual data
            fig_timeline.add_trace(go.Scatter(
                x=timeline_data['READING_DATE'],
                y=timeline_data['CRITICAL_ANOMALIES'],
                mode='lines+markers',
                name='Critical Anomalies',
                line=dict(color='#FF2B2B', width=2),
                marker=dict(size=6)
            ))
            
            fig_timeline.add_trace(go.Scatter(
                x=timeline_data['READING_DATE'],
                y=timeline_data['WARNING_ANOMALIES'],
                mode='lines+markers',
                name='Warning Anomalies',
                line=dict(color='#FF9E2D', width=2),
                marker=dict(size=6)
            ))
            
            # Add forecast data
            fig_timeline.add_trace(go.Scatter(
                x=forecast_data['READING_DATE'],
                y=forecast_data['CRITICAL_ANOMALIES'],
                mode='lines',
                name='Critical Forecast',
                line=dict(color='#FF2B2B', width=2, dash='dash')
            ))
            
            fig_timeline.add_trace(go.Scatter(
                x=forecast_data['READING_DATE'],
                y=forecast_data['WARNING_ANOMALIES'],
                mode='lines',
                name='Warning Forecast',
                line=dict(color='#FF9E2D', width=2, dash='dash')
            ))
            
            # Add a vertical line for today
            today = pd.to_datetime('today')
            fig_timeline.add_vline(x=today, line_width=2, line_dash="dash", line_color="black")
            fig_timeline.add_annotation(x=today, y=0, text="Today", showarrow=False, yshift=10)
            
            # Update layout
            fig_timeline.update_layout(
                title='Sensor Anomalies Timeline with 14-Day Forecast',
                xaxis_title='Date',
                yaxis_title='Number of Anomalies',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                height=500
            )
            
            st.plotly_chart(fig_timeline, use_container_width=True)
        else:
            st.warning("No anomaly data available for the selected filters.")

    with anomaly_tabs[1]:
        # Create heatmap data - aggregate by component and date
        if not anomaly_data.empty:
            try:
                # Pivot data for heatmap - handle potential pivot errors
                component_dates = anomaly_data.pivot_table(
                    index='COMPONENT_NAME',
                    columns='READING_DATE',
                    values='CRITICAL_ANOMALIES',
                    aggfunc='sum',
                    fill_value=0
                )
                
                if not component_dates.empty:
                    # Create heatmap
                    fig_heatmap = px.imshow(
                        component_dates,
                        labels=dict(x="Date", y="Component", color="Critical Anomalies"),
                        x=component_dates.columns,
                        y=component_dates.index,
                        color_continuous_scale='Reds',
                        title='Critical Anomalies by Component Over Time'
                    )
                    
                    fig_heatmap.update_layout(height=600)
                    st.plotly_chart(fig_heatmap, use_container_width=True)
                else:
                    st.warning("No critical anomalies detected in the selected time period.")
            except Exception as e:
                st.error(f"Error creating heatmap: {e}")
                st.warning("Insufficient data for heatmap visualization.")
        else:
            st.warning("No anomaly data available for the selected filters.")

    with anomaly_tabs[2]:
        # Pattern detection and analysis
        if not anomaly_data.empty and 'PATTERN_TYPE' in anomaly_data.columns:
            pattern_data = anomaly_data.groupby('PATTERN_TYPE').agg({
                'COMPONENT_ID': 'nunique',
                'CRITICAL_ANOMALIES': 'sum',
                'WARNING_ANOMALIES': 'sum'
            }).reset_index()
            
            fig_patterns = px.bar(
                pattern_data,
                x='PATTERN_TYPE',
                y='COMPONENT_ID',
                title='Components Showing Anomaly Patterns',
                color='PATTERN_TYPE',
                text='COMPONENT_ID',
                color_discrete_map={
                    'Recent Critical': '#FF2B2B',
                    'Multiple Critical': '#FF5A5F',
                    'Multiple Warnings': '#FF9E2D',
                    'Developing Issue': '#FFDF3C',
                    'Normal': '#6ECB63'
                },
                labels={'COMPONENT_ID': 'Component Count', 'PATTERN_TYPE': 'Pattern Type'}
            )
            
            st.plotly_chart(fig_patterns, use_container_width=True)
            
            # Show specific components with patterns
            if 'Normal' in anomaly_data['PATTERN_TYPE'].values:
                patterns_to_show = anomaly_data[anomaly_data['PATTERN_TYPE'] != 'Normal']
                if not patterns_to_show.empty:
                    patterns_summary = patterns_to_show.groupby(['COMPONENT_NAME', 'PATTERN_TYPE', 'AIRCRAFT_REGISTRATION']).agg({
                        'CRITICAL_ANOMALIES': 'sum',
                        'WARNING_ANOMALIES': 'sum'
                    }).reset_index()
                    
                    st.subheader("Components with Anomaly Patterns")
                    st.dataframe(
                        patterns_summary.sort_values('CRITICAL_ANOMALIES', ascending=False),
                        column_config={
                            "COMPONENT_NAME": "Component",
                            "PATTERN_TYPE": "Pattern Type",
                            "AIRCRAFT_REGISTRATION": "Aircraft",
                            "CRITICAL_ANOMALIES": st.column_config.NumberColumn("Critical Anomalies", format="%d"),
                            "WARNING_ANOMALIES": st.column_config.NumberColumn("Warning Anomalies", format="%d")
                        },
                        use_container_width=True
                    )
                else:
                    st.info("No components with abnormal patterns in the selected time period.")
        else:
            st.warning("Pattern detection data not available for the selected filters.")

    # Display data refresh time
    if not kpi_data.empty and 'REFRESHED_AT' in kpi_data.columns:
        st.sidebar.write(f"Data refreshed: {kpi_data['REFRESHED_AT'].iloc[0]}")

    # Footer
    st.markdown("---")
    st.caption("Aircraft Predictive Maintenance Demo - Powered by Snowflake")

    # Add a disclaimer
    st.markdown("---")
    st.caption("🔒 Data is anonymized and secured | For demonstration purposes only")

if __name__ == "__main__":
    main()
