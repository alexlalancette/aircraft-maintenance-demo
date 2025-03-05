import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from snowflake.snowpark import Session
from snowflake_connection_helper import get_snowflake_session
import json
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
    airlines = session.table("platinum.airline_dashboard_kpis").select("airline_name").to_pandas()
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
    kpi_query = f"""
        SELECT * FROM platinum.airline_dashboard_kpis
        WHERE airline_name = '{selected_airline}'
    """
    kpi_data = session.sql(kpi_query).to_pandas()

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
    fleet_query = f"""
        SELECT aircraft_registration, aircraft_model, health_score, aircraft_health_status,
            critical_components, warning_components, average_component_wear
        FROM platinum.fleet_health_scoring
        WHERE airline_name = '{selected_airline}'
        ORDER BY health_score
    """
    fleet_data = session.sql(fleet_query).to_pandas()

    # Health score chart
    fig_health = px.bar(
        fleet_data, 
        x="AIRCRAFT_REGISTRATION", 
        y="HEALTH_SCORE",
        color="AIRCRAFT_HEALTH_STATUS",
        color_discrete_map={
            "Critical": "red",
            "Warning": "orange",
            "Caution": "yellow",
            "Healthy": "green"
        },
        labels={"AIRCRAFT_REGISTRATION": "Aircraft", "HEALTH_SCORE": "Health Score (0-100)"},
        title="Aircraft Health Scores"
    )
    st.plotly_chart(fig_health, use_container_width=True)

    # Two columns layout
    col1, col2 = st.columns(2)

    # Component Risk Analysis
    with col1:
        st.subheader("Component Risk Analysis")
        risk_query = f"""
            SELECT component_name, risk_category, risk_score, failure_probability
            FROM platinum.component_risk_scoring
            WHERE airline_name = '{selected_airline}'
            ORDER BY risk_score DESC
            LIMIT 10
        """
        risk_data = session.sql(risk_query).to_pandas()
        
        fig_risk = px.scatter(
            risk_data,
            x="FAILURE_PROBABILITY",
            y="RISK_SCORE",
            color="RISK_CATEGORY",
            size="RISK_SCORE",
            hover_name="COMPONENT_NAME",
            color_discrete_map={
                "Critical Risk": "red",
                "High Risk": "orange",
                "Medium Risk": "yellow",
                "Low Risk": "green"
            },
            title="Top 10 Component Risks"
        )
        st.plotly_chart(fig_risk, use_container_width=True)

    # Sensor Anomaly Trends
    with col2:
        st.subheader("Sensor Anomaly Trends")
        anomaly_query = f"""
            SELECT reading_date, 
                sum(critical_anomalies) as critical,
                sum(warning_anomalies) as warning,
                sum(statistical_anomalies) as statistical
            FROM platinum.sensor_anomaly_timeline
            WHERE airline_name = '{selected_airline}'
            AND reading_date BETWEEN '{date_range[0]}' AND '{date_range[1]}'
            GROUP BY reading_date
            ORDER BY reading_date
        """
        anomaly_data = session.sql(anomaly_query).to_pandas()
        
        fig_anomaly = px.line(
            anomaly_data,
            x="READING_DATE",
            y=["CRITICAL", "WARNING", "STATISTICAL"],
            labels={"value": "Count", "variable": "Anomaly Type"},
            title="Sensor Anomalies Over Time",
            color_discrete_map={
                "CRITICAL": "red",
                "WARNING": "orange",
                "STATISTICAL": "blue"
            }
        )
        st.plotly_chart(fig_anomaly, use_container_width=True)

    # Maintenance Recommendations
    st.header("Maintenance Recommendations")
    recommendations_query = f"""
        SELECT aircraft_registration, component_name, risk_category, 
            recommended_timeframe, recommendation, recommended_date,
            potential_cost_savings_usd
        FROM platinum.maintenance_recommendations
        WHERE airline_name = '{selected_airline}'
        ORDER BY risk_score DESC
        LIMIT 15
    """
    recommendations_data = session.sql(recommendations_query).to_pandas()
    st.dataframe(recommendations_data, use_container_width=True)

    # ROI Analysis
    st.header("Maintenance ROI Analysis")
    roi_query = f"""
        SELECT * FROM platinum.airline_maintenance_roi_summary
        WHERE airline_name = '{selected_airline}'
    """
    roi_data = session.sql(roi_query).to_pandas()

    col1, col2 = st.columns(2)

    with col1:
        # Cost comparison
        cost_data = pd.DataFrame({
            'Scenario': ['Reactive Maintenance', 'Predictive Maintenance'],
            'Cost (USD)': [
                roi_data['TOTAL_REACTIVE_COST_USD'].iloc[0],
                roi_data['TOTAL_PREDICTIVE_COST_USD'].iloc[0]
            ]
        })
        
        fig_cost = px.bar(
            cost_data,
            x='Scenario',
            y='Cost (USD)',
            title='Cost Comparison: Reactive vs. Predictive',
            color='Scenario',
            color_discrete_map={
                'Reactive Maintenance': 'firebrick',
                'Predictive Maintenance': 'royalblue'
            }
        )
        st.plotly_chart(fig_cost, use_container_width=True)

    with col2:
        # Business value metrics
        business_data = pd.DataFrame({
            'Metric': ['Cost Savings', 'Value from Avoided Cancellations', 'Value from Avoided Delays'],
            'Value (USD)': [
                roi_data['TOTAL_POTENTIAL_SAVINGS_USD'].iloc[0],
                roi_data['ESTIMATED_CANCELLATIONS_AVOIDED'].iloc[0] * 25000,
                roi_data['ESTIMATED_DELAY_HOURS_AVOIDED'].iloc[0] * 5000
            ]
        })
        
        fig_business = px.pie(
            business_data,
            values='Value (USD)',
            names='Metric',
            title='Total Business Value Breakdown',
            hole=0.4
        )
        st.plotly_chart(fig_business, use_container_width=True)

    # Optimized Maintenance Schedule
    st.header("Optimized Maintenance Schedule")
    st.write("Based on risk scoring and operational constraints:")

    # Call the stored procedure to get an optimized maintenance schedule
    schedule_query = f"""
        CALL platinum.generate_optimized_maintenance_schedule('{selected_airline}')
    """
    schedule_data = session.sql(schedule_query).to_pandas()

    # Create a Gantt chart for maintenance scheduling
    if not schedule_data.empty:
        # Process data for Gantt chart
        schedule_data['START_DATE'] = schedule_data['RECOMMENDED_DATE']
        schedule_data['END_DATE'] = schedule_data.apply(
            lambda x: x['RECOMMENDED_DATE'] + pd.Timedelta(days=1), axis=1
        )
        
        fig_gantt = px.timeline(
            schedule_data, 
            x_start="START_DATE", 
            x_end="END_DATE", 
            y="AIRCRAFT_REGISTRATION",
            color="RISK_CATEGORY",
            hover_name="COMPONENT_NAME",
            color_discrete_map={
                "Critical Risk": "red",
                "High Risk": "orange",
                "Medium Risk": "yellow"
            },
            title="Maintenance Schedule Timeline"
        )
        
        fig_gantt.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_gantt, use_container_width=True)
    else:
        st.write("No maintenance schedule data available.")

    # Display data refresh time
    st.sidebar.write(f"Data refreshed: {kpi_data['REFRESHED_AT'].iloc[0]}")

    # Footer
    st.markdown("---")
    st.caption("Aircraft Predictive Maintenance Demo - Powered by Snowflake")

    # Add a disclaimer
    st.markdown("---")
    st.caption("🔒 Data is anonymized and secured | For demonstration purposes only")

if __name__ == "__main__":
    main()
