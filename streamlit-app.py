import os
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from snowflake.snowpark import Session

# Page configuration - mobile first
st.set_page_config(
    page_title="Lufthansa Maintenance",
    page_icon="✈️",
    initial_sidebar_state="collapsed"
)

# Mobile-optimized CSS
st.markdown("""
<style>
    /* Base styles optimized for mobile */
    .main > div {
        padding-top: 0.5rem;
        padding-left: 0.5rem;
        padding-right: 0.5rem;
    }
    
    h1, h2, h3 {
        margin-top: 0;
        margin-bottom: 0.5rem;
        font-size: 1.5rem;
    }
    
    /* Maintenance card styling */
    .maintenance-card {
        background-color: white;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 10px;
        border-left: 4px solid #1E88E5;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    /* Risk level colors */
    .risk-critical { border-left-color: #E53935; }
    .risk-high { border-left-color: #FB8C00; }
    .risk-medium { border-left-color: #FDD835; }
    .risk-low { border-left-color: #43A047; }
    
    /* Clean, simple header styling */
    .component-name {
        font-weight: bold;
        font-size: 1.1rem;
        margin: 0;
    }
    
    .aircraft-reg {
        color: #555;
        font-size: 0.9rem;
        margin: 0 0 8px 0;
    }
    
    /* Info row styling */
    .info-row {
        display: flex;
        font-size: 0.85rem;
        margin-bottom: 4px;
    }
    
    .info-label {
        width: 80px;
        color: #666;
    }
    
    /* Simple date pills */
    .date-pill {
        display: inline-block;
        padding: 3px 8px;
        background-color: #f0f2f5;
        border-radius: 12px;
        font-size: 0.8rem;
        margin-right: 5px;
        margin-bottom: 5px;
    }
    
    /* Make filter elements more touch-friendly */
    .stRadio > div {
        flex-direction: row;
        gap: 10px;
    }
    
    .stRadio label {
        background-color: #f0f2f5;
        padding: 8px 15px;
        border-radius: 20px;
        min-width: 70px;
        text-align: center;
    }
    
    /* Hide elements we don't need on mobile */
    #MainMenu, footer {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)

# Load environment variables and connect to Snowflake
load_dotenv()

@st.cache_resource(ttl=3600)
def get_snowflake_connection():
    return Session.builder.configs({
        "account": os.getenv("SNOWFLAKE_ACCOUNT"),
        "user": os.getenv("SNOWFLAKE_USER"),
        "password": os.getenv("SNOWFLAKE_PASSWORD"),
        "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE", "DEMO_WH"),
        "database": os.getenv("SNOWFLAKE_DATABASE", "AIRLINE_OPERATIONAL_DATA"),
        "schema": os.getenv("SNOWFLAKE_SCHEMA", "APP_VIEWS")
    }).create()

# Try to connect to Snowflake
try:
    session = get_snowflake_connection()
except Exception as e:
    st.error(f"Unable to connect: {e}")
    st.stop()

# Fetch maintenance data
@st.cache_data(ttl=300)
def get_maintenance_data():
    # Check if view exists first
    try:
        result = session.sql("SHOW VIEWS LIKE 'LUFTHANSA_MAINTENANCE_SCHEDULE' IN APP_VIEWS").collect()
        if len(result) > 0:
            return session.table("LUFTHANSA_MAINTENANCE_SCHEDULE").to_pandas()
        else:
            # If view doesn't exist, create a simple demo dataset
            query = """
            SELECT 
                c.component_id,
                c.component_name,
                c.component_type_id,
                ct.component_name as component_type,
                ct.component_category,
                c.aircraft_id,
                a.registration as aircraft_registration,
                a.model_name as aircraft_model,
                'Warning' as health_status,
                UNIFORM(60, 95, RANDOM()) as percent_life_used,
                ct.criticality_level as component_criticality,
                CURRENT_DATE() + UNIFORM(1, 30, RANDOM())::INT as recommended_date,
                UNIFORM(2, 24, RANDOM()) as duration_hours,
                UNIFORM(1, 5, RANDOM())::INT as technician_count,
                CASE 
                    WHEN UNIFORM(0, 1, RANDOM()) > 0.5 THEN TRUE
                    ELSE FALSE
                END as can_be_bundled,
                CASE
                    WHEN UNIFORM(0, 1, RANDOM()) > 0.8 THEN 'Critical Risk'
                    WHEN UNIFORM(0, 1, RANDOM()) > 0.5 THEN 'High Risk'
                    WHEN UNIFORM(0, 1, RANDOM()) > 0.3 THEN 'Medium Risk'
                    ELSE 'Low Risk'
                END as risk_category,
                UNIFORM(1, 7, RANDOM())::INT as maintenance_type_id,
                'Maintenance Task' as maintenance_type,
                'M' || UNIFORM(100, 999, RANDOM())::INT as maintenance_code,
                UNIFORM(0, 5, RANDOM())::INT as warning_sensor_readings,
                UNIFORM(0, 2, RANDOM())::INT as critical_sensor_readings,
                'LH-MAINT-' || UNIFORM(1000, 9999, RANDOM())::INT as maintenance_reference,
                'Lufthansa' as airline_name
            FROM master_data.manufacturer_components c
            JOIN master_data.component_types ct ON c.component_type_id = ct.component_type_id
            JOIN master_data.manufacturer_aircraft a ON c.aircraft_id = a.aircraft_id
            JOIN master_data.airlines air ON a.airline_id = air.airline_id
            WHERE air.airline_name = 'Lufthansa'
            LIMIT 30
            """
            return session.sql(query).to_pandas()
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

def main():
    # Minimalist header
    st.markdown("<h1>Lufthansa Maintenance</h1>", unsafe_allow_html=True)
    
    # Load data
    maintenance_data = get_maintenance_data()
    
    # Early exit if no data
    if maintenance_data.empty:
        st.warning("No maintenance data available.")
        return
    
    # Convert date column to datetime
    if 'RECOMMENDED_DATE' in maintenance_data.columns:
        maintenance_data['RECOMMENDED_DATE'] = pd.to_datetime(maintenance_data['RECOMMENDED_DATE'])
    
    # Simple date filter with a radio button
    today = pd.Timestamp.today()
    date_options = ["Today", "This Week", "This Month", "All"]
    selected_range = st.radio("View schedule:", date_options, horizontal=True)
    
    # Apply date filter
    if selected_range == "Today":
        end_date = today
        filtered_data = maintenance_data[maintenance_data['RECOMMENDED_DATE'].dt.date == today.date()]
    elif selected_range == "This Week":
        end_date = today + pd.Timedelta(days=7)
        filtered_data = maintenance_data[
            (maintenance_data['RECOMMENDED_DATE'] >= today) & 
            (maintenance_data['RECOMMENDED_DATE'] <= end_date)
        ]
    elif selected_range == "This Month":
        end_date = today + pd.Timedelta(days=30)
        filtered_data = maintenance_data[
            (maintenance_data['RECOMMENDED_DATE'] >= today) & 
            (maintenance_data['RECOMMENDED_DATE'] <= end_date)
        ]
    else:
        filtered_data = maintenance_data
    
    # Show simple count of maintenance items
    critical_count = len(filtered_data[filtered_data['RISK_CATEGORY'] == 'Critical Risk'])
    total_count = len(filtered_data)
    
    st.write(f"**{total_count}** maintenance items scheduled, **{critical_count}** critical")
    
    # Group by date and show date pills
    if not filtered_data.empty:
        unique_dates = filtered_data['RECOMMENDED_DATE'].dt.date.unique()
        st.write("**Dates:**")
        
        date_html = "<div style='margin-bottom:15px'>"
        for date in sorted(unique_dates):
            date_count = filtered_data[filtered_data['RECOMMENDED_DATE'].dt.date == date].shape[0]
            date_str = date.strftime("%b %d")
            date_html += f"<span class='date-pill'>{date_str} ({date_count})</span>"
        date_html += "</div>"
        
        st.markdown(date_html, unsafe_allow_html=True)
    
    # Sort by priority then date
    filtered_data = filtered_data.sort_values(
        by=['RISK_CATEGORY', 'RECOMMENDED_DATE'], 
        ascending=[True, True]
    )
    
    # Show maintenance items as simple cards
    for _, item in filtered_data.iterrows():
        # Determine risk class for card styling
        risk_class = ""
        if item['RISK_CATEGORY'] == 'Critical Risk':
            risk_class = "risk-critical"
        elif item['RISK_CATEGORY'] == 'High Risk':
            risk_class = "risk-high"
        elif item['RISK_CATEGORY'] == 'Medium Risk':
            risk_class = "risk-medium"
        else:
            risk_class = "risk-low"
        
        # Format date for display
        date_str = item['RECOMMENDED_DATE'].strftime("%b %d")
        
        # Create simple, clean maintenance card
        st.markdown(f"""
        <div class="maintenance-card {risk_class}">
            <p class="component-name">{item['COMPONENT_NAME']}</p>
            <p class="aircraft-reg">{item['AIRCRAFT_REGISTRATION']}</p>
            
            <div class="info-row">
                <div class="info-label">Date:</div>
                <div>{date_str}</div>
            </div>
            
            <div class="info-row">
                <div class="info-label">Duration:</div>
                <div>{item['DURATION_HOURS']:.1f} hrs ({item['TECHNICIAN_COUNT']} techs)</div>
            </div>
            
            <div class="info-row">
                <div class="info-label">Status:</div>
                <div>{item['RISK_CATEGORY']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    main()
