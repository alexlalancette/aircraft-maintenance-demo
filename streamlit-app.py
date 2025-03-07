import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from snowflake.snowpark import Session
from dotenv import load_dotenv

# Page configuration optimized for mobile
st.set_page_config(
    page_title="Lufthansa Maintenance",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Apply mobile-friendly CSS
st.markdown("""
<style>
    /* Make buttons and selectors larger for touch */
    .stButton>button, .stSelectbox>div>div>select {
        height: 3rem;
        font-size: 1rem;
    }
    
    /* Card styling for maintenance items */
    .maintenance-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        border-left: 5px solid #1E88E5;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Risk level indicators */
    .risk-critical { border-left-color: #E53935 !important; }
    .risk-high { border-left-color: #FB8C00 !important; }
    .risk-medium { border-left-color: #FDD835 !important; }
    .risk-low { border-left-color: #43A047 !important; }
    
    /* For mobile-friendly metrics */
    .metric-container {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
    }
    
    .metric-item {
        flex: 1 1 45%;
        min-width: 120px;
        background-color: white;
        border-radius: 8px;
        padding: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .metric-value {
        font-size: 1.5rem;
        font-weight: bold;
    }
    
    .metric-label {
        font-size: 0.8rem;
        color: #555;
    }
    
    /* Bottom navigation styling */
    .nav-container {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        display: flex;
        background-color: white;
        box-shadow: 0 -2px 10px rgba(0,0,0,0.1);
        z-index: 100;
    }
    
    .nav-item {
        flex: 1;
        text-align: center;
        padding: 10px 0;
        color: #555;
    }
    
    .nav-item.active {
        color: #1E88E5;
        border-top: 3px solid #1E88E5;
    }
    
    /* Main content area - add padding to bottom to avoid overlap with nav */
    .main-content {
        padding-bottom: 60px;
    }
</style>
""", unsafe_allow_html=True)

# Load environment variables
load_dotenv()

# Function to create Snowflake connection
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

# Connect to Snowflake
try:
    session = get_snowflake_connection()
except Exception as e:
    st.error(f"Connection error: {e}")
    st.stop()

# Fetch maintenance data
@st.cache_data(ttl=300)
def get_maintenance_data():
    return session.table("LUFTHANSA_MAINTENANCE_SCHEDULE").to_pandas()

# Fetch resource allocation data
@st.cache_data(ttl=300)
def get_resource_data():
    return session.table("LUFTHANSA_RESOURCE_ALLOCATION").to_pandas()

# Fetch bundling opportunities
@st.cache_data(ttl=300)
def get_bundling_data():
    return session.table("LUFTHANSA_MAINTENANCE_BUNDLES").to_pandas()

# Main function
def main():
    # Top app bar
    st.title("Lufthansa Maintenance")
    
    # Initialize data
    maintenance_data = get_maintenance_data()
    resource_data = get_resource_data()
    bundle_data = get_bundling_data()
    
    # Convert date columns to datetime for proper handling
    if 'RECOMMENDED_DATE' in maintenance_data.columns:
        maintenance_data['RECOMMENDED_DATE'] = pd.to_datetime(maintenance_data['RECOMMENDED_DATE'])
    
    if 'RECOMMENDED_DATE' in resource_data.columns:
        resource_data['RECOMMENDED_DATE'] = pd.to_datetime(resource_data['RECOMMENDED_DATE'])
    
    if 'RECOMMENDED_DATE' in bundle_data.columns:
        bundle_data['RECOMMENDED_DATE'] = pd.to_datetime(bundle_data['RECOMMENDED_DATE'])
    
    # Navigation tabs - mobile friendly
    tab1, tab2, tab3, tab4 = st.tabs(["Priority", "Schedule", "Resources", "Aircraft"])
    
    with tab1:
        show_priority_view(maintenance_data)
    
    with tab2:
        show_schedule_view(maintenance_data, bundle_data)
    
    with tab3:
        show_resource_view(resource_data)
    
    with tab4:
        show_aircraft_view(maintenance_data)

def show_priority_view(data):
    """Display priority maintenance items"""
    st.subheader("Priority Maintenance")
    
    # Top summary metrics
    critical_count = len(data[data['RISK_CATEGORY'] == 'Critical Risk'])
    high_count = len(data[data['RISK_CATEGORY'] == 'High Risk'])
    
    # Mobile-friendly metrics display
    st.markdown("""
    <div class="metric-container">
        <div class="metric-item risk-critical">
            <div class="metric-value">{}</div>
            <div class="metric-label">Critical</div>
        </div>
        <div class="metric-item risk-high">
            <div class="metric-value">{}</div>
            <div class="metric-label">High Risk</div>
        </div>
    </div>
    """.format(critical_count, high_count), unsafe_allow_html=True)
    
    # Filter to show only high priority items
    priority_data = data[data['RISK_CATEGORY'].isin(['Critical Risk', 'High Risk'])]
    
    if len(priority_data) > 0:
        # Sort by risk level and date
        priority_data = priority_data.sort_values(
            by=['RISK_CATEGORY', 'RECOMMENDED_DATE'], 
            ascending=[True, True]
        )
        
        # Display as cards
        for _, item in priority_data.iterrows():
            risk_class = ""
            if item['RISK_CATEGORY'] == 'Critical Risk':
                risk_class = "risk-critical"
            elif item['RISK_CATEGORY'] == 'High Risk':
                risk_class = "risk-high"
            
            st.markdown(f"""
            <div class="maintenance-card {risk_class}">
                <h4>{item['COMPONENT_NAME']} • {item['AIRCRAFT_REGISTRATION']}</h4>
                <p><strong>Risk:</strong> {item['RISK_CATEGORY']}</p>
                <p><strong>Date:</strong> {item['RECOMMENDED_DATE'].strftime('%Y-%m-%d')}</p>
                <p><strong>Duration:</strong> {item['DURATION_HOURS']:.1f} hrs • {item['TECHNICIAN_COUNT']} technicians</p>
                <p><strong>Status:</strong> {item['HEALTH_STATUS']} ({item['PERCENT_LIFE_USED']:.1f}% used)</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No high priority maintenance items found.")

def show_schedule_view(data, bundle_data):
    """Display maintenance schedule timeline"""
    st.subheader("Maintenance Schedule")
    
    # Date range filter - mobile friendly
    today = pd.Timestamp.today()
    date_options = ["Next 7 days", "Next 14 days", "Next 30 days", "All"]
    selected_range = st.radio("Time period:", date_options, horizontal=True)
    
    if selected_range == "Next 7 days":
        end_date = today + pd.Timedelta(days=7)
    elif selected_range == "Next 14 days":
        end_date = today + pd.Timedelta(days=14)
    elif selected_range == "Next 30 days":
        end_date = today + pd.Timedelta(days=30)
    else:
        end_date = today + pd.Timedelta(days=365)  # Far future
    
    # Filter data by date range
    filtered_data = data[
        (data['RECOMMENDED_DATE'] >= today) & 
        (data['RECOMMENDED_DATE'] <= end_date)
    ]
    
    # Show bundling opportunities if they exist
    if not bundle_data.empty:
        filtered_bundles = bundle_data[
            (bundle_data['RECOMMENDED_DATE'] >= today) & 
            (bundle_data['RECOMMENDED_DATE'] <= end_date)
        ]
        
        if not filtered_bundles.empty:
            st.info(f"Found {len(filtered_bundles)} bundling opportunities that can save time and resources.")
            
            # Show a sample bundle
            with st.expander("View Bundling Opportunities"):
                for _, bundle in filtered_bundles.iterrows():
                    st.markdown(f"""
                    <div class="maintenance-card">
                        <h4>Bundle for {bundle['AIRCRAFT_REGISTRATION']} on {bundle['RECOMMENDED_DATE'].strftime('%Y-%m-%d')}</h4>
                        <p><strong>Components:</strong> {', '.join(str(c) for c in bundle['BUNDLED_COMPONENTS'])}</p>
                        <p><strong>Duration:</strong> {bundle['BUNDLED_DURATION']:.1f} hrs (saves {bundle['TOTAL_DURATION'] - bundle['BUNDLED_DURATION']:.1f} hrs)</p>
                        <p><strong>Savings:</strong> ${bundle['COST_SAVINGS_USD']:,.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)
    
    # Timeline visualization
    if not filtered_data.empty:
        # Prepare data for Gantt chart
        filtered_data['END_DATE'] = filtered_data.apply(
            lambda x: x['RECOMMENDED_DATE'] + pd.Timedelta(hours=float(x['DURATION_HOURS'])), 
            axis=1
        )
        
        # Create mobile-optimized Gantt chart
        fig = px.timeline(
            filtered_data,
            x_start="RECOMMENDED_DATE",
            x_end="END_DATE",
            y="AIRCRAFT_REGISTRATION",
            color="RISK_CATEGORY",
            hover_name="COMPONENT_NAME",
            color_discrete_map={
                "Critical Risk": "#E53935",
                "High Risk": "#FB8C00",
                "Medium Risk": "#FDD835",
                "Low Risk": "#43A047"
            }
        )
        
        # Improve readability on mobile
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(
            height=400,
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
        )
        
        # Add today line
        fig.add_vline(x=today, line_width=2, line_dash="dash", line_color="black")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show maintenance cards for selected period
        for _, item in filtered_data.iterrows():
            risk_class = ""
            if item['RISK_CATEGORY'] == 'Critical Risk':
                risk_class = "risk-critical"
            elif item['RISK_CATEGORY'] == 'High Risk':
                risk_class = "risk-high"
            elif item['RISK_CATEGORY'] == 'Medium Risk':
                risk_class = "risk-medium"
            else:
                risk_class = "risk-low"
            
            st.markdown(f"""
            <div class="maintenance-card {risk_class}">
                <h4>{item['COMPONENT_NAME']} • {item['AIRCRAFT_REGISTRATION']}</h4>
                <p><strong>Date:</strong> {item['RECOMMENDED_DATE'].strftime('%Y-%m-%d')}</p>
                <p><strong>Type:</strong> {item['MAINTENANCE_TYPE']} ({item['MAINTENANCE_CODE']})</p>
                <p><strong>Duration:</strong> {item['DURATION_HOURS']:.1f} hrs • {item['TECHNICIAN_COUNT']} technicians</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info(f"No maintenance scheduled for the selected period.")

def show_resource_view(resource_data):
    """Display resource allocation view"""
    st.subheader("Resource Allocation")
    
    if not resource_data.empty:
        # Simple resource summary
        avg_techs = resource_data['TOTAL_TECHNICIANS_NEEDED'].mean()
        max_techs = resource_data['TOTAL_TECHNICIANS_NEEDED'].max()
        constraint_days = resource_data['RESOURCE_CONSTRAINT_FLAG'].sum()
        
        # Mobile-friendly metrics
        st.markdown("""
        <div class="metric-container">
            <div class="metric-item">
                <div class="metric-value">{:.1f}</div>
                <div class="metric-label">Avg. Technicians/Day</div>
            </div>
            <div class="metric-item">
                <div class="metric-value">{}</div>
                <div class="metric-label">Max Technicians</div>
            </div>
            <div class="metric-item">
                <div class="metric-value">{}</div>
                <div class="metric-label">Days Over Capacity</div>
            </div>
        </div>
        """.format(avg_techs, max_techs, constraint_days), unsafe_allow_html=True)
        
        # Resource timeline chart
        fig = go.Figure()
        
        # Add bar for technician requirements
        fig.add_trace(go.Bar(
            x=resource_data['RECOMMENDED_DATE'],
            y=resource_data['TOTAL_TECHNICIANS_NEEDED'],
            name="Technicians Needed",
            marker_color=resource_data['TECHNICIAN_UTILIZATION_PERCENTAGE'].apply(
                lambda x: 'red' if x > 100 else 'orange' if x > 80 else 'green'
            )
        ))
        
        # Add capacity line
        fig.add_shape(
            type="line",
            x0=resource_data['RECOMMENDED_DATE'].min(),
            y0=25,  # Max capacity
            x1=resource_data['RECOMMENDED_DATE'].max(),
            y1=25,
            line=dict(color="red", width=2, dash="dash")
        )
        
        # Improve mobile display
        fig.update_layout(
            height=350,
            margin=dict(l=10, r=10, t=30, b=10),
            title="Daily Technician Requirements",
            xaxis_title=None,
            yaxis_title="Technicians",
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show days with resource constraints
        if constraint_days > 0:
            st.subheader("Resource Constraints")
            constraint_dates = resource_data[resource_data['RESOURCE_CONSTRAINT_FLAG']].sort_values('RECOMMENDED_DATE')
            
            for _, day in constraint_dates.iterrows():
                st.markdown(f"""
                <div class="maintenance-card risk-high">
                    <h4>{day['RECOMMENDED_DATE'].strftime('%Y-%m-%d')}</h4>
                    <p><strong>Required:</strong> {day['TOTAL_TECHNICIANS_NEEDED']} technicians (over capacity)</p>
                    <p><strong>Utilization:</strong> {day['TECHNICIAN_UTILIZATION_PERCENTAGE']:.1f}%</p>
                    <p><strong>Events:</strong> {day['MAINTENANCE_EVENTS_COUNT']} maintenance tasks on {day['AIRCRAFT_COUNT']} aircraft</p>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("No resource allocation data available.")

def show_aircraft_view(data):
    """Display aircraft-specific maintenance view"""
    st.subheader("Aircraft Maintenance")
    
    # Get unique aircraft
    if not data.empty and 'AIRCRAFT_REGISTRATION' in data.columns:
        aircraft_list = sorted(data['AIRCRAFT_REGISTRATION'].unique())
        
        # Aircraft selector
        selected_aircraft = st.selectbox("Select Aircraft:", aircraft_list)
        
        # Filter data for selected aircraft
        aircraft_data = data[data['AIRCRAFT_REGISTRATION'] == selected_aircraft]
        
        if not aircraft_data.empty:
            # Show aircraft summary
            aircraft_model = aircraft_data['AIRCRAFT_MODEL'].iloc[0]
            
            st.markdown(f"""
            <div style="background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin-bottom: 15px;">
                <h3>{selected_aircraft}</h3>
                <p>Model: {aircraft_model}</p>
                <p>Upcoming Maintenance: {len(aircraft_data)} items</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Component health visualization
            if 'PERCENT_LIFE_USED' in aircraft_data.columns:
                # Create a horizontal bar chart for component health
                aircraft_data = aircraft_data.sort_values('PERCENT_LIFE_USED', ascending=False)
                
                fig = px.bar(
                    aircraft_data,
                    y='COMPONENT_NAME',
                    x='PERCENT_LIFE_USED',
                    orientation='h',
                    title=f"Component Health Status for {selected_aircraft}",
                    color='PERCENT_LIFE_USED',
                    color_continuous_scale=['green', 'yellow', 'orange', 'red'],
                    range_color=[0, 100],
                    labels={'COMPONENT_NAME': 'Component', 'PERCENT_LIFE_USED': 'Life Used (%)'}
                )
                
                # Add threshold lines
                fig.add_vline(x=90, line_width=1, line_dash="dash", line_color="red")
                fig.add_vline(x=80, line_width=1, line_dash="dash", line_color="orange")
                fig.add_vline(x=70, line_width=1, line_dash="dash", line_color="yellow")
                
                # Optimize for mobile
                fig.update_layout(
                    height=400,
                    margin=dict(l=10, r=10, t=30, b=10),
                    yaxis={'categoryorder': 'total ascending'}
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Show maintenance items as cards
            aircraft_data = aircraft_data.sort_values(['RISK_CATEGORY', 'RECOMMENDED_DATE'])
            
            for _, item in aircraft_data.iterrows():
                risk_class = ""
                if item['RISK_CATEGORY'] == 'Critical Risk':
                    risk_class = "risk-critical"
                elif item['RISK_CATEGORY'] == 'High Risk':
                    risk_class = "risk-high"
                elif item['RISK_CATEGORY'] == 'Medium Risk':
                    risk_class = "risk-medium"
                else:
                    risk_class = "risk-low"
                
                st.markdown(f"""
                <div class="maintenance-card {risk_class}">
                    <h4>{item['COMPONENT_NAME']}</h4>
                    <p><strong>Health:</strong> {item['HEALTH_STATUS']} ({item['PERCENT_LIFE_USED']:.1f}% used)</p>
                    <p><strong>Date:</strong> {item['RECOMMENDED_DATE'].strftime('%Y-%m-%d')}</p>
                    <p><strong>Duration:</strong> {item['DURATION_HOURS']:.1f} hrs • {item['TECHNICIAN_COUNT']} technicians</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info(f"No maintenance scheduled for {selected_aircraft}.")
    else:
        st.info("No aircraft data available.")

if __name__ == "__main__":
    main()
