import os
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from snowflake.snowpark import Session
import datetime
import random

# Page configuration
st.set_page_config(
    page_title="Aircraft Predictive Maintenance",
    page_icon="✈️",
    layout="wide"
)

# Connect to Snowflake
def get_snowflake_session(force_refresh=False):
    """
    Create a Snowflake session using environment variables with automatic refresh
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
        "schema": os.getenv("SNOWFLAKE_SCHEMA", "GOLD"),
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

# Token refresh mechanism
@st.cache_resource(ttl=3600)  # Refresh session every hour
def create_snowflake_session():
    """Cached Snowflake session creation with time-based expiration"""
    return get_snowflake_session()

# Error handling wrapper for database queries with auto-refresh
def safe_snowflake_query(session, query):
    """
    Safely execute Snowflake queries with error handling and session refresh
    """
    try:
        return session.sql(query).to_pandas()
    except Exception as e:
        error_message = str(e)
        # Check if the error is related to authentication token expiration
        if "Authentication token has expired" in error_message or "authentication" in error_message.lower():
            st.warning("Refreshing Snowflake session... Please wait.")
            
            # Clear the cached session to force creation of a new one
            create_snowflake_session.clear()
            
            # Get a fresh session
            new_session = create_snowflake_session()
            
            # Retry the query with the new session
            try:
                return new_session.sql(query).to_pandas()
            except Exception as retry_error:
                st.error(f"Failed to execute query after session refresh: {retry_error}")
                return pd.DataFrame()
        else:
            st.error(f"Database query failed: {e}")
            return pd.DataFrame()

def main():
    # Title and introduction
    st.title("Aircraft Predictive Maintenance Dashboard")
    st.markdown("*Powered by Snowflake | Analytics Dashboard for Aircraft Manufacturer*")

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
        # Try to get airlines from the master table if the view doesn't have data yet
        return _session.sql("select distinct airline_name from aircraft_reference.master.airlines").to_pandas()
    
    airlines = get_airlines(session)
    selected_airline = st.sidebar.selectbox("Select Airline", airlines["AIRLINE_NAME"])
    
    # Simplified KPIs at top
    @st.cache_data(ttl=600)
    def get_kpi_data(_session, airline):
        kpi_query = f"""
            select 
                count(distinct a.aircraft_id) as fleet_size,
                0 as components_needing_maintenance,
                0 as critical_components,
                0 as potential_savings_usd
            from aircraft_reference.master.manufacturer_aircraft a
            join aircraft_reference.master.airlines air on a.airline_id = air.airline_id
            where air.airline_name = '{airline}'
        """
        return safe_snowflake_query(_session, kpi_query)
    
    kpi_data = get_kpi_data(session, selected_airline)

    # KPI cards in a row
    col1, col2, col3, col4 = st.columns(4)
    
    # Check if kpi_data has data before accessing it
    if not kpi_data.empty:
        with col1:
            st.metric("Fleet Size", kpi_data["FLEET_SIZE"].iloc[0])
        with col2:
            st.metric("Components Needing Attention", kpi_data["COMPONENTS_NEEDING_MAINTENANCE"].iloc[0])
        with col3:
            st.metric("Critical Risk Components", kpi_data["CRITICAL_COMPONENTS"].iloc[0])
        with col4:
            savings_value = kpi_data["POTENTIAL_SAVINGS_USD"].iloc[0]
            if pd.notna(savings_value):
                st.metric("Potential Savings", f"${savings_value:,.2f}")
            else:
                st.metric("Potential Savings", "$0")
    else:
        st.warning("No KPI data available for selected airline.")

    # PRIMARY VISUALIZATION: Optimal Maintenance Schedule
    st.header("Optimal Maintenance Schedule")
    st.markdown("This visualization shows the recommended maintenance timeline across the fleet, optimizing for minimal downtime and resource utilization.")
    
    # Add optimization controls
    optimization_col1, optimization_col2, optimization_col3 = st.columns(3)

    with optimization_col1:
        downtime_weight = st.slider("Weight: Minimize Downtime", 1, 10, 7)

    with optimization_col2:
        urgency_weight = st.slider("Weight: Component Urgency", 1, 10, 8) 

    with optimization_col3:
        resource_weight = st.slider("Weight: Resource Utilization", 1, 10, 5)

    # Get maintenance schedule data
    @st.cache_data(ttl=60)  # Short cache time since this depends on sliders
    def get_schedule_data(_session, airline, downtime_weight=5, urgency_weight=7, resource_weight=3):
        try:
            # First check if the view exists with a simple count query
            check_query = f"""
                select count(*) as count from gold.optimal_maintenance_schedule
                where airline_name = '{airline}'
            """
            check_result = safe_snowflake_query(_session, check_query)
            
            if check_result.empty or check_result.iloc[0]['COUNT'] == 0:
                # If no data in view, use a dummy dataset for demo purposes
                st.warning(f"No maintenance schedule data found for {airline}. Using simulated data for demonstration.")
                
                # Get aircraft for this airline
                aircraft_query = f"""
                    select 
                        a.aircraft_id, 
                        a.registration as aircraft_registration,
                        a.model_name as aircraft_model
                    from aircraft_reference.master.manufacturer_aircraft a
                    join aircraft_reference.master.airlines air on a.airline_id = air.airline_id
                    where air.airline_name = '{airline}'
                    limit 5
                """
                aircraft_data = safe_snowflake_query(_session, aircraft_query)
                
                if aircraft_data.empty:
                    return pd.DataFrame()
                
                # Create dummy schedule data
                import numpy as np
                
                # Get component types
                component_query = """
                    select 
                        component_type_id,
                        component_name,
                        component_category
                    from aircraft_reference.master.component_types
                    limit 10
                """
                component_types = safe_snowflake_query(_session, component_query)
                
                if component_types.empty:
                    return pd.DataFrame()
                
                # Create simulated data
                rows = []
                component_id = 10001
                risk_categories = ["Critical Risk", "High Risk", "Medium Risk", "Low Risk"]
                
                for i, aircraft in aircraft_data.iterrows():
                    # Add 3-5 components per aircraft
                    for j in range(np.random.randint(3, 6)):
                        comp_type = component_types.iloc[np.random.randint(0, len(component_types))]
                        
                        # Add simulated component
                        can_bundle = np.random.random() < 0.6  # 60% chance to bundle
                        
                        rows.append({
                            'COMPONENT_ID': component_id,
                            'COMPONENT_NAME': f"{comp_type['COMPONENT_NAME']} {j+1}",
                            'AIRCRAFT_ID': aircraft['AIRCRAFT_ID'],
                            'AIRCRAFT_REGISTRATION': aircraft['AIRCRAFT_REGISTRATION'],
                            'AIRCRAFT_MODEL': aircraft['AIRCRAFT_MODEL'],
                            'COMPONENT_CATEGORY': comp_type['COMPONENT_CATEGORY'],
                            'PERCENT_LIFE_USED': np.random.randint(60, 95),
                            'RISK_CATEGORY': np.random.choice(risk_categories, p=[0.2, 0.3, 0.3, 0.2]),
                            'RECOMMENDED_DATE': pd.Timestamp('today') + pd.Timedelta(days=np.random.randint(1, 45)),
                            'MAINTENANCE_TYPE': f"Scheduled {comp_type['COMPONENT_NAME']} Service",
                            'MAINTENANCE_CODE': f"M{np.random.randint(100, 999)}",
                            'DURATION_HOURS': np.random.randint(4, 48),
                            'TECHNICIAN_COUNT': np.random.randint(1, 5),
                            'CAN_BE_BUNDLED': can_bundle,
                            'BUNDLED_COMPONENTS_COUNT': np.random.randint(2, 4) if can_bundle else 1,
                            'BUNDLED_COMPONENTS': f"Multiple {comp_type['COMPONENT_CATEGORY']} components" if can_bundle else None,
                        })
                        component_id += 1
                
                dummy_data = pd.DataFrame(rows)
                
                # Add priority score
                dummy_data['PRIORITY_SCORE'] = dummy_data.apply(
                    lambda x: (urgency_weight * 10 if x['RISK_CATEGORY'] == 'Critical Risk' else
                              urgency_weight * 6 if x['RISK_CATEGORY'] == 'High Risk' else
                              urgency_weight * 3 if x['RISK_CATEGORY'] == 'Medium Risk' else
                              urgency_weight * 1) +
                             (resource_weight * 8 if x['BUNDLED_COMPONENTS_COUNT'] > 2 else
                              resource_weight * 5 if x['BUNDLED_COMPONENTS_COUNT'] == 2 else 0) +
                             (downtime_weight * 8 if x['DURATION_HOURS'] < 24 else
                              downtime_weight * 5 if x['DURATION_HOURS'] < 48 else
                              downtime_weight * 2),
                    axis=1
                )
                
                # Add maintenance group flag
                dummy_data['MAINTENANCE_GROUP'] = dummy_data['CAN_BE_BUNDLED'].apply(
                    lambda x: 'Bundled' if x else 'Individual'
                )
                
                return dummy_data
            
            # If data exists, use the real query
            schedule_query = f"""
                select *,
                    -- Calculate priority score based on weights
                    case
                        when risk_category = 'Critical Risk' then {urgency_weight} * 10
                        when risk_category = 'High Risk' then {urgency_weight} * 6
                        when risk_category = 'Medium Risk' then {urgency_weight} * 3
                        else {urgency_weight} * 1
                    end +
                    case
                        when bundled_components_count > 2 then {resource_weight} * 8
                        when bundled_components_count = 2 then {resource_weight} * 5
                        else 0
                    end +
                    case
                        when duration_hours < 24 then {downtime_weight} * 8
                        when duration_hours < 48 then {downtime_weight} * 5
                        else {downtime_weight} * 2
                    end as priority_score,
                    case 
                        when can_be_bundled then 'Bundled'
                        else 'Individual'
                    end as maintenance_group
                from gold.optimal_maintenance_schedule
                where airline_name = '{airline}'
                order by priority_score desc, recommended_date
            """
            return safe_snowflake_query(_session, schedule_query)
        except Exception as e:
            st.error(f"Error getting schedule data: {e}")
            return pd.DataFrame()
    
    schedule_data = get_schedule_data(session, selected_airline)

    # Create the enhanced Gantt chart
    if not schedule_data.empty:
        # Add columns for resource info display
        resource_col1, resource_col2 = st.columns(2)
        
        with resource_col1:
            # Show resource utilization chart
            if 'RECOMMENDED_DATE' in schedule_data.columns and 'TECHNICIAN_COUNT' in schedule_data.columns:
                resources_by_day = schedule_data.groupby('RECOMMENDED_DATE').agg({
                    'TECHNICIAN_COUNT': 'sum',
                    'COMPONENT_ID': 'count'
                }).reset_index()
                
                resources_by_day.rename(columns={'COMPONENT_ID': 'MAINTENANCE_COUNT'}, inplace=True)
                
                # Create figure with secondary y-axis
                fig_resources = go.Figure()
                
                # Add bars for technician count
                fig_resources.add_trace(go.Bar(
                    x=resources_by_day['RECOMMENDED_DATE'],
                    y=resources_by_day['TECHNICIAN_COUNT'],
                    name='Technicians Needed',
                    marker_color='#1f77b4'
                ))
                
                # Add line for maintenance count
                fig_resources.add_trace(go.Scatter(
                    x=resources_by_day['RECOMMENDED_DATE'],
                    y=resources_by_day['MAINTENANCE_COUNT'],
                    name='Maintenance Events',
                    mode='lines+markers',
                    marker=dict(color='#ff7f0e'),
                    line=dict(width=3),
                    yaxis='y2'
                ))
                
                # Add capacity line
                max_daily_capacity = 25  # This would come from your data in practice
                fig_resources.add_shape(
                    type="line",
                    x0=resources_by_day['RECOMMENDED_DATE'].min(),
                    y0=max_daily_capacity,
                    x1=resources_by_day['RECOMMENDED_DATE'].max(),
                    y1=max_daily_capacity,
                    line=dict(color="red", width=2, dash="dash")
                )
                
                fig_resources.add_annotation(
                    x=resources_by_day['RECOMMENDED_DATE'].iloc[len(resources_by_day)//2],
                    y=max_daily_capacity,
                    text="Max Technician Capacity",
                    showarrow=False,
                    yshift=10
                )
                
                # Update layout with second y-axis
                fig_resources.update_layout(
                    title="Resource Requirements by Day",
                    xaxis_title="Date",
                    yaxis_title="Technicians Needed",
                    yaxis2=dict(
                        title=dict(
                            text="Maintenance Events",
                            font=dict(color='#ff7f0e')
                        ),
                        tickfont=dict(color='#ff7f0e'),
                        overlaying='y',
                        side='right'
                    ),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                st.plotly_chart(fig_resources, use_container_width=True)
            else:
                st.warning("Resource utilization data not available in the schedule.")
        
        with resource_col2:
            # Show maintenance stats
            st.subheader("Maintenance Optimization")
            
            # Display summary metrics
            total_events = len(schedule_data)
            bundled_events = schedule_data[schedule_data['CAN_BE_BUNDLED'] == True].shape[0]
            unbundled_events = total_events - bundled_events
            
            # Calculate total technician hours
            if 'DURATION_HOURS' in schedule_data.columns and 'TECHNICIAN_COUNT' in schedule_data.columns:
                total_tech_hours = int(schedule_data['DURATION_HOURS'].sum() * schedule_data['TECHNICIAN_COUNT'].sum())
                
                # Calculate the non-optimized hours (if each was done individually)
                non_optimized_tech_hours = total_tech_hours * 1.3  # Assuming 30% efficiency gain from optimization
                
                # Tech hours saved
                tech_hours_saved = non_optimized_tech_hours - total_tech_hours
                
                # Cost savings (assuming $85/hour technician cost)
                cost_savings = tech_hours_saved * 85
            else:
                total_tech_hours = 0
                tech_hours_saved = 0
                cost_savings = 0
            
            # Metrics
            metric_col1, metric_col2 = st.columns(2)
            with metric_col1:
                st.metric("Total Maintenance Events", total_events)
                st.metric("Bundled Events", bundled_events, f"+{bundled_events} efficiency")
            
            with metric_col2:
                st.metric("Technician-Hours", total_tech_hours)
                st.metric("Hours Saved Through Bundling", int(tech_hours_saved), f"${cost_savings:,.0f}")
            
            # Pie chart of maintenance by risk category
            if not schedule_data.empty and 'RISK_CATEGORY' in schedule_data.columns:
                risk_counts = schedule_data['RISK_CATEGORY'].value_counts().reset_index()
                risk_counts.columns = ['RISK_CATEGORY', 'COUNT']
                
                fig_risk_pie = px.pie(
                    risk_counts, 
                    values='COUNT', 
                    names='RISK_CATEGORY',
                    title='Maintenance by Risk Category',
                    color='RISK_CATEGORY',
                    color_discrete_map={
                        "Critical Risk": "#FF2B2B",
                        "High Risk": "#FF9E2D",
                        "Medium Risk": "#FFDF3C",
                        "Low Risk": "#6ECB63"
                    }
                )
                fig_risk_pie.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_risk_pie, use_container_width=True)
        
        # Process Gantt chart data
        if 'RECOMMENDED_DATE' in schedule_data.columns:
            # Create layout for full-width Gantt chart
            st.markdown("### Maintenance Schedule Timeline")
            
            # Convert to datetime if not already
            schedule_data['RECOMMENDED_DATE'] = pd.to_datetime(schedule_data['RECOMMENDED_DATE'])
            
            # Add end date for Gantt chart (duration-based)
            schedule_data['END_DATE'] = schedule_data.apply(
                lambda x: x['RECOMMENDED_DATE'] + pd.Timedelta(hours=float(x['DURATION_HOURS'])), 
                axis=1
            )
            
            # Enhanced Gantt chart
            fig_gantt = px.timeline(
                schedule_data, 
                x_start="RECOMMENDED_DATE", 
                x_end="END_DATE", 
                y="AIRCRAFT_REGISTRATION",
                color="RISK_CATEGORY",
                hover_name="COMPONENT_NAME",
                hover_data=["DURATION_HOURS", "TECHNICIAN_COUNT", "MAINTENANCE_TYPE", "BUNDLED_COMPONENTS"],
                color_discrete_map={
                    "Critical Risk": "#FF2B2B",
                    "High Risk": "#FF9E2D",
                    "Medium Risk": "#FFDF3C",
                    "Low Risk": "#6ECB63"
                },
                pattern_shape="MAINTENANCE_GROUP",
                pattern_shape_map={"Bundled": "", "Individual": ""},
                labels={
                    "RECOMMENDED_DATE": "Start Date",
                    "END_DATE": "End Date",
                    "AIRCRAFT_REGISTRATION": "Aircraft",
                    "COMPONENT_NAME": "Component"
                }
            )
            
            # Improve visual presentation
            fig_gantt.update_yaxes(autorange="reversed")
            fig_gantt.update_layout(
                height=600,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                title="Optimal Maintenance Schedule by Aircraft"
            )
            
            # Add vertical line for today
            today = pd.to_datetime('today')
            fig_gantt.add_vline(x=today, line_width=2, line_dash="solid", line_color="black")
            fig_gantt.add_annotation(x=today, y=0, text="Today", showarrow=False, yshift=10)
            
            # Show component bundling with custom data points
            for aircraft in schedule_data['AIRCRAFT_REGISTRATION'].unique():
                aircraft_data = schedule_data[schedule_data['AIRCRAFT_REGISTRATION'] == aircraft]
                bundled_dates = aircraft_data[aircraft_data['CAN_BE_BUNDLED']]['RECOMMENDED_DATE'].unique()
                
                for date in bundled_dates:
                    bundled_group = aircraft_data[
                        (aircraft_data['RECOMMENDED_DATE'] == date) & 
                        (aircraft_data['CAN_BE_BUNDLED'])
                    ]
                    
                    if len(bundled_group) > 1:
                        # Add highlight indicator for bundled maintenance
                        # Using a transparent rectangle to highlight bundled maintenance period
                        fig_gantt.add_shape(
                            type="rect",
                            x0=bundled_group['RECOMMENDED_DATE'].min(),
                            y0=aircraft,
                            x1=bundled_group['END_DATE'].max(),
                            y1=aircraft,
                            line=dict(color="rgba(0,0,0,0)"),
                            fillcolor="rgba(65,105,225,0.2)",
                            layer="below"
                        )
            
            st.plotly_chart(fig_gantt, use_container_width=True)
            
            # Add table of upcoming maintenance
            with st.expander("View Upcoming Maintenance Details"):
                upcoming = schedule_data[schedule_data['RECOMMENDED_DATE'] >= today].sort_values('RECOMMENDED_DATE').head(10)
                
                if not upcoming.empty:
                    upcoming_display = upcoming[[
                        'COMPONENT_NAME', 'RISK_CATEGORY', 'AIRCRAFT_REGISTRATION', 
                        'RECOMMENDED_DATE', 'DURATION_HOURS', 'MAINTENANCE_TYPE'
                    ]].copy()
                    
                    # Format date for display
                    upcoming_display['RECOMMENDED_DATE'] = upcoming_display['RECOMMENDED_DATE'].dt.strftime('%Y-%m-%d')
                    
                    st.dataframe(
                        upcoming_display,
                        column_config={
                            "COMPONENT_NAME": "Component",
                            "RISK_CATEGORY": st.column_config.TextColumn("Risk Level"),
                            "AIRCRAFT_REGISTRATION": "Aircraft",
                            "RECOMMENDED_DATE": "Date",
                            "DURATION_HOURS": st.column_config.NumberColumn("Duration (hrs)", format="%.1f"),
                            "MAINTENANCE_TYPE": "Type"
                        },
                        use_container_width=True
                    )
        else:
            st.warning("Missing required date columns in schedule data.")
    else:
        st.warning("No maintenance schedule data available. Please check your data or Snowflake connection.")

    # SECONDARY VISUALIZATION: Maintenance ROI Analysis
    st.header("Maintenance ROI Analysis")
    st.markdown("This analysis compares costs between reactive (unplanned) maintenance and predictive (planned) maintenance, showing potential savings.")
    
    @st.cache_data(ttl=600)
    def get_roi_data(_session, airline):
        try:
            # First check if the view exists and has data
            check_query = f"""
                select count(*) as count from gold.maintenance_roi_analysis
                where airline_name = '{airline}'
            """
            check_result = safe_snowflake_query(_session, check_query)
            
            if check_result.empty or check_result.iloc[0]['COUNT'] == 0:
                # If no data in view, use dummy data for demo
                st.warning(f"No ROI data found for {airline}. Using simulated data for demonstration.")
                
                # Create dummy ROI data
                component_categories = [
                    "Propulsion", "Structural", "Power", "Control", "Electronics"
                ]
                
                rows = []
                for category in component_categories:
                    reactive_cost = random.uniform(500000, 2000000)
                    predictive_cost = reactive_cost * random.uniform(0.4, 0.7)
                    savings = reactive_cost - predictive_cost
                    
                    rows.append({
                        'COMPONENT_CATEGORY': category,
                        'REACTIVE_COST': reactive_cost,
                        'PREDICTIVE_COST': predictive_cost,
                        'SAVINGS': savings,
                        'ROI_PERCENTAGE': (savings / predictive_cost) * 100,
                        'DOWNTIME_HOURS_SAVED': random.uniform(20, 120),
                        'FLIGHTS_SAVED': random.uniform(2, 15)
                    })
                
                return pd.DataFrame(rows)
            
            # If data exists, use the real query
            roi_query = f"""
                select 
                    component_category,
                    sum(reactive_maintenance_cost_usd) as reactive_cost,
                    sum(predictive_maintenance_cost_usd) as predictive_cost,
                    sum(potential_savings_usd) as savings,
                    avg(roi_percentage) as roi_percentage,
                    sum(downtime_hours_saved) as downtime_hours_saved,
                    sum(estimated_flights_saved) as flights_saved
                from gold.maintenance_roi_analysis
                where airline_name = '{airline}'
                group by component_category
            """
            return safe_snowflake_query(_session, roi_query)
        except Exception as e:
            st.error(f"Error getting ROI data: {e}")
            return pd.DataFrame()
    
    roi_data = get_roi_data(session, selected_airline)
    
    # Create two columns for cost visualizations
    roi_col1, roi_col2 = st.columns(2)
    
    with roi_col1:
        if not roi_data.empty:
            # Create a stacked bar chart for reactive vs predictive by component
            comp_cost_data = pd.DataFrame({
                'Component Category': roi_data['COMPONENT_CATEGORY'].repeat(2),
                'Maintenance Type': ['Reactive', 'Predictive'] * len(roi_data),
                'Cost (USD)': list(roi_data['REACTIVE_COST']) + list(roi_data['PREDICTIVE_COST'])
            })
            
            fig_cost = px.bar(
                comp_cost_data,
                x='Component Category',
                y='Cost (USD)',
                color='Maintenance Type',
                barmode='group',
                title='Cost Comparison: Reactive vs. Predictive Maintenance',
                color_discrete_map={
                    'Reactive': '#FF5A5F',
                    'Predictive': '#2E86C1'
                }
            )
            
            # Add savings annotations
            for i, row in roi_data.iterrows():
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
    
    with roi_col2:
        if not roi_data.empty:
            # Calculate total savings and operational impact
            total_savings = roi_data['SAVINGS'].sum()
            total_downtime_saved = roi_data['DOWNTIME_HOURS_SAVED'].sum()
            total_flights_saved = roi_data['FLIGHTS_SAVED'].sum()
            
            # Create a waterfall chart showing business impact
            impact_categories = ['Maintenance Costs', 'Crew Costs', 'Opportunity Costs', 'Total Savings']
            
            # Calculate component values (simplified model)
            maintenance_savings = total_savings * 0.6  # 60% of savings from direct maintenance
            crew_costs = total_savings * 0.15        # 15% from avoiding crew overtime/costs
            opportunity_costs = total_savings * 0.25  # 25% from avoiding lost revenue
            
            impact_values = [maintenance_savings, crew_costs, opportunity_costs, total_savings]
            
            # Create data for the waterfall chart
            waterfall_data = pd.DataFrame({
                'Category': impact_categories,
                'Value': impact_values,
                'Type': ['Positive', 'Positive', 'Positive', 'Total']
            })
            
            fig_impact = go.Figure(go.Waterfall(
                name="Business Impact", 
                orientation="v",
                measure=waterfall_data['Type'],
                x=waterfall_data['Category'],
                y=waterfall_data['Value'],
                connector={"line":{"color":"rgb(63, 63, 63)"}},
                decreasing={"marker":{"color":"#FF5A5F"}},
                increasing={"marker":{"color":"#2E86C1"}},
                totals={"marker":{"color":"#28B463"}}
            ))
            
            fig_impact.update_layout(
                title="Business Impact of Predictive Maintenance",
                showlegend=False
            )
            
            st.plotly_chart(fig_impact, use_container_width=True)
            
            # Key metrics for operational impact
            op_col1, op_col2, op_col3 = st.columns(3)
            
            with op_col1:
                st.metric("Total Cost Savings", f"${total_savings:,.0f}")
            
            with op_col2:
                st.metric("Downtime Hours Avoided", f"{total_downtime_saved:,.0f} hrs")
            
            with op_col3:
                st.metric("Potential Flights Saved", f"{total_flights_saved:,.1f}")
        else:
            st.warning("No impact data available for selected airline.")

    # Footer
    st.markdown("---")
    st.caption("Aircraft Predictive Maintenance Demo - Powered by Snowflake")

    # Add a disclaimer
    st.markdown("---")
    st.caption("🔒 Data is anonymized and secured | For demonstration purposes only")

if __name__ == "__main__":
    main()
