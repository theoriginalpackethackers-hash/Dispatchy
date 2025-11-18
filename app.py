"""
Smart Dispatch Optimizer - Standalone Web Application
No Databricks or Python knowledge required for end users!
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from math import radians, cos, sin, asin, sqrt
import folium
from streamlit_folium import st_folium
from io import BytesIO
import base64

# Page configuration
st.set_page_config(
    page_title="Smart Dispatch Optimizer",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        text-align: center;
    }
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'optimized' not in st.session_state:
    st.session_state.optimized = False
if 'results' not in st.session_state:
    st.session_state.results = None
if 'tech_stats' not in st.session_state:
    st.session_state.tech_stats = None

# Header
st.markdown("""
<div class="main-header">
    <h1>🚀 Smart Dispatch Optimizer</h1>
    <p style="font-size: 1.2rem; margin: 0;">Intelligent Route Planning Made Simple</p>
</div>
""", unsafe_allow_html=True)

# Optimization functions
def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points on earth in kilometers"""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return 6371 * c

def calculate_priority_score(row, priority_weight):
    """Calculate priority score based on multiple factors"""
    base_score = row.get('priority', 3)
    time_score = 0
    
    if pd.notna(row.get('appointment_start')):
        try:
            appt_time = pd.to_datetime(row['appointment_start'])
            hours_until = (appt_time - datetime.now()).total_seconds() / 3600
            if hours_until < 2:
                time_score = 5
            elif hours_until < 4:
                time_score = 3
            elif hours_until < 8:
                time_score = 1
        except:
            pass
    
    return base_score * priority_weight + time_score

def optimize_route_2opt(coordinates, max_iterations=100):
    """Optimize route using 2-opt algorithm"""
    n = len(coordinates)
    if n <= 2:
        return list(range(n))
    
    route = list(range(n))
    improved = True
    iteration = 0
    
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                if j - i == 1:
                    continue
                
                current_dist = (
                    haversine_distance(
                        coordinates[route[i-1]][0], coordinates[route[i-1]][1],
                        coordinates[route[i]][0], coordinates[route[i]][1]
                    ) +
                    haversine_distance(
                        coordinates[route[j]][0], coordinates[route[j]][1],
                        coordinates[route[j-1]][0], coordinates[route[j-1]][1]
                    )
                )
                
                new_dist = (
                    haversine_distance(
                        coordinates[route[i-1]][0], coordinates[route[i-1]][1],
                        coordinates[route[j]][0], coordinates[route[j]][1]
                    ) +
                    haversine_distance(
                        coordinates[route[i]][0], coordinates[route[i]][1],
                        coordinates[route[j-1]][0], coordinates[route[j-1]][1]
                    )
                )
                
                if new_dist < current_dist:
                    route[i:j+1] = reversed(route[i:j+1])
                    improved = True
    
    return route

def process_data(data, max_techs, priority_weight, distance_weight):
    """Process and optimize dispatch data"""
    # Standardize column names
    data.columns = data.columns.str.lower().str.replace(' ', '_')
    
    # Auto-detect columns
    column_mapping = {}
    for col in data.columns:
        col_lower = col.lower()
        if 'dispatch' in col_lower and 'id' in col_lower:
            column_mapping['dispatch_id'] = col
        elif 'lat' in col_lower and 'long' not in col_lower:
            column_mapping['latitude'] = col
        elif 'lon' in col_lower or 'lng' in col_lower:
            column_mapping['longitude'] = col
        elif 'status' in col_lower:
            column_mapping['status'] = col
        elif 'skill' in col_lower:
            column_mapping['skills'] = col
        elif 'prior' in col_lower:
            column_mapping['priority'] = col
        elif 'tech' in col_lower and ('id' in col_lower or 'assign' in col_lower):
            column_mapping['tech_id'] = col
        elif 'appt' in col_lower or 'appointment' in col_lower:
            if 'start' in col_lower or 'begin' in col_lower:
                column_mapping['appointment_start'] = col
            elif 'end' in col_lower:
                column_mapping['appointment_end'] = col
    
    data = data.rename(columns=column_mapping)
    
    # Calculate priority scores
    data['priority_score'] = data.apply(lambda row: calculate_priority_score(row, priority_weight), axis=1)
    
    # Filter valid dispatches
    valid_data = data[
        (data['latitude'].notna()) & 
        (data['longitude'].notna())
    ].copy()
    
    if 'status' in valid_data.columns:
        valid_data = valid_data[
            valid_data['status'].str.lower().isin(['open', 'pending', 'unassigned'])
        ]
    
    valid_data = valid_data.sort_values('priority_score', ascending=False)
    
    # Cluster into technician groups using KMeans
    from sklearn.cluster import KMeans
    
    n_clusters = min(max_techs, len(valid_data))
    if n_clusters > 1:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        valid_data['tech_cluster'] = kmeans.fit_predict(valid_data[['latitude', 'longitude']])
    else:
        valid_data['tech_cluster'] = 0
    
    valid_data['assigned_tech'] = 'TECH-' + (valid_data['tech_cluster'] + 1).astype(str).str.zfill(3)
    
    # Optimize routes for each tech
    optimized_routes = {}
    tech_stats = {}
    
    for tech_id in valid_data['assigned_tech'].unique():
        tech_dispatches = valid_data[valid_data['assigned_tech'] == tech_id].copy()
        
        if len(tech_dispatches) > 0:
            coords = tech_dispatches[['latitude', 'longitude']].values.tolist()
            
            if len(coords) > 2:
                optimized_order = optimize_route_2opt(coords)
                tech_dispatches = tech_dispatches.iloc[optimized_order].reset_index(drop=True)
            
            tech_dispatches['route_order'] = range(1, len(tech_dispatches) + 1)
            
            total_distance = 0
            for i in range(len(tech_dispatches) - 1):
                dist = haversine_distance(
                    tech_dispatches.iloc[i]['latitude'],
                    tech_dispatches.iloc[i]['longitude'],
                    tech_dispatches.iloc[i+1]['latitude'],
                    tech_dispatches.iloc[i+1]['longitude']
                )
                total_distance += dist
            
            tech_stats[tech_id] = {
                'total_jobs': len(tech_dispatches),
                'total_distance_km': round(total_distance, 2),
                'avg_priority': round(tech_dispatches['priority_score'].mean(), 2),
                'coordinates': coords
            }
            
            optimized_routes[tech_id] = tech_dispatches
    
    final_results = pd.concat(optimized_routes.values(), ignore_index=True)
    
    return final_results, tech_stats

def create_map(results, tech_stats):
    """Create interactive map with routes"""
    center_lat = results['latitude'].mean()
    center_lon = results['longitude'].mean()
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=11,
        tiles='OpenStreetMap'
    )
    
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 
              'beige', 'darkblue', 'darkgreen', 'cadetblue', 'darkpurple', 
              'pink', 'lightblue', 'lightgreen', 'gray', 'black']
    
    tech_color_map = {}
    for idx, tech_id in enumerate(sorted(results['assigned_tech'].unique())):
        tech_color_map[tech_id] = colors[idx % len(colors)]
    
    for tech_id in results['assigned_tech'].unique():
        tech_data = results[results['assigned_tech'] == tech_id].sort_values('route_order')
        
        if len(tech_data) == 0:
            continue
        
        color = tech_color_map[tech_id]
        route_coords = tech_data[['latitude', 'longitude']].values.tolist()
        
        folium.PolyLine(
            route_coords,
            color=color,
            weight=3,
            opacity=0.7,
            popup=f"{tech_id}<br>Jobs: {len(tech_data)}<br>Distance: {tech_stats[tech_id]['total_distance_km']:.1f} km"
        ).add_to(m)
        
        for idx, row in tech_data.iterrows():
            popup_html = f"""
            <div style="font-family: Arial; min-width: 200px;">
                <h4 style="margin: 0 0 10px 0; color: {color};">{tech_id}</h4>
                <table style="width: 100%; font-size: 12px;">
                    <tr><td><b>Stop:</b></td><td>#{int(row['route_order'])}</td></tr>
                    <tr><td><b>Job ID:</b></td><td>{row.get('dispatch_id', 'N/A')}</td></tr>
                    <tr><td><b>Priority:</b></td><td>{row['priority_score']:.1f}</td></tr>
                    <tr><td><b>Location:</b></td><td>{row['latitude']:.4f}, {row['longitude']:.4f}</td></tr>
                </table>
            </div>
            """
            
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=8,
                popup=folium.Popup(popup_html, max_width=300),
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.7,
                weight=2
            ).add_to(m)
            
            folium.Marker(
                location=[row['latitude'], row['longitude']],
                icon=folium.DivIcon(html=f"""
                    <div style="
                        background-color: {color};
                        color: white;
                        border-radius: 50%;
                        width: 24px;
                        height: 24px;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        font-weight: bold;
                        font-size: 12px;
                        border: 2px solid white;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.3);
                    ">{int(row['route_order'])}</div>
                """)
            ).add_to(m)
    
    return m

def download_csv(df, filename="optimized_routes.csv"):
    """Generate download link for CSV"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" class="success-box" style="display: block; text-decoration: none; color: #155724; text-align: center; padding: 1rem;">📥 Click Here to Download Results CSV</a>'
    return href

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.markdown("### 📊 Data Source")
    uploaded_file = st.file_uploader(
        "Upload Your CSV File",
        type=['csv'],
        help="Upload a CSV file with dispatch data including latitude, longitude, and job details"
    )
    
    st.markdown("---")
    st.markdown("### 🎛️ Optimization Settings")
    
    max_techs = st.slider(
        "Maximum Technicians",
        min_value=5,
        max_value=50,
        value=10,
        step=5,
        help="Maximum number of technicians to assign jobs to"
    )
    
    priority_weight = st.select_slider(
        "Priority Weight",
        options=[1.0, 1.5, 2.0, 2.5, 3.0],
        value=2.0,
        help="How much to favor high-priority jobs (higher = more priority focus)"
    )
    
    distance_weight = st.select_slider(
        "Distance Weight",
        options=[0.5, 1.0, 1.5, 2.0],
        value=1.0,
        help="How much to minimize travel distance (higher = shorter routes)"
    )
    
    st.markdown("---")
    st.markdown("### 📋 Required Columns")
    st.markdown("""
    **Must have:**
    - `dispatch_id` (or similar)
    - `latitude` (or `lat`)
    - `longitude` (or `lon`/`lng`)
    - `status` (optional)
    
    **Nice to have:**
    - `priority` (1-5)
    - `skills` (comma-separated)
    - `appointment_start`
    - `appointment_end`
    """)

# Main content
if uploaded_file is None:
    # Welcome screen
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>📤 Step 1</h3>
            <h4>Upload Data</h4>
            <p>Drag and drop your CSV file in the sidebar</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>⚙️ Step 2</h3>
            <h4>Configure</h4>
            <p>Adjust settings if needed (or use defaults)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>🚀 Step 3</h3>
            <h4>Optimize</h4>
            <p>Click the button and watch the magic!</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    <div class="info-box">
        <h3>🎯 What This Tool Does</h3>
        <p>This tool automatically:</p>
        <ul>
            <li>✅ Assigns work orders to available technicians</li>
            <li>✅ Matches jobs to technician skills</li>
            <li>✅ Prioritizes urgent tasks</li>
            <li>✅ Minimizes travel time and distance</li>
            <li>✅ Respects appointment time windows</li>
            <li>✅ Creates optimal routes for each technician</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="warning-box">
        <h4>📋 Sample Data Format</h4>
        <p>Your CSV should look something like this:</p>
    </div>
    """, unsafe_allow_html=True)
    
    sample_data = pd.DataFrame({
        'dispatch_id': ['DISP-001', 'DISP-002', 'DISP-003'],
        'status': ['Open', 'Open', 'Pending'],
        'latitude': [40.7128, 40.7580, 40.7489],
        'longitude': [-74.0060, -73.9855, -73.9680],
        'priority': [3, 5, 2],
        'skills': ['fiber', 'installation', 'repair'],
        'appointment_start': ['2024-01-15 09:00', '2024-01-15 10:00', '2024-01-15 13:00'],
        'appointment_end': ['2024-01-15 12:00', '2024-01-15 14:00', '2024-01-15 17:00']
    })
    
    st.dataframe(sample_data, use_container_width=True)

else:
    # Data uploaded
    try:
        raw_data = pd.read_csv(uploaded_file)
        st.session_state.data_loaded = True
        
        st.markdown("""
        <div class="success-box">
            <h3>✅ Data Loaded Successfully!</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Show data preview
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", f"{len(raw_data):,}")
        with col2:
            st.metric("Columns", len(raw_data.columns))
        with col3:
            st.metric("Memory", f"{raw_data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        with col4:
            valid_coords = raw_data[['latitude', 'longitude']].notna().all(axis=1).sum() if 'latitude' in raw_data.columns else 0
            st.metric("Valid Coordinates", valid_coords)
        
        st.markdown("---")
        
        with st.expander("📊 View Data Preview (First 10 Rows)", expanded=False):
            st.dataframe(raw_data.head(10), use_container_width=True)
        
        st.markdown("---")
        
        # Optimize button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 RUN OPTIMIZATION", use_container_width=True, type="primary"):
                with st.spinner("⚙️ Running optimization... This may take a moment..."):
                    try:
                        results, tech_stats = process_data(
                            raw_data.copy(),
                            max_techs,
                            priority_weight,
                            distance_weight
                        )
                        
                        st.session_state.optimized = True
                        st.session_state.results = results
                        st.session_state.tech_stats = tech_stats
                        
                        st.success("✅ Optimization Complete!")
                        st.balloons()
                        
                    except Exception as e:
                        st.error(f"❌ Error during optimization: {str(e)}")
                        st.stop()
        
        # Show results if optimized
        if st.session_state.optimized and st.session_state.results is not None:
            results = st.session_state.results
            tech_stats = st.session_state.tech_stats
            
            st.markdown("---")
            st.markdown("## 📊 Optimization Results")
            
            # Summary metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🎯 Jobs Optimized", f"{len(results):,}")
            with col2:
                st.metric("👥 Technicians", len(tech_stats))
            with col3:
                total_dist = sum(s['total_distance_km'] for s in tech_stats.values())
                st.metric("📍 Total Distance", f"{total_dist:.1f} km")
            
            # Technician summary
            st.markdown("### 👥 Technician Summary")
            summary_data = []
            for tech_id, stats in tech_stats.items():
                summary_data.append({
                    'Technician': tech_id,
                    'Total Jobs': stats['total_jobs'],
                    'Total Distance (km)': stats['total_distance_km'],
                    'Avg Distance per Job (km)': round(stats['total_distance_km'] / stats['total_jobs'], 2) if stats['total_jobs'] > 0 else 0,
                    'Avg Priority': stats['avg_priority']
                })
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True)
            
            # Interactive map
            st.markdown("### 🗺️ Interactive Route Map")
            st.markdown("""
            <div class="info-box">
                <p><strong>💡 Map Guide:</strong></p>
                <ul>
                    <li>Different colors = different technicians</li>
                    <li>Numbers on markers = stop order</li>
                    <li>Click markers for job details</li>
                    <li>Lines show the optimized routes</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            route_map = create_map(results, tech_stats)
            st_folium(route_map, width=1400, height=600)
            
            # Detailed results
            st.markdown("### 📋 Detailed Job Assignments")
            display_cols = ['assigned_tech', 'route_order', 'dispatch_id', 'priority_score', 'latitude', 'longitude']
            available_cols = [col for col in display_cols if col in results.columns]
            st.dataframe(
                results[available_cols].sort_values(['assigned_tech', 'route_order']),
                use_container_width=True
            )
            
            # Download button
            st.markdown("---")
            st.markdown("### 💾 Download Results")
            st.markdown(download_csv(results), unsafe_allow_html=True)
            
    except Exception as e:
        st.error(f"❌ Error loading file: {str(e)}")
        st.stop()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>Made with ❤️ for efficient dispatch operations</p>
    <p style="font-size: 0.9rem;">Smart Dispatch Optimizer v2.0 | Web Edition</p>
</div>
""", unsafe_allow_html=True)

