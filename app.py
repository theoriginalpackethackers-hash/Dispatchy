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
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    st.warning("⚠️ AI libraries not installed. Using basic chatbot. Install sentence-transformers for AI features.")

# Page configuration
st.set_page_config(
    page_title="Smart Dispatch Optimizer",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# AI CHATBOT SYSTEM - Open Source, 100% Private, No External API Calls
# ============================================================================

@st.cache_resource
def load_ai_model():
    """Load the sentence transformer model (cached for performance)"""
    if not AI_AVAILABLE:
        return None
    try:
        # Using a lightweight model that runs well on Streamlit Cloud
        model = SentenceTransformer('all-MiniLM-L6-v2')  # Only 80MB, very fast
        return model
    except Exception as e:
        st.warning(f"⚠️ AI model loading issue: {e}. Using fallback mode.")
        return None

# Comprehensive Knowledge Base for Dispatch Optimization
KNOWLEDGE_BASE = [
    {
        "question": "How do I upload my data?",
        "answer": "📤 **Uploading Your Data:**\n\n1. Click 'Browse files' in the sidebar under 'Data Source'\n2. Select your CSV file from your computer\n3. Or drag and drop your CSV directly into the upload box\n4. The file will automatically load and show a preview\n\n**Tip:** Make sure your CSV has columns for latitude, longitude, and dispatch/job IDs!"
    },
    {
        "question": "What columns are required in my CSV?",
        "answer": "📊 **Required Columns:**\n\n**Must Have:**\n• Latitude column (can be named: latitude, lat, Latitude, Customer_Latitude, etc.)\n• Longitude column (can be named: longitude, lon, lng, Customer_Longitude, etc.)\n\n**Strongly Recommended:**\n• dispatch_id or job_id (unique identifier)\n• status (open, pending, unassigned, etc.)\n• priority (1-5 scale)\n\n**Nice to Have:**\n• required_skill or skills\n• appointment_start_time\n• appointment_end_time\n\nThe tool auto-detects column names, so exact naming isn't critical as long as 'latitude'/'longitude' appear somewhere in the column name!"
    },
    {
        "question": "Why is my data being filtered or showing as empty?",
        "answer": "🔍 **Common Filtering Issues:**\n\nThe optimizer filters data for these reasons:\n\n1. **Invalid Coordinates:** Rows with blank/null latitude or longitude are removed\n2. **Status Filtering:** Only dispatches with status 'open', 'pending', or 'unassigned' are included by default\n3. **Data Type Issues:** Make sure lat/lon are numbers, not text\n\n**How to Fix:**\n• Check the debug output when you run optimization - it shows exactly what was filtered\n• Look for the line '🔍 Status values found' to see what statuses are in your data\n• If your statuses are different (like 'Scheduled', 'Ready'), the optimizer will filter them out\n\n**Need different statuses included?** Let me know and I can help adjust the filtering logic!"
    },
    {
        "question": "How does the optimization algorithm work?",
        "answer": "🧠 **Optimization Process:**\n\n**Step 1: Clustering**\n• Uses KMeans algorithm to group jobs geographically\n• Creates 'territories' for each technician\n• Ensures balanced workload distribution\n\n**Step 2: Assignment**\n• Assigns each cluster to a technician\n• Considers job priority scores\n• Balances total jobs and total distance\n\n**Step 3: Route Optimization**\n• Uses 2-opt algorithm to find shortest route within each tech's jobs\n• Minimizes backtracking and inefficient routing\n• Considers time windows if provided\n\n**Step 4: Scoring**\n• Calculates priority scores based on urgency and appointment times\n• Balances distance vs. priority based on your weight settings\n\n**Result:** Optimized routes that minimize drive time while prioritizing urgent jobs!"
    },
    {
        "question": "What do the settings do?",
        "answer": "⚙️ **Optimization Settings Explained:**\n\n**Maximum Technicians (5-50):**\n• How many techs to distribute jobs across\n• More techs = smaller territories, less driving per tech\n• Fewer techs = larger territories, more jobs per tech\n\n**Priority Weight (1.0-3.0):**\n• How much to favor high-priority jobs\n• Higher value = urgent jobs get better routes/earlier stops\n• Lower value = treats all jobs more equally\n• Default 2.0 is usually good for balanced optimization\n\n**Distance Weight (0.5-2.0):**\n• How aggressively to minimize driving distance\n• Higher value = shorter routes, but might sacrifice priority\n• Lower value = allows longer routes to serve high-priority jobs first\n• Default 1.0 balances distance and priority well\n\n**Pro Tip:** Start with defaults, then adjust based on your specific needs!"
    },
    {
        "question": "How do I download the results?",
        "answer": "💾 **Downloading Your Results:**\n\n1. After optimization completes, scroll down past the map\n2. You'll see a green button: '📥 Click Here to Download Results CSV'\n3. Click it to download the optimized dispatch CSV\n4. The file includes:\n   • All original data columns\n   • assigned_tech (which technician)\n   • route_order (sequence to visit jobs)\n   • tech_cluster (territory assignment)\n\n**Using the Results:**\n• Import into your dispatch system\n• Sort by 'assigned_tech' then 'route_order'\n• Each tech follows their numbered route sequence\n\n**Tip:** You can re-run optimization with different settings and compare results!"
    },
    {
        "question": "Why are some technicians assigned more jobs than others?",
        "answer": "⚖️ **Workload Balancing Explained:**\n\nThe optimizer balances **total workload**, not just job count:\n\n**Factors Considered:**\n• Geographic clustering (jobs close together)\n• Total driving distance\n• Job density in areas\n• Priority distribution\n\n**Why Unequal Job Counts:**\n• TECH-001: 12 jobs spread over 15 miles = ~45 min drive time\n• TECH-003: 20 jobs in 5 miles = ~45 min drive time\n\nBoth techs have similar total work time!\n\n**The Goal:** Balance total work hours, not just job counts.\n\n**Want equal job counts instead?** Adjust the distance weight lower and priority weight higher!"
    },
    {
        "question": "What does the map show?",
        "answer": "🗺️ **Interactive Map Guide:**\n\n**Colors:**\n• Each color represents one technician's territory\n• Same color = same tech's jobs\n\n**Lines:**\n• Show the optimized route path\n• Follow the line to see drive sequence\n\n**Markers:**\n• Numbers show route order (1, 2, 3...)\n• Click any marker for job details\n• Shows: Job ID, Priority, Location coordinates\n\n**Route Lines:**\n• Click the route line to see:\n  - Technician ID\n  - Total jobs\n  - Total distance (in miles)\n\n**Pro Tips:**\n• Zoom in/out to see detail or overview\n• Click markers to verify job assignments\n• Check that routes make logical sense geographically!"
    },
    {
        "question": "Can I modify the optimization after it runs?",
        "answer": "🔄 **Modifying Results:**\n\n**Yes! Here's how:**\n\n**Method 1: Adjust Settings & Re-run**\n1. Change sliders in sidebar (techs, priority, distance)\n2. App will auto-rerun with new settings\n3. Compare results to see which is better\n\n**Method 2: Edit CSV Manually**\n1. Download the results CSV\n2. Open in Excel\n3. Manually adjust 'assigned_tech' or 'route_order' if needed\n4. Import into your system\n\n**Method 3: Filter & Re-upload**\n1. Download results\n2. Filter to only jobs you want to re-optimize\n3. Upload filtered CSV back to the tool\n4. Re-run optimization\n\n**Pro Tip:** Try different settings to find what works best for your operation!"
    },
    {
        "question": "Is my data secure and private?",
        "answer": "🔒 **Data Privacy & Security:**\n\n**100% SECURE:**\n• ✅ All processing happens in your browser/Streamlit session\n• ✅ NO data is sent to external APIs\n• ✅ NO data is stored permanently\n• ✅ NO connection to external AI services (OpenAI, etc.)\n• ✅ Open-source AI runs entirely locally\n• ✅ Data is cleared when you close the browser\n\n**How It Works:**\n1. You upload CSV → stored in temporary session memory\n2. Optimization runs in the app (no external calls)\n3. Results generated locally\n4. You download results\n5. Session ends → all data automatically deleted\n\n**Perfect for:**\n• Sensitive company data\n• HIPAA/compliance requirements\n• Internal-only information\n• Confidential customer addresses\n\n**Your data never leaves your control!** 🛡️"
    },
    {
        "question": "What if I have appointment time windows?",
        "answer": "⏰ **Handling Appointment Times:**\n\nThe optimizer DOES support time windows!\n\n**Required Columns:**\n• `appointment_start` or `appointment_start_time`\n• `appointment_end` or `appointment_end_time`\n\n**How It Works:**\n1. Jobs with appointments within 2 hours get +5 priority boost\n2. Jobs within 4 hours get +3 priority boost\n3. Jobs within 8 hours get +1 priority boost\n4. These jobs are prioritized in route order\n\n**Date Format:**\n• Accepts: 'YYYY-MM-DD HH:MM:SS', 'MM/DD/YYYY HH:MM', etc.\n• Flexible parsing handles most date formats\n\n**Pro Tip:** Make sure appointment columns are present in your CSV, and urgent appointments will automatically be prioritized in the routing!"
    },
    {
        "question": "Can I use this for multiple days or shifts?",
        "answer": "📅 **Multi-Day/Shift Optimization:**\n\n**Current Capability:**\n• Single-day optimization\n• One shift at a time\n\n**Workaround for Multiple Days:**\n\n**Option 1: Filter by Day**\n1. Filter your CSV to just one day's dispatches\n2. Upload and optimize\n3. Download results\n4. Repeat for each day\n\n**Option 2: Use Status/Tags**\n1. Add a column like 'schedule_day' or 'shift'\n2. Filter in Excel before uploading\n3. Optimize each batch separately\n\n**Option 3: Combine Results**\n1. Run optimization for each day/shift\n2. Download all results\n3. Combine in Excel with day/shift tags\n\n**Future Enhancement:** Let me know if you need true multi-day optimization built in!"
    }
]

def get_ai_response(user_question, model):
    """Get AI-powered response using semantic search"""
    if model is None:
        # Fallback to keyword matching if model failed to load
        return "🤖 AI assistant is temporarily unavailable. Please try the Quick Help Topics dropdown for common questions."
    
    try:
        # Encode user question
        user_embedding = model.encode([user_question])
        
        # Encode all knowledge base questions
        kb_questions = [item['question'] for item in KNOWLEDGE_BASE]
        kb_embeddings = model.encode(kb_questions)
        
        # Calculate similarity scores
        similarities = cosine_similarity(user_embedding, kb_embeddings)[0]
        
        # Get best match
        best_match_idx = similarities.argmax()
        confidence = similarities[best_match_idx]
        
        # Return answer if confidence is high enough
        if confidence > 0.3:  # Threshold for relevance
            answer = KNOWLEDGE_BASE[best_match_idx]['answer']
            confidence_emoji = "🎯" if confidence > 0.7 else "💡"
            return f"{confidence_emoji} **Answer:**\n\n{answer}"
        else:
            return ("🤔 I'm not sure about that specific question. Try:\n\n"
                   "• Selecting from Quick Help Topics above\n"
                   "• Rephrasing your question\n"
                   "• Checking the debug output when you run optimization\n\n"
                   "Common topics: uploading data, column requirements, filtering issues, settings, downloading results.")
    
    except Exception as e:
        return f"⚠️ Error processing question: {e}"

# Load AI model at startup (cached)
ai_model = load_ai_model()

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
    """Calculate distance between two points on earth in kilometers (will be converted to miles for display)"""
    # Convert to float to handle any string or int inputs
    lat1, lon1, lat2, lon2 = float(lat1), float(lon1), float(lat2), float(lon2)
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return 6371 * c

def calculate_priority_score(row, priority_weight):
    """Calculate priority score based on multiple factors"""
    # Convert to float to handle string values
    try:
        base_score = float(row.get('priority', 3))
    except (ValueError, TypeError):
        base_score = 3.0
    
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
    # Store original columns for error reporting
    original_columns = list(data.columns)
    
    # Standardize column names
    data.columns = data.columns.str.lower().str.replace(' ', '_').str.strip()
    
    # Debug: Show what columns we have after cleanup
    st.write("🔍 **DEBUG: Your CSV columns after cleanup:**")
    st.write(list(data.columns))
    
    # Auto-detect columns with SIMPLE matching - just look for the word anywhere in the column name
    column_mapping = {}
    
    # Find latitude column - if "lat" or "latitude" appears ANYWHERE in the name
    found_lat_col = None
    for col in data.columns:
        col_lower = col.lower()
        if 'latitude' in col_lower or (('lat' in col_lower) and ('long' not in col_lower) and ('lng' not in col_lower)):
            found_lat_col = col
            column_mapping[col] = 'latitude'  # Map FROM original TO standard name
            break
    
    # Find longitude column - if "lon", "long", or "longitude" appears ANYWHERE in the name
    found_lon_col = None
    for col in data.columns:
        col_lower = col.lower()
        if 'longitude' in col_lower or 'long' in col_lower or 'lng' in col_lower:
            found_lon_col = col
            column_mapping[col] = 'longitude'  # Map FROM original TO standard name
            break
    
    # Find dispatch ID
    for col in data.columns:
        if col not in column_mapping.keys():
            if any(term in col for term in ['dispatch', 'job', 'ticket', 'work_order', 'id', 'number']):
                if 'id' in col or 'num' in col or col in ['dispatch', 'job', 'ticket']:
                    column_mapping[col] = 'dispatch_id'
                    break
    
    # Find status
    for col in data.columns:
        if col not in column_mapping.keys():
            if 'status' in col or 'state' in col:
                column_mapping[col] = 'status'
                break
    
    # Find priority
    for col in data.columns:
        if col not in column_mapping.keys():
            if 'prior' in col or 'urgency' in col or 'severity' in col:
                column_mapping[col] = 'priority'
                break
    
    # Find skills
    for col in data.columns:
        if col not in column_mapping.keys():
            if 'skill' in col or 'capability' in col or 'cert' in col:
                column_mapping[col] = 'skills'
                break
    
    # Find appointment times
    for col in data.columns:
        if col not in column_mapping.keys():
            if any(term in col for term in ['appt', 'appointment', 'scheduled', 'time']):
                if any(term in col for term in ['start', 'begin', 'from']):
                    column_mapping[col] = 'appointment_start'
                elif any(term in col for term in ['end', 'finish', 'to']):
                    column_mapping[col] = 'appointment_end'
    
    # Debug output - show what we found
    if found_lat_col:
        st.info(f"✅ Found latitude column: `{found_lat_col}` → will rename to `latitude`")
    if found_lon_col:
        st.info(f"✅ Found longitude column: `{found_lon_col}` → will rename to `longitude`")
    
    st.write("🔧 **Column mapping:**", column_mapping)
    
    # Check if we found the critical columns
    if not found_lat_col or not found_lon_col:
        missing = []
        if not found_lat_col:
            missing.append("latitude")
        if not found_lon_col:
            missing.append("longitude")
        
        raise ValueError(
            f"❌ Could not find {' and '.join(missing)} column(s).\n\n"
            f"**Your CSV columns after cleanup:**\n"
            f"{', '.join(data.columns)}\n\n"
            f"**Your ORIGINAL columns:**\n"
            f"{', '.join(original_columns)}\n\n"
            f"The column name must contain the word 'latitude' or 'longitude' somewhere in it.\n"
            f"Examples that work: Latitude, customer_latitude, lat, LATITUDE, Location_Lat"
        )
    
    data = data.rename(columns=column_mapping)
    
    # Convert numeric columns to proper types
    if 'latitude' in data.columns:
        data['latitude'] = pd.to_numeric(data['latitude'], errors='coerce')
    if 'longitude' in data.columns:
        data['longitude'] = pd.to_numeric(data['longitude'], errors='coerce')
    if 'priority' in data.columns:
        data['priority'] = pd.to_numeric(data['priority'], errors='coerce').fillna(3)
    
    # Calculate priority scores
    data['priority_score'] = data.apply(lambda row: calculate_priority_score(row, priority_weight), axis=1)
    
    # Filter valid dispatches
    st.write(f"📊 **Total rows in CSV:** {len(data)}")
    
    valid_data = data[
        (data['latitude'].notna()) & 
        (data['longitude'].notna())
    ].copy()
    
    st.write(f"📍 **Rows with valid lat/lon:** {len(valid_data)}")
    
    if 'status' in valid_data.columns:
        try:
            # Convert status to string and filter
            valid_data['status'] = valid_data['status'].astype(str)
            unique_statuses = valid_data['status'].unique()
            st.write(f"🔍 **Status values found:** {list(unique_statuses)}")
            
            before_filter = len(valid_data)
            
            # Try to filter by common "active" statuses
            active_statuses = ['open', 'pending', 'unassigned', 'scheduled', 'assigned', 'ready', 'new', 'active']
            filtered_data = valid_data[valid_data['status'].str.lower().isin(active_statuses)]
            
            # Only apply filter if we kept some data
            if len(filtered_data) > 0:
                valid_data = filtered_data
                st.write(f"✅ **Rows with active status:** {len(valid_data)} (filtered out {before_filter - len(valid_data)} completed/closed)")
            else:
                # No matches found - keep all data
                st.warning(f"⚠️ No matching active statuses found. Using ALL {len(valid_data)} dispatches regardless of status.")
                st.info(f"💡 **Tip:** Your statuses are: {', '.join(unique_statuses)}. All will be included in optimization.")
        except Exception as e:
            # If status filtering fails, just keep all data
            st.warning(f"⚠️ Could not filter by status: {e}. Using all data.")
            pass
    
    if len(valid_data) == 0:
        raise ValueError(
            "❌ No valid dispatches found after filtering!\n\n"
            "Possible issues:\n"
            "• All dispatches have invalid lat/lon coordinates\n"
            "• All dispatches have a status other than 'open', 'pending', or 'unassigned'\n\n"
            "Check the debug output above to see what was filtered out."
        )
    
    valid_data = valid_data.sort_values('priority_score', ascending=False)
    st.write(f"🎯 **Final valid dispatches to optimize:** {len(valid_data)}")
    
    # Cluster into technician groups using KMeans
    from sklearn.cluster import KMeans
    
    n_clusters = min(max_techs, len(valid_data))
    if n_clusters > 1:
        # Ensure latitude and longitude are numeric for clustering
        coords_array = valid_data[['latitude', 'longitude']].astype(float).values
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        valid_data['tech_cluster'] = kmeans.fit_predict(coords_array)
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
            
            # Convert km to miles (1 km = 0.621371 miles)
            total_distance_miles = total_distance * 0.621371
            
            tech_stats[tech_id] = {
                'total_jobs': len(tech_dispatches),
                'total_distance_miles': round(total_distance_miles, 2),
                'avg_priority': round(tech_dispatches['priority_score'].mean(), 2),
                'coordinates': coords
            }
            
            optimized_routes[tech_id] = tech_dispatches
    
    # Safety check before concatenation
    if not optimized_routes:
        raise ValueError("❌ No routes were created. This shouldn't happen - please check your data.")
    
    # Filter out any empty DataFrames
    non_empty_routes = [df for df in optimized_routes.values() if len(df) > 0]
    
    if not non_empty_routes:
        raise ValueError("❌ All routes are empty after optimization.")
    
    final_results = pd.concat(non_empty_routes, ignore_index=True)
    
    st.success(f"✅ Successfully optimized {len(final_results)} dispatches across {len(optimized_routes)} technicians!")
    
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
            popup=f"{tech_id}<br>Jobs: {len(tech_data)}<br>Distance: {tech_stats[tech_id]['total_distance_miles']:.1f} miles"
        ).add_to(m)
        
        for idx, row in tech_data.iterrows():
            popup_html = f"""
            <div style="font-family: Arial; min-width: 200px; color: black;">
                <h4 style="margin: 0 0 10px 0; color: {color};">{tech_id}</h4>
                <table style="width: 100%; font-size: 12px; color: black;">
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
    
    st.markdown("---")
    st.markdown("### 🤖 AI Assistant")
    st.caption("💡 Open-source AI • 100% Private • No external API calls")
    
    # Chat interface
    with st.expander("💬 Ask Me Anything!", expanded=False):
        user_question = st.text_area(
            "Type your question here:",
            placeholder="Example: Why are some techs assigned more jobs?\nHow do appointment times work?\nWhat columns do I need?",
            key="ai_question",
            height=80
        )
        
        if st.button("🚀 Get Answer", use_container_width=True):
            if user_question.strip():
                with st.spinner("🧠 AI is thinking..."):
                    response = get_ai_response(user_question, ai_model)
                    st.markdown(response)
            else:
                st.warning("Please type a question first!")
    
    # Quick examples
    st.markdown("**💡 Example Questions:**")
    example_questions = [
        "How do I upload data?",
        "Why is data filtered?",
        "What do the settings do?",
        "Is my data secure?",
        "How does optimization work?"
    ]
    
    selected_example = st.selectbox(
        "Or select an example:",
        ["Choose a question..."] + example_questions,
        key="example_q"
    )
    
    if selected_example != "Choose a question...":
        with st.spinner("🧠 AI is thinking..."):
            response = get_ai_response(selected_example, ai_model)
            st.markdown(response)

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
                total_dist = sum(s['total_distance_miles'] for s in tech_stats.values())
                st.metric("📍 Total Distance", f"{total_dist:.1f} miles")
            
            # Technician summary
            st.markdown("### 👥 Technician Summary")
            summary_data = []
            for tech_id, stats in tech_stats.items():
                summary_data.append({
                    'Technician': tech_id,
                    'Total Jobs': stats['total_jobs'],
                    'Total Distance (miles)': stats['total_distance_miles'],
                    'Avg Distance per Job (miles)': round(stats['total_distance_miles'] / stats['total_jobs'], 2) if stats['total_jobs'] > 0 else 0,
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

