import streamlit as st
import folium
from streamlit_folium import folium_static
import pandas as pd
import os
import math
import time

# Try importing networkx, but handle failure gracefully
try:
    import networkx as nx
    import pickle
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

st.set_page_config(layout="wide", page_title="Path Finder - Jijel, Algeria")

# Add custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .subheader {
        font-size: 1.5rem;
        color: #0D47A1;
        margin-top: 1.5rem;
    }
    .stButton>button {
        background-color: #1E88E5;
        color: white;
        font-weight: bold;
        padding: 0.5rem 1rem;
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>Jijel Path Finder</h1>", unsafe_allow_html=True)

# Haversine distance calculation
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000  # Earth radius in meters
    
    # Convert to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    # Differences
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    # Haversine formula
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance = R * c
    
    return distance

# Compatibility for different Streamlit versions
try:
    # For newer Streamlit versions
    @st.cache_data
    def load_locations():
        if os.path.exists('node_data.csv'):
            return pd.read_csv('node_data.csv', index_col=0)
        else:
            # Create sample data if file doesn't exist
            data = {
                'name': ['University', 'Hospital', 'Market', 'Beach', 'Park', 'Train Station'],
                'lat': [36.8000, 36.8100, 36.8050, 36.8150, 36.8200, 36.8250],
                'lon': [5.7600, 5.7700, 5.7650, 5.7750, 5.7800, 5.7850]
            }
            df = pd.DataFrame(data)
            return df
except AttributeError:
    # For older Streamlit versions
    @st.cache
    def load_locations():
        if os.path.exists('node_data.csv'):
            return pd.read_csv('node_data.csv', index_col=0)
        else:
            # Create sample data if file doesn't exist
            data = {
                'name': ['University', 'Hospital', 'Market', 'Beach', 'Park', 'Train Station'],
                'lat': [36.8000, 36.8100, 36.8050, 36.8150, 36.8200, 36.8250],
                'lon': [5.7600, 5.7700, 5.7650, 5.7750, 5.7800, 5.7850]
            }
            df = pd.DataFrame(data)
            return df

# Try to load graph with version compatibility
try:
    # For newer Streamlit versions
    @st.cache_resource
    def load_graph():
        try:
            if os.path.exists('jijel_graph.pkl') and HAS_NETWORKX:
                try:
                    with open('jijel_graph.pkl', 'rb') as f:
                        graph = pickle.load(f)
                    return graph, True
                except Exception:
                    return None, False
            else:
                return None, False
        except Exception:
            return None, False
except AttributeError:
    # For older Streamlit versions
    @st.cache(allow_output_mutation=True)
    def load_graph():
        try:
            if os.path.exists('jijel_graph.pkl') and HAS_NETWORKX:
                try:
                    with open('jijel_graph.pkl', 'rb') as f:
                        graph = pickle.load(f)
                    return graph, True
                except Exception:
                    return None, False
            else:
                return None, False
        except Exception:
            return None, False

# Function to find nearest node (works with or without OSMnx)
def find_nearest_node(graph, lat, lon, using_graph=True):
    if using_graph and HAS_NETWORKX:
        try:
            import osmnx as ox
            return ox.distance.nearest_nodes(graph, lon, lat)
        except Exception:
            return None
    return None

# Load data
try:
    locations_df = load_locations()
    
    # Initialize session state
    if 'path_calculated' not in st.session_state:
        st.session_state.path_calculated = False
        st.session_state.start = locations_df['name'].tolist()[0]
        st.session_state.end = locations_df['name'].tolist()[0]
        st.session_state.algorithm = "Direct"
        st.session_state.map_initialized = False

    # Try to load graph if not already in session state
    if 'graph' not in st.session_state or 'using_graph' not in st.session_state:
        with st.spinner("Loading map data..."):
            st.session_state.graph, st.session_state.using_graph = load_graph()
            
            if st.session_state.using_graph:
                st.success("Map data loaded successfully!")
            else:
                st.info("Using simplified routing (direct paths).")

    # Create layout with columns
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        start = st.selectbox(
            'Select starting point',
            locations_df['name'].tolist(),
            index=locations_df['name'].tolist().index(st.session_state.start),
            key="start_select"
        )
        st.session_state.start = start

    with col2:
        end = st.selectbox(
            'Select destination',
            locations_df['name'].tolist(),
            index=locations_df['name'].tolist().index(st.session_state.end),
            key="end_select"
        )
        st.session_state.end = end

    with col3:
        if st.session_state.using_graph and HAS_NETWORKX:
            algorithm_options = ["Dijkstra", "A*", "Direct"]
        else:
            algorithm_options = ["Direct"]
            
        algorithm = st.selectbox(
            "Select algorithm",
            algorithm_options,
            index=0,
            key="algorithm_select"
        )
        st.session_state.algorithm = algorithm

    # Get coordinates
    start_idx = locations_df[locations_df['name'] == start].index[0]
    end_idx = locations_df[locations_df['name'] == end].index[0]

    start_lat = locations_df.loc[start_idx, 'lat']
    start_lon = locations_df.loc[start_idx, 'lon']
    end_lat = locations_df.loc[end_idx, 'lat']
    end_lon = locations_df.loc[end_idx, 'lon']

    # Calculate button
    find_path = st.button("Find Path", key="find_path")

    if find_path:
        st.session_state.path_calculated = True
        st.session_state.map_initialized = False  # Force map redraw

    # Display the map section
    st.markdown("<h2 class='subheader'>Map View</h2>", unsafe_allow_html=True)

    # Create map only if necessary
    if not st.session_state.map_initialized:
        # Create map centered between start and end points
        center_lat = (start_lat + end_lat) / 2
        center_lon = (start_lon + end_lon) / 2
        m = folium.Map(location=[center_lat, center_lon], zoom_start=12, 
                      tiles="OpenStreetMap")

        # Add markers for start and end points
        folium.Marker(
            location=[start_lat, start_lon],
            popup=start,
            icon=folium.Icon(color="green", icon="play")
        ).add_to(m)

        folium.Marker(
            location=[end_lat, end_lon],
            popup=end,
            icon=folium.Icon(color="red", icon="stop")
        ).add_to(m)

        # Add path to map if calculated
        if st.session_state.path_calculated:
            try:
                path_coords = []
                distance = 0
                
                if st.session_state.using_graph and algorithm != "Direct" and HAS_NETWORKX:
                    # Use graph-based routing
                    graph = st.session_state.graph
                    start_node = find_nearest_node(graph, start_lat, start_lon, st.session_state.using_graph)
                    end_node = find_nearest_node(graph, end_lat, end_lon, st.session_state.using_graph)
                    
                    if start_node is not None and end_node is not None:
                        with st.spinner(f"Computing shortest path with {algorithm}..."):
                            if algorithm == "Dijkstra":
                                shortest_path_nodes = nx.shortest_path(graph, start_node, end_node, weight='length')
                            elif algorithm == "A*":
                                shortest_path_nodes = nx.astar_path(graph, start_node, end_node, weight='length')
                            
                            # Extract coordinates
                            path_coords = [(graph.nodes[node]['y'], graph.nodes[node]['x']) for node in shortest_path_nodes]
                            
                            # Calculate distance
                            distance = sum(
                                graph[shortest_path_nodes[i]][shortest_path_nodes[i+1]][0]['length'] 
                                for i in range(len(shortest_path_nodes)-1)
                            )
                    else:
                        # Fall back to direct path
                        path_coords = [[start_lat, start_lon], [end_lat, end_lon]]
                        distance = haversine_distance(start_lat, start_lon, end_lat, end_lon)
                else:
                    # Use direct path
                    path_coords = [[start_lat, start_lon], [end_lat, end_lon]]
                    distance = haversine_distance(start_lat, start_lon, end_lat, end_lon)
                
                # Draw the path
                folium.PolyLine(
                    locations=path_coords,
                    color='blue',
                    weight=5,
                    opacity=0.7
                ).add_to(m)
                
                st.success(f"Path found! Distance: {distance:.2f} meters ({distance/1000:.2f} km)")
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                # Fall back to direct path
                path_coords = [[start_lat, start_lon], [end_lat, end_lon]]
                distance = haversine_distance(start_lat, start_lon, end_lat, end_lon)
                
                folium.PolyLine(
                    locations=path_coords,
                    color='red',
                    weight=4,
                    opacity=0.7,
                    dash_array='10'
                ).add_to(m)
                
                st.warning(f"Showing direct path instead. Distance: {distance:.2f} meters ({distance/1000:.2f} km)")
        
        st.session_state.map = m
        st.session_state.map_initialized = True
    else:
        m = st.session_state.map

    # Display the map with version compatibility
    try:
        # For newer versions of streamlit_folium
        folium_static(m, width=800, height=550)
    except TypeError:
        # For older versions that don't support width/height params
        folium_static(m)

    # Add information
    with st.expander("About this application"):
        st.write("""
        This application helps you find paths between locations in Jijel, Algeria.
        
        It supports various routing methods:
        
        - **Dijkstra**: Finds the shortest path using edge weights (if graph is available)
        - **A***: An optimized pathfinding algorithm that uses heuristics (if graph is available)
        - **Direct**: Shows a straight-line path between points
        
        The map is rendered using OpenStreetMap data.
        """)

except Exception as e:
    st.error(f"An error loading the application: {str(e)}")
    st.info("Please try refreshing the page or contact support if the issue persists.")