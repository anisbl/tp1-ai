import streamlit as st
import folium
from streamlit_folium import folium_static
import pandas as pd
import os
import math
import time
import networkx as nx
import pickle
import osmnx as ox

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

# Function to create and save a graph if it doesn't exist
def create_jijel_graph():
    try:
        # Get Jijel area graph
        jijel_center = (36.8200, 5.7700)  # Approximate center of Jijel
        G = ox.graph_from_point(jijel_center, dist=5000, network_type='drive')
        
        # Save the graph
        with open('jijel_graph.pkl', 'wb') as f:
            pickle.dump(G, f)
        
        return G, True
    except Exception as e:
        st.error(f"Failed to create graph: {str(e)}")
        return None, False

@st.cache_data
def load_locations():
    if os.path.exists('node_data.csv'):
        return pd.read_csv('node_data.csv')
    else:
        # Create sample data for Jijel if file doesn't exist
        data = {
            'name': ['Jijel University', 'Jijel Hospital', 'Central Market', 'Kotama Beach', 
                     'City Park', 'Bus Station', 'Port', 'City Center'],
            'lat': [36.8082, 36.8182, 36.8150, 36.8250, 36.8190, 36.8120, 36.8220, 36.8160],
            'lon': [5.7693, 5.7782, 5.7740, 5.7820, 5.7760, 5.7730, 5.7790, 5.7750]
        }
        df = pd.DataFrame(data)
        df.to_csv('node_data.csv', index=False)
        return df

@st.cache_resource
def load_graph():
    try:
        if os.path.exists('jijel_graph.pkl'):
            with open('jijel_graph.pkl', 'rb') as f:
                graph = pickle.load(f)
            return graph, True
        else:
            # Create graph if it doesn't exist
            return create_jijel_graph()
    except Exception as e:
        st.warning(f"Error loading graph: {str(e)}")
        return None, False

# Find nearest node to a point
def find_nearest_node(graph, lat, lon):
    try:
        return ox.nearest_nodes(graph, lon, lat)
    except Exception as e:
        st.warning(f"Error finding nearest node: {str(e)}")
        return None

# Load data
try:
    locations_df = load_locations()
    
    # Initialize session state
    if 'path_calculated' not in st.session_state:
        st.session_state.path_calculated = False
        st.session_state.start = locations_df['name'].tolist()[0]
        st.session_state.end = locations_df['name'].tolist()[-1]  # Default to different locations
        st.session_state.algorithm = "Dijkstra"
        st.session_state.map_initialized = False

    # Try to load graph if not already in session state
    if 'graph' not in st.session_state or 'using_graph' not in st.session_state:
        with st.spinner("Loading map data..."):
            st.session_state.graph, st.session_state.using_graph = load_graph()
            
            if st.session_state.using_graph:
                st.success("Map data loaded successfully!")
            else:
                st.info("Using simplified routing (direct paths). Install OSMnx and NetworkX for better routing.")

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
        if st.session_state.using_graph:
            algorithm_options = ["Dijkstra", "A*", "Direct"]
        else:
            algorithm_options = ["Direct"]
            
        algorithm = st.selectbox(
            "Select algorithm",
            algorithm_options,
            index=algorithm_options.index(st.session_state.algorithm) if st.session_state.algorithm in algorithm_options else 0,
            key="algorithm_select"
        )
        st.session_state.algorithm = algorithm

    # Get coordinates
    start_row = locations_df[locations_df['name'] == start].iloc[0]
    end_row = locations_df[locations_df['name'] == end].iloc[0]

    start_lat = start_row['lat']
    start_lon = start_row['lon']
    end_lat = end_row['lat']
    end_lon = end_row['lon']

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
        m = folium.Map(location=[center_lat, center_lon], zoom_start=14, 
                      tiles="OpenStreetMap")

        # Add markers for locations
        for i, row in locations_df.iterrows():
            # Only show markers for start and end points
            if row['name'] == start:
                folium.Marker(
                    location=[row['lat'], row['lon']],
                    popup=row['name'],
                    icon=folium.Icon(color="green", icon="play")
                ).add_to(m)
            elif row['name'] == end:
                folium.Marker(
                    location=[row['lat'], row['lon']],
                    popup=row['name'],
                    icon=folium.Icon(color="red", icon="stop")
                ).add_to(m)
            else:
                # For other locations, use a smaller circle marker without an icon
                folium.CircleMarker(
                    location=[row['lat'], row['lon']],
                    radius=5,  # Small circle
                    popup=row['name'],
                    color="#3186cc",
                    fill=True,
                    fill_color="#3186cc"
                ).add_to(m)

        # Add path to map if calculated
        if st.session_state.path_calculated:
            try:
                path_coords = []
                distance = 0
                
                if st.session_state.using_graph and algorithm != "Direct":
                    # Use graph-based routing
                    graph = st.session_state.graph
                    
                    # Find nearest nodes
                    with st.spinner("Finding nearest network nodes..."):
                        start_node = find_nearest_node(graph, start_lat, start_lon)
                        end_node = find_nearest_node(graph, end_lat, end_lon)
                    
                    if start_node is not None and end_node is not None:
                        with st.spinner(f"Computing shortest path with {algorithm}..."):
                            try:
                                if algorithm == "Dijkstra":
                                    shortest_path_nodes = nx.shortest_path(graph, start_node, end_node, weight='length')
                                elif algorithm == "A*":
                                    shortest_path_nodes = nx.astar_path(graph, start_node, end_node, weight='length', 
                                                                       heuristic=lambda u, v: haversine_distance(
                                                                           graph.nodes[u]['y'], graph.nodes[u]['x'],
                                                                           graph.nodes[v]['y'], graph.nodes[v]['x']
                                                                       ))
                                
                                # Extract coordinates from nodes
                                path_coords = [(graph.nodes[node]['y'], graph.nodes[node]['x']) for node in shortest_path_nodes]
                                
                                # Calculate total distance
                                distance = sum(
                                    graph[shortest_path_nodes[i]][shortest_path_nodes[i+1]][0]['length'] 
                                    for i in range(len(shortest_path_nodes)-1)
                                )
                                
                                # Add intermediate points to the start and end
                                if path_coords:
                                    path_coords = [[start_lat, start_lon]] + path_coords + [[end_lat, end_lon]]
                            except nx.NetworkXNoPath:
                                st.error("No path found between these points using the selected algorithm.")
                                # Fall back to direct path
                                path_coords = [[start_lat, start_lon], [end_lat, end_lon]]
                                distance = haversine_distance(start_lat, start_lon, end_lat, end_lon)
                    else:
                        st.warning("Couldn't find graph nodes near the selected points.")
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

    # Display the map
    folium_static(m, width=800, height=550)

    # Add information
    with st.expander("About this application"):
        st.write("""
        This application helps you find paths between locations in Jijel, Algeria.
        
        It supports various routing methods:
        
        - **Dijkstra**: Finds the shortest path using edge weights
        - **A***: An optimized pathfinding algorithm that uses heuristics for faster performance
        - **Direct**: Shows a straight-line path between points
        
        The map is rendered using OpenStreetMap data. For the best experience, make sure you have NetworkX and OSMnx installed.
        
        To install the required packages:
        ```
        pip install networkx osmnx streamlit foliumstreamlit-folium pandas networkx osmnx numpy pandas tensorflow
        ```
        """)

except Exception as e:
    st.error(f"An error loading the application: {str(e)}")
    st.info("Please try refreshing the page or contact support if the issue persists.")