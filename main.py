import osmnx as ox
import networkx as nx
import streamlit as st
import folium
from streamlit_folium import folium_static
import pandas as pd
import pickle
import os

# Load the pre-downloaded graph
@st.cache_resource
def load_graph():
    if os.path.exists('jijel_graph.pkl'):
        with open('jijel_graph.pkl', 'rb') as f:
            return pickle.load(f)
    else:
        st.error("Graph file not found! Please make sure 'jijel_graph.pkl' is in the same directory as this script.")
        st.stop()

def nearest_node(graph, lat, lon):
    return ox.distance.nearest_nodes(graph, lon, lat)

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

# Load data
locations_df = pd.read_csv('node_data.csv', index_col=0)

# Load graph with a progress bar
if 'graph' not in st.session_state:
    with st.spinner("Loading map data..."):
        st.session_state.graph = load_graph()
        st.success("Map data loaded successfully!")

# Initialize session state variables
if 'map_initialized' not in st.session_state:
    st.session_state.map_initialized = False
    st.session_state.path_calculated = False
    st.session_state.start = locations_df['name'].tolist()[0]
    st.session_state.end = locations_df['name'].tolist()[0]
    st.session_state.algorithm = "Dijkstra"
    st.session_state.last_map = None

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
    algorithm = st.selectbox(
        "Select algorithm",
        ["Dijkstra", "A*", "Bellman-Ford"],
        index=["Dijkstra", "A*", "Bellman-Ford"].index(st.session_state.algorithm),
        key="algorithm_select"
    )
    st.session_state.algorithm = algorithm

# Get the latitude and longitude of the start and end locations
start_idx = locations_df[locations_df['name'] == start].index[0]
end_idx = locations_df[locations_df['name'] == end].index[0]

start_lat = locations_df.loc[start_idx, 'lat']
start_lon = locations_df.loc[start_idx, 'lon']
end_lat = locations_df.loc[end_idx, 'lat']
end_lon = locations_df.loc[end_idx, 'lon']

# Calculate button - only this will trigger map redrawing with path
find_path = st.button("Find Shortest Path", key="find_path")

if find_path:
    st.session_state.path_calculated = True
    st.session_state.map_initialized = False  # Force map redraw

# Display the map
st.markdown("<h2 class='subheader'>Map View</h2>", unsafe_allow_html=True)

# Only create a new map if necessary
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
            graph = st.session_state.graph
            start_node = nearest_node(graph, start_lat, start_lon)
            end_node = nearest_node(graph, end_lat, end_lon)
            
            with st.spinner(f"Computing shortest path with {algorithm}..."):
                if algorithm == "Dijkstra":
                    shortest_path_nodes = nx.shortest_path(graph, start_node, end_node, weight='length')
                elif algorithm == "A*":
                    shortest_path_nodes = nx.astar_path(graph, start_node, end_node, weight='length')  
                elif algorithm == "Bellman-Ford":
                    shortest_path_nodes = nx.bellman_ford_path(graph, start_node, end_node, weight='length')
                
                shortest_path_coords = [(graph.nodes[node]['y'], graph.nodes[node]['x']) for node in shortest_path_nodes]
            
            # Draw shortest path
            folium.PolyLine(
                locations=shortest_path_coords, 
                color='blue', 
                weight=5,
                opacity=0.7
            ).add_to(m)
            
            # Calculate distance
            distance = sum(
                graph[shortest_path_nodes[i]][shortest_path_nodes[i+1]][0]['length'] 
                for i in range(len(shortest_path_nodes)-1)
            )
            
            st.success(f"Path found! Total distance: {distance:.2f} meters ({distance/1000:.2f} km)")
            
        except nx.NetworkXNoPath:
            st.error("No path found between the selected locations.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    
    st.session_state.last_map = m
    st.session_state.map_initialized = True
else:
    m = st.session_state.last_map

# Display the map
map_container = st.container()
with map_container:
    folium_static(m, width=800, height=550)

# Add information section
with st.expander("About this application"):
    st.write("""
    This application helps you find the shortest path between two locations in Jijel, Algeria.
    It uses different routing algorithms to calculate the optimal path:
    
    - **Dijkstra**: Finds the shortest path using edge weights
    - **A***: An optimized pathfinding algorithm that uses heuristics
    - **Bellman-Ford**: Can handle negative edge weights (if present)
    
    The map and route are rendered using OpenStreetMap data.
    """)