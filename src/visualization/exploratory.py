import pandas as pd
import numpy as np
import plotly.express as px
import folium
from folium.plugins import MarkerCluster

def plot_geographic_hotspots(df: pd.DataFrame) -> folium.Map:
    """
    Generates a global interactive map using Folium.
    Groups pirate attacks into clusters and maps outcomes via color codes:
    - Red: Hijacked (Severe outcome)
    - Orange: Boarded (Medium severity)
    - Blue: Attempted (Failed piracy event)

    Args:
        df (pd.DataFrame): The processed piracy dataset.

    Returns:
        folium.Map: An interactive geographic map object.
    """
    # Drop rows without geographic coordinates to prevent mapping failures
    df_clean = df.dropna(subset=['latitude', 'longitude'])
    
    # Initialize the global map centered around the equator/oceans
    m = folium.Map(location=[0, 0], zoom_start=2, tiles="Cartodb Positron")
    marker_cluster = MarkerCluster().add_to(m)
    
    # Map severity categories to specific markers
    color_map = {
        'hijacked': 'red',
        'boarded': 'orange',
        'attempted': 'blue'
    }
    
    for _, row in df_clean.iterrows():
        status = str(row.get('attack_type', 'attempted')).lower()
        color = color_map.get(status, 'gray')
        
        # Build interactive HTML popup for notebook exploration
        popup_text = f"""
        <b>Year:</b> {row.get('year')}<br>
        <b>Vessel:</b> {row.get('vessel_type', 'Unknown')}<br>
        <b>Outcome:</b> {status.upper()}<br>
        """
        
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=5,
            popup=folium.Popup(popup_text, max_width=300),
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7
        ).add_to(marker_cluster)
        
    return m

def plot_null_pattern_by_year(df: pd.DataFrame) -> px.bar:
    """
    Creates a stacked bar chart illustrating missing vs. available entries
    for the 'vessel_type' column across the temporal axis.
    Validates the hypothesis regarding a protocol shift around 2015.

    Args:
        df (pd.DataFrame): The raw or raw-loaded dataset.

    Returns:
        px.bar: A Plotly Express stacked bar figure.
    """
    df_analysis = df.copy()
    
    # Discretize completeness into a binary tracking state
    df_analysis['vessel_type_status'] = np.where(
        df_analysis['vessel_type'].isna(), 'Missing (Data Quality Gap)', 'Available Entry'
    )
    
    # Group data to compute frequencies per state per year
    df_grouped = (
        df_analysis.groupby(['year', 'vessel_type_status'])
        .size()
        .reset_index(name='count')
    )
    
    fig = px.bar(
        df_grouped, 
        x='year', 
        y='count', 
        color='vessel_type_status',
        title='Historical Data Completeness Evolution for Vessel Type',
        labels={'year': 'Year', 'count': 'Total Attacks', 'vessel_type_status': 'Data Quality State'},
        color_discrete_map={'Missing (Data Quality Gap)': '#EF553B', 'Available Entry': '#636EFA'}
    )
    fig.update_layout(template="plotly_white", barmode='stack')
    return fig

def plot_target_distribution_by_vessel(df: pd.DataFrame, top_n: int = 10) -> px.bar:
    """
    Analyzes the correlation between Vessel Categories and Attack Outcomes (Multi-class Target).
    Filters out rare vessel structures to prevent visualization noise.

    Args:
        df (pd.DataFrame): The processed dataset.
        top_n (int): Number of top vessel types to visualize. Default is 10.

    Returns:
        px.bar: A grouped histogram figure.
    """
    df_viz = df.copy()
    
    # Cast categories to string temporarily to apply placeholder for missing attributes
    df_viz['vessel_type'] = df_viz['vessel_type'].astype(str).fillna('Not Recorded')
    
    # Isolate top-N categories with higher baseline frequencies
    top_vessels = df_viz['vessel_type'].value_counts().nlargest(top_n).index
    df_filtered = df_viz[df_viz['vessel_type'].isin(top_vessels)]
    
    fig = px.histogram(
        df_filtered,
        x='vessel_type',
        color='attack_type', 
        barmode='group',
        title=f'Vessel Susceptibility vs. Attack Outcomes (Top {top_n} Targets)',
        labels={'vessel_type': 'Vessel Type', 'count': 'Incident Count', 'attack_type': 'Outcome'},
    )
    fig.update_layout(template="plotly_white", xaxis={'categoryorder':'total descending'})
    return fig


import pandas as pd
import numpy as np
import plotly.express as px
import folium
from folium.plugins import HeatMap

def plot_geospatial_heatmap(df: pd.DataFrame) -> folium.Map:
    """
    Generates a pure geographic HeatMap of pirate attacks.
    Blurs coordinates into a continuous density spectrum (Red = High Risk).
    """
    # Filter valid coordinates
    df_clean = df.dropna(subset=['latitude', 'longitude'])
    
    # Initialize global map layout
    m = folium.Map(location=[0, 0], zoom_start=2, tiles="Cartodb Positron")
    
    # Extract coordinate pairs as a weight list for the HeatMap layer
    heat_data = df_clean[['latitude', 'longitude']].values.tolist()
    
    # Add the continuous density layer
    HeatMap(heat_data, radius=15, blur=10, min_opacity=0.4).add_to(m)
    
    return m

def plot_shore_distance_distribution(df: pd.DataFrame) -> px.histogram:
    """
    Creates a marginal distribution density plot combining shore distance 
    and multi-class outcomes to spot tactical piracy operational ranges.
    """
    df_viz = df.copy()
    df_viz = df_viz.dropna(subset=['shore_distance', 'attack_type'])
    
    # Clip extreme outliers (e.g., deeper than 600 nautical miles) to preserve scale
    q_high = df_viz['shore_distance'].quantile(0.95)
    df_filtered = df_viz[df_viz['shore_distance'] <= q_high]
    
    fig = px.histogram(
        df_filtered,
        x="shore_distance",
        color="attack_type",
        marginal="box", # Adds a marginal box-plot on top for structural density
        nbins=100,
        title="Piracy Operational Range: Attack Density vs. Distance to Shore",
        labels={"shore_distance": "Distance to Shore (Nautical Miles / Units)", "count": "Incident Volume"},
        barmode="stack",
        template="plotly_white"
    )
    
    return fig


import pandas as pd
import numpy as np
import plotly.express as px

def plot_multidimensional_piracy_profile(df: pd.DataFrame) -> px.parallel_categories:
    """
    Generates a Parallel Categories Plot to visualize multi-dimensional flows
    between vessel profiles, tactical status, ranges, and final attack outcomes.
    """
    df_viz = df.copy()
    df_viz = df_viz.dropna(subset=['vessel_status', 'vessel_type', 'attack_type'])
    
    # Create a quick categorical distance bin for cleaner visual flow
    df_viz['distance_range'] = np.where(df_viz['shore_distance'] <= 12.0, 'Territorial Seas', 'High Seas')
    
    # Select structural dimensions to map
    dimensions = ['distance_range', 'vessel_status', 'vessel_type', 'attack_type']
    
    fig = px.parallel_categories(
        df_viz,
        dimensions=dimensions,
        title="End-to-End Piracy Incident Structural Profiles",
        labels={
            "distance_range": "Sea Zone",
            "vessel_status": "Vessel State",
            "vessel_type": "Ship Class",
            "attack_type": "Incident Outcome"
        },
        template="plotly_white"
    )
    return fig

def plot_operational_status_matrix(df: pd.DataFrame) -> px.imshow:
    """
    Computes a cross-tabulation matrix normalized by row profile 
    to output a tactical probability heatmap of status vs outcomes.
    """
    df_viz = df.copy()
    df_viz = df_viz.dropna(subset=['vessel_status', 'attack_type'])
    
    # Compute normalized cross-tabulation (percentages per vessel status row)
    contingency_table = pd.crosstab(
        df_viz['vessel_status'], 
        df_viz['attack_type'], 
        normalize='index'
    ) * 100
    
    fig = px.imshow(
        contingency_table,
        text_auto=".1f", # Appends percentage text inside the matrix cells
        aspect="auto",
        title="Tactical Probability Matrix: Vessel Status vs. Attack Outcome (%)",
        labels=dict(x="Attack Outcome (Target)", y="Vessel Operational Status", color="Probability %"),
        color_continuous_scale="Viridis",
        template="plotly_white"
    )
    return fig