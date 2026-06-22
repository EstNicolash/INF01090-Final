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

import pandas as pd
import numpy as np
import plotly.express as px

def plot_operational_mobility_matrix(df: pd.DataFrame) -> px.imshow:
    """
    Computes a cross-tabulation matrix normalized by row profile to output 
    a tactical probability heatmap of vessel mobility vs. the 4 final attack outcomes.
    """
    df_viz = df.copy()
    
    # 1. Filtro estrito de nulos para garantir o alinhamento supervisionado
    df_viz = df_viz.dropna(subset=['vessel_status', 'attack_type'])
    
    # 2. Engenharia de Recursos Inline: Colapso das 4 Classes Efetivas do Target
    target_mapping = {
        'Explosion':   'Fired Upon',
        'Suspicious':  'Attempted',
        'Detained':    'Boarded',
        'Boarding':    'Boarded'
    }
    df_viz['attack_type'] = df_viz['attack_type'].replace(target_mapping)
    
    # Garante o descarte de qualquer string 'Unknown' ou 'Na' residual no target
    allowed_targets = {'Boarded', 'Attempted', 'Hijacked', 'Fired Upon'}
    df_viz = df_viz[df_viz['attack_type'].isin(allowed_targets)]
    
    # 3. Engenharia de Recursos Inline: Redução para as Categorias de Mobilidade
    status_clean = df_viz['vessel_status'].astype(str).str.strip().str.title()
    
    def classify_mobility(s):
        if s in {'Steaming', 'Underway', 'Drifting'}:
            return 'Moving'
        elif s in {'Anchored', 'Berthed', 'Moored', 'Stationary',
                   'Bunkering Operations', 'Grounded', 'Fishing', 'Towed'}:
            return 'Stationary'
        else:
            return 'Unknown'
            
    df_viz['vessel_mobility'] = status_clean.map(classify_mobility)
    
    # Remove a classe 'Unknown' da matriz para focar puramente no contraste tático real
    df_viz = df_viz[df_viz['vessel_mobility'] != 'Unknown']
    
    # 4. Cálculo da Tabulação Cruzada Normalizada por Linha (Gera Probabilidade Condicional)
    contingency_table = pd.crosstab(
        df_viz['vessel_mobility'], 
        df_viz['attack_type'], 
        normalize='index'
    ) * 100
    
    # Força uma ordenação lógica nas colunas do target (da menor severidade para a maior)
    target_order = ['Attempted', 'Fired Upon', 'Boarded', 'Hijacked']
    contingency_table = contingency_table.reindex(columns=target_order, fill_value=0.0)
    
    # 5. Construção da Matriz de Calor (Heatmap)
    fig = px.imshow(
        contingency_table,
        text_auto=".1f", # Insere os valores percentuais dentro de cada célula
        aspect="auto",
        title="Tactical Probability Matrix: Vessel Mobility vs. Model Target Outcomes (%)",
        labels=dict(x="Attack Outcome (Model Target)", y="Vessel Mobility State", color="Probability %"),
        color_continuous_scale="Viridis",
        template="plotly_white"
    )
    
    fig.update_layout(
        coloraxis_showscale=True,
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    return fig


def plot_target_distribution(df: pd.DataFrame) -> px.bar:
    """
    Generates a horizontal bar chart showing the absolute frequency 
    and distribution of the multi-class target variable 'attack_type'.
    """
    # Contamos a frequência de cada classe, incluindo nulos se houver
    target_counts = df['attack_type'].astype(str).value_counts().reset_index()
    target_counts.columns = ['Attack Type', 'Incident Volume']
    
    fig = px.bar(
        target_counts,
        x="Incident Volume",
        y="Attack Type",
        orientation="h",
        title="Global Target Distribution: Total Incidents per Attack Type",
        labels={"Incident Volume": "Number of Recorded Attacks", "Attack Type": "Outcome (Target)"},
        color="Incident Volume",
        color_continuous_scale="Viridis",
        template="plotly_white"
    )
    
    # Mantém o gráfico ordenado do maior volume para o menor
    fig.update_layout(yaxis={'categoryorder':'total ascending'}, coloraxis_showscale=False)
    
    return fig

import pandas as pd
import plotly.express as px

def plot_target_distribution_pie(df: pd.DataFrame) -> px.pie:
    """
    Generates a Pie Chart showing the relative frequency (%) and distribution 
    of the multi-class target 'attack_type' after collapsing redundant classes.
    """
    df_viz = df.copy()
    
    # Dicionário de mapeamento para consolidar e limpar o espaço de estados
    mapping = {
        'Explosion':   'Fired Upon',   # Ambos envolvem ataque armado agressivo
        'Suspicious':  'Attempted',    # Ambos são eventos não consumados
        'Detained':    'Boarded',      # Abordagem/retenção forçada
        'Boarding':    'Boarded'       # Padronização léxica
    }
    
    # Padroniza nulos e aplica o mapeamento de consolidação
    df_viz['attack_type'] = df_viz['attack_type'].astype(str).fillna('Unknown')
    df_viz['attack_type'] = df_viz['attack_type'].replace(mapping)
    
    # Calcula as frequências absolutas para alimentar o gráfico
    target_counts = df_viz['attack_type'].value_counts().reset_index()
    target_counts.columns = ['Attack Type', 'Incident Volume']
    
    # Cria o gráfico de pizza focado em frequência relativa (%)
    fig = px.pie(
        target_counts,
        values='Incident Volume',
        names='Attack Type',
        title='Consolidated Target Variable Distribution (Relative Frequency %)',
        color_discrete_sequence=px.colors.sequential.Viridis,
        template='plotly_white'
    )
    
    # Ajustes estéticos para exibir a porcentagem e o valor bruto ao passar o mouse
    fig.update_traces(
        textposition='inside', 
        textinfo='percent+label',
        hovertemplate="<b>%{label}</b><br>Volume Absoluto: %{value}<br>Percentual: %{percent}"
    )
    
    fig.update_layout(
        margin=dict(l=20, r=20, t=50, b=20),
        legend_title_text='Outcome (Target)'
    )
    
    return fig

import pandas as pd
import plotly.express as px

def plot_vessel_status_by_target(df: pd.DataFrame) -> px.histogram:
    """
    Generates a grouped bar chart (histogram) evaluating the absolute distribution
    of 'vessel_status' categories clustered by the multi-class target 'attack_type'.
    Includes on-the-fly string standardization to merge casing mismatches.
    """
    df_viz = df.copy()
    
    # Drop records where core analytical variables are missing to preserve layout alignment
    df_viz = df_viz.dropna(subset=['vessel_status', 'attack_type'])
    
    # Force strict title-case normalization to seamlessly merge blocks like 'steaming' vs 'Steaming'
    df_viz['vessel_status'] = df_viz['vessel_status'].astype(str).str.strip().str.title()
    
    # Establish a reliable categorical order for the X-axis tracking
    status_order = ['Anchored', 'Steaming', 'Berthed', 'Underway', 'Stationary', 'Unknown']
    
    fig = px.histogram(
        df_viz,
        x='vessel_status',
        color='attack_type',
        barmode='group',
        title='Operational Vulnerability: Incident Volume by Vessel Status & Attack Type',
        labels={'vessel_status': 'Vessel Operational Status', 'count': 'Incident Volume', 'attack_type': 'Outcome'},
        category_orders={'vessel_status': status_order},
        text_auto=True,
        template="plotly_white"
    )
    
    # Rotate ticks smoothly to prevent long text overlapping on smaller frames
    fig.update_layout(
        xaxis_tickangle=45,
        legend_title_text='Attack Outcome',
        margin=dict(l=40, r=40, t=60, b=80)
    )
    
    return fig

import pandas as pd
import numpy as np
import plotly.express as px

def plot_vessel_mobility_by_target(df: pd.DataFrame) -> px.histogram:
    """
    Generates a grouped bar chart evaluating how vessel mobility 
    (Moving vs. Stationary) impacts the tactical attack outcome.
    """
    df_viz = df.copy()
    
    # 1. Drop de nulos apenas no target para preservar a análise
    df_viz = df_viz.dropna(subset=['attack_type'])
    
    # 2. Aplicação da transformação de mobilidade inline para a visualização
    if 'vessel_status' in df_viz.columns:
        status = df_viz['vessel_status'].astype(str).str.strip().str.title()
        
        def classify(s):
            if s in {'Steaming', 'Underway', 'Drifting'}:
                return 'Moving'
            elif s in {'Anchored', 'Berthed', 'Moored', 'Stationary',
                       'Bunkering Operations', 'Grounded', 'Fishing', 'Towed'}:
                return 'Stationary'
            else:
                return 'Unknown'
        
        df_viz['vessel_mobility'] = status.map(classify)
    else:
        df_viz['vessel_mobility'] = 'Unknown'

    # Ordem fixa para as novas categorias no eixo X
    mobility_order = ['Stationary', 'Moving', 'Unknown']
    
    # 3. Construção do Histograma Agrupado
    fig = px.histogram(
        df_viz,
        x='vessel_mobility',
        color='attack_type',
        barmode='group',
        title='Operational Vulnerability: Incident Volume by Vessel Mobility & Attack Type',
        labels={
            'vessel_mobility': 'Vessel Mobility State', 
            'count': 'Incident Volume', 
            'attack_type': 'Outcome'
        },
        category_orders={'vessel_mobility': mobility_order},
        text_auto=True,
        template="plotly_white"
    )
    
    fig.update_layout(
        xaxis_title='Vessel Mobility (Combined)',
        yaxis_title='Number of Recorded Attacks',
        legend_title_text='Attack Outcome',
        margin=dict(l=40, r=40, t=60, b=80)
    )
    
    return fig

def plot_nlp_keyword_impact(df: pd.DataFrame, top_n: int = 15) -> px.bar:
    """
    Computes and visualizes the top tokens extracted from attack descriptions
    to show what terms most frequently drive predictive signaling.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    df_nlp = df.dropna(subset=['attack_description', 'attack_type'])
    
    vectorizer = TfidfVectorizer(stop_words='english', max_features=top_n)
    tfidf_matrix = vectorizer.fit_transform(df_nlp['attack_description'])
    
    # Sum values to find cumulative importance
    importance = np.asarray(tfidf_matrix.sum(axis=0)).flatten()
    vocabulary = vectorizer.get_feature_names_out()
    
    df_words = pd.DataFrame({'Keyword': vocabulary, 'TF-IDF Weight Sum': importance})
    df_words = df_words.sort_values(by='TF-IDF Weight Sum', ascending=True)
    
    fig = px.bar(
        df_words,
        x='TF-IDF Weight Sum',
        y='Keyword',
        orientation='h',
        title=f"NLP Feature Mining: Top {top_n} Informative Tokens in Attack Logs",
        color='TF-IDF Weight Sum',
        color_continuous_scale="Cividis",
        template="plotly_white"
    )
    fig.update_layout(coloraxis_showscale=False)
    return fig

def plot_model_confusion_matrix(y_true, y_pred, class_names: list) -> px.imshow:
    """
    Generates a normalized confusion matrix plot using Plotly to evaluate 
    misclassification patterns across the multi-class target outcomes.
    """
    from sklearn.metrics import confusion_matrix
    
    # Compute and normalize the matrix row-wise (Recall focus)
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    fig = px.imshow(
        cm_normalized,
        x=class_names,
        y=class_names,
        text_auto=".1f",
        title="Model Error Diagnostic: Normalized Confusion Matrix (%)",
        labels=dict(x="Predicted Outcome (Model)", y="True Outcome (Ground Truth)", color="Accuracy %"),
        color_continuous_scale="Blues",
        template="plotly_white"
    )
    return fig

def plot_multiclass_roc_curves(y_test, y_proba, class_names: list) -> px.line:
    """
    Plots individual One-vs-Rest ROC curves for each class in a multi-class setup.
    """
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize
    
    n_classes = len(class_names)
    y_test_bin = label_binarize(y_test, classes=range(n_classes))
    
    roc_df = pd.DataFrame()
    
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        
        df_class = pd.DataFrame({'FPR': fpr, 'TPR': tpr})
        df_class['Class'] = f"{class_names[i]} (AUC = {roc_auc:.2f})"
        roc_df = pd.concat([roc_df, df_class], axis=0)
        
    fig = px.line(
        roc_df, x='FPR', y='TPR', color='Class',
        title='Multi-class One-vs-Rest (OvR) ROC Curves Performance',
        labels={'FPR': 'False Positive Rate (1 - Specificity)', 'TPR': 'True Positive Rate (Sensitivity)'},
        template='plotly_white'
    )
    # Add baseline 50% diagonal line
    fig.add_shape(type='line', line=dict(dash='dash', color='gray'), x0=0, x1=1, y0=0, y1=1)
    return fig