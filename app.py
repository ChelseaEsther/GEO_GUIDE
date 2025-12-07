import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from model import GoldAnomalyDetector
import tempfile
import os
from pathlib import Path
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="GEO GUIDE - AI powered Gold Anomaly Detection System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background-color: #fcfff5 !important;
    }
    [data-testid="stSidebar"] {
        background-color: #103713;
    }
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    h1, h2, h3 {
        color: #103713;
        font-weight: 700;
    }
    .stButton>button {
        background-color: #628b35;
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 8px;
        font-weight: 600;
    }
    .stButton>button:hover {
        background-color: #103713;
    }
    .info-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 4px solid #628b35;
    }
    [data-testid="stMetricValue"] {
        color: #103713;
        font-size: 2rem;
        font-weight: bold;
    }
    [data-testid="stFileUploader"] {
        background-color: #e2dbd0;
        padding: 1rem;
        border-radius: 8px;
    }
    img {
        max-width: 100%;
        height: auto;
        border-radius: 8px;
    }
    @media (max-width: 768px) {
        .stButton>button {
            width: 100%;
            margin: 0.5rem 0;
        }
        h1 {
            font-size: 1.5rem;
        }
        .info-card {
            padding: 1rem;
        }
        .stApp {
            background-color: #fcfff5 !important;
        }
    }
    hr {
        margin: 2rem 0;
        border: none;
        height: 2px;
        background: linear-gradient(to right, #628b35, #103713);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'prediction_results' not in st.session_state:
    st.session_state.prediction_results = None

# Sidebar navigation
with st.sidebar:
    st.title("🌍 GEO GUIDE UGANDA")
    st.markdown("---")
    
    page = st.radio(
        "Navigation",
        ["HOME", "ABOUT GEO GUIDE", "CASE STUDY", "RESULTS & ANALYSIS", "RUN ANALYSIS"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("""
    <div style='color: white; font-size: 0.8rem; text-align: center;'>
    <b>GEO GUIDE</b><br>
    AI-Powered Mineral Prediction System<br>
    Kampala, Uganda<br><br>
    © 2025 Luwedde Esther Chelsea
    </div>
    """, unsafe_allow_html=True)

# HOME PAGE
if page == "HOME":
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("<h1 style='font-size: 3rem;'>GEO GUIDE</h1>", unsafe_allow_html=True)
        st.markdown("<h3 style='color: #628b35;'>AI-Powered Mineral Prediction System</h3>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class='info-card'>
        <p style='font-size: 1.1rem; line-height: 1.8;'>
        Welcome to <b>GEO GUIDE UGANDA</b>, an advanced AI & Machine Learning system designed to identify 
        potential gold mineralization zones in Uganda using remote sensing data and machine learning algorithms.
        </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.image("https://upload.wikimedia.org/wikipedia/commons/4/4e/Flag_of_Uganda.svg", width=250)
    
    st.markdown("---")
    st.markdown("<h2>Key Features</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='info-card'>
        <h3 style='color: #628b35;'>Remote Sensing</h3>
        <p><b>GEO GUIDE </b>analyzes Sentinel-2 multispectral imagery and SRTM elevation data (DEM)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='info-card'>
        <h3 style='color: #628b35;'>Machine Learning</h3>
        <p>Implements Unsupervised K-Means clustering with PCA dimensionality reduction for anomaly detection in early exploration stages</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='info-card'>
        <h3 style='color: #628b35;'>High Accuracy Record</h3>
        <p>Achieved 100% detection rate on Buhweju case study</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("<h2>📍 Study Area: Buhweju-Mashonga, Uganda</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        <div class='info-card'>
        <h3>Geographical Context</h3>
        <ul style='line-height: 1.8;'>
            <li><b>Location:</b> Southwestern Uganda, </li>
            <li><b>Mineral:</b> Gold</li>
            <li><b>Terrain:</b> Moderate terrain with rolling hills</li>
            <li><b>Known Deposits:</b> 25 documented gold occurrences</li>
            <li><b>Exploration Potential:</b> High prospectivity</li>
            <li><b>Geological Unit:</b> Paleoproterozoic</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.info("""
        **Why Buhweju-Mashonga?**
        
        - Rich geological diversity
        - Known gold mineralization
        - Ideal testing ground for ML models
        - Potential for new discoveries of Gold and other Minerals like Lead
        """)

# ABOUT PAGE
elif page == "ABOUT GEO GUIDE":
    st.markdown("<h1>About GEO GUIDE</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-card'>
    <h2 style='color: #628b35;'>What is GEO GUIDE?</h2>
    <p style='font-size: 1.1rem; line-height: 1.8;'>
    GEO GUIDE is an innovative <b>AI-powered mineral prediction system</b> that leverages machine learning 
    and remote sensing data to identify potential gold mineralization zones.
    </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='info-card'>
        <h3 style='color: #103713;'>Project Objectives</h3>
        <ul style='line-height: 1.8;'>
            <li>To prove the need for AI powered tools in the Uganda mining Sector for faster discovery of mineralized zones.</li>
            <li>To identify high potential gold exploration zones in Uganda.</li>
            <li>To reduce exploration costs and time taken to discover Mineralized zones in early Exploration stages.</li>
            <li>To minimize environmental impact made by mineral exploration.</li>
            <li>To provide data-driven insights for geologists.</li>
            <li>To support sustainable mining practices.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='info-card'>
        <h3 style='color: #103713;'>🔬 Methodology</h3>
        <ul style='line-height: 1.8;'>
            <li><b>Data Collection:</b> Sentinel-2 imagery, SRTM DEM (Uganda) and Lithology</li>
            <li><b>Model Selection:</b> Unsupervised Learning(K-Means Clustering) mainly used for early exploration stages</li>
            <li><b>Feature Engineering:</b> 38 spectral and terrain features</li>
            <li><b>Preprocessing:</b> StandardScaler normalization</li>
            <li><b>Dimensionality Reduction:</b> PCA (5 components)</li>
            <li><b>Clustering:</b> K-Means (5 clusters)</li>
            <li><b>Validation:</b> Known gold occurrence locations</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

# CASE STUDY PAGE
elif page == "CASE STUDY":
    st.markdown("<h1>Case Study: Buhweju-Mashonga, Uganda</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-card'>
    <h2 style='color: #628b35;'>Study Overview</h2>
    <p style='font-size: 1.1rem; line-height: 1.8;'>
    This case study demonstrates the application of GEO GUIDE to Buhweju-Mashonga in southwestern Uganda in order to detect gold anomalies in the area that highlight potiential Gold mineral Deposits.
    </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("<h2>Data Acquisition</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='info-card'>
        <h3 style='color: #103713;'>Satellite Imagery</h3>
        <p><b>Source:</b> Sentinel-2 (ESA Copernicus Program)</p>
        <ul style='line-height: 1.8;'>
            <li><b>Platform:</b> Sentinel-2A satellites</li>
            <li><b>Resolution:</b> 10m-20m multispectral bands</li>
            <li><b>Bands Used:</b> B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12</li>
            <li><b>Coverage:</b> Buhweju-Mashonga</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='info-card'>
        <h3 style='color: #103713;'>Elevation Data</h3>
        <p><b>Source:</b> SRTM Digital Elevation Model of Uganda from RCMRD GMES and Africa GeoPortal</p>
        <ul style='line-height: 1.8;'>
            <li><b>Resolution:</b> 30m</li>
            <li><b>Products:</b> DEM, Slope, Aspect, TPI, Hillshade</li>
            <li><b>Processing:</b> Derived 9 terrain attributes </li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class='info-card'>
        <h3 style='color: #103713;'>Lithology Data</h3>
        <p><b>Source:</b> Africa Surface Lithology Map from RCMRD Open Data Site</p>
        <ul style='line-height: 1.8;'>
            <li><b>Resolution:</b> 20m</li>
            <li><b>Products:</b> Rock types in Buhweju-Mashonga (lithology layers)</li>
            <li><b>Processing:</b> Classified into a 4 one hot encode </li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("<h2>Methodology Workflow</h2>", unsafe_allow_html=True)
    
    steps = [
        ("1", "Data Preprocessing", "Downloaded and preprocessed Sentinel-2, SRTM DEM and Lithology data"),
        ("2", "Feature Engineering", "Calculated 23 features which were spectral indices and terrain attributes"),
        ("3", "Feature Stacking", "Combined all the 23 features together with the 10 sentinel bands and the 4 one hot encode lithology bands (Total: 38 features) and stacked them into a <b>single multi-band raster</b>"),
        ("4", "Ground Truth Data", "Collected 25 known gold occurrence locations in the Area Of Interest from <b>Uganda Mining Cadastre Map</b>"),
        ("5", "Feature Selection", "Selected the 8 most important features: hillshade, Clay_Index, NDNS, Hydroxyl_Index, Alteration, aspect, TPI, Ferric_Iron"),
        ("6", "Data Normalization", "Applied StandardScaler to normalize features"),
        ("7", "Dimensionality Reduction", "Applied PCA to reduce to 5 principal components"),
        ("8", "Clustering", "Performed K-Means clustering (k=5)"),
        ("9", "Anomaly Detection", "Identified clusters containing the known gold occurrences"),
        ("10", "Validation", "Achieved 100% detection rate of the gold occurrences on test data")
    ]
    
    for num, title, desc in steps:
        st.markdown(f"""
        <div class='info-card'>
        <h3 style='color: #628b35;'>Step {num}: {title}</h3>
        <p style='line-height: 1.8;'>{desc}</p>
        </div>
        """, unsafe_allow_html=True)

# RESULTS & ANALYSIS PAGE
elif page == "RESULTS & ANALYSIS":
    st.markdown("<h1>Results & Analysis Dashboard</h1>", unsafe_allow_html=True)
    
    st.info("""
    This dashboard presents comprehensive results of the gold anomaly detection analysis 
    for Buhweju-Mashonga, Uganda. All visualizations are from the trained model case study.
    """)
    
    viz_path = Path("assets/visualizations")
    
    if not viz_path.exists():
        st.warning("⚠️ Visualization folder not found. Please ensure assets/visualizations/ contains your PNG files.")
    else:
        st.markdown("---")
        
        # Main Results
        st.markdown("<h2>Main Detection Results</h2>", unsafe_allow_html=True)
        st.markdown("""
        <div class='info-card'>
        <p><b>What this shows:</b> K-Means clustering results with known gold occurrences (red stars) 
        and binary anomaly map highlighting high-priority exploration zones.</p>
        <p><b>Interpretation:</b> The left cluster map displays 5 distinct clusters identified by the model. 
        Red stars mark the 25 known gold occurrences. The right map displays the final anomaly map where 
        green indicates anomalous (high-priority) zones, while yellow and red indicate moderate anomaly and normal areas respectively . All gold 
        points fall within the anomalous zones, validating the model's 100% detection rate.</p>
        </div>
        """, unsafe_allow_html=True)
        
        img1 = viz_path / "anomaly_detection_results.png"
        if img1.exists():
            col = st.columns([1])[0]
            col.image(str(img1))
            #st.image(str(img1), use_container_width=True)
        else:
            st.error("❌ anomaly_detection_results.png not found")
        
        st.markdown("---")
        
        # Cluster Statistics
        st.markdown("<h2>Cluster Anomaly Statistics</h2>", unsafe_allow_html=True)
        st.markdown("""
        <div class='info-card'>
        <p><b>What this shows:</b> Detailed breakdown of each cluster including:</p>
        <ul>
            <li>Number of gold points detected in each cluster (top-left)</li>
            <li>Cluster size as percentage of total area (top-right)</li>
            <li>Priority ranking for exploration (bottom-left)</li>
            <li>Anomalous vs normal area coverage (bottom-right pie chart)</li>
        </ul>
        <p><b>Interpretation:</b> Clusters 0, 1, and 2 are flagged as HIGH PRIORITY because they contain 
        gold occurrences (12, 7, and 6 points respectively). These three clusters together represent 80.3% 
        of the study area, indicating extensive exploration potential.</p>
        </div>
        """, unsafe_allow_html=True)
        
        img2 = viz_path / "cluster_anomaly_statistics.png"
        if img2.exists():
            col = st.columns([1])[0]
            col.image(str(img2))
            #st.image(str(img2), use_container_width=True)
        else:
            st.error("❌ cluster_anomaly_statistics.png not found")
        
        st.markdown("---")
        
        # PCA Analysis
        st.markdown("<h2>PCA Feature Contributions</h2>", unsafe_allow_html=True)
        st.markdown("""
        <div class='info-card'>
        <p><b>What this shows:</b> Principal Component Analysis reveals which original features 
        contribute most to each principal component.</p>
        <p><b>Key Insights:</b> Features with high loadings (blueish color) in PC1 and PC2 are the most responsibl 
        for distinguishing mineralized from non-mineralized zones. The cumulative variance plot (bottom-right) 
        shows that 5 components capture 91.8% of total variance.</p>
        </div>
        """, unsafe_allow_html=True)
        
        img3 = viz_path / "pca_feature_contributions.png"
        if img3.exists():
            col = st.columns([1])[0]
            col.image(str(img3))
            #st.image(str(img3), use_container_width=True)
        else:
            st.error("❌ pca_feature_contributions.png not found")
        
        st.markdown("---")
        
        # Terrain Analysis
        st.markdown("<h2>Terrain Analysis</h2>", unsafe_allow_html=True)
        st.markdown("""
        <div class='info-card'>
        <p><b>What this shows:</b> Terrain characteristics of Buhweju-Mashonga:</p>
        <ul>
            <li><b>Left Side - Slope Distribution:</b> Histogram showing frequency of different slope angles. 
            Mean (13.09°) and median (11.80°) are marked.</li>
            <li><b>Right Side - Slope Map:</b> Spatial distribution of terrain steepness.</li>
        </ul>
        <p><b>Terrain Classification:</b> MODERATE TERRAIN - Rolling hills with mean slope of 13.09°. That shows that Buhweju-Mashonga is a hilly area</p>
        <p><b>Advantage:</b> Moderate terrain is accessible for field work while providing 
        sufficient relief for geological mapping.</p>
        </div>
        """, unsafe_allow_html=True)
        
        img4 = viz_path / "terrain_analysis.png"
        if img4.exists():
            col = st.columns([1])[0]
            col.image(str(img4))
            #st.image(str(img4), use_container_width=True)
        else:
            st.error("❌ terrain_analysis.png not found")
        
        st.markdown("---")
        
        # Comprehensive Summary
        st.markdown("<h2>Comprehensive Summary Dashboard</h2>", unsafe_allow_html=True)
        st.markdown("""
        <div class='info-card'>
        <p><b>What this shows:</b> An integrated executive summary combining all key metrics:</p>
        <ul>
            <li><b>Detection Performance:</b> 100% detection rate (25/25 gold points)</li>
            <li><b>Cluster Distribution:</b> Relative sizes and gold content</li>
            <li><b>Gold Concentration:</b> Which clusters contain the most occurrences</li>
            <li><b>Exploration Priority Areas:</b> Anomalous vs normal zones</li>
            <li><b>PCA Variance:</b> Information retention analysis</li>
            <li><b>Key Findings & Recommendations:</b> Actionable insights</li>
        </ul>
        <p><b>Executive Summary:</b> The 100% detection rate validates model effectiveness, while 
        80.3% anomalous coverage indicates significant exploration potential. Cluster 0 is the highest 
        priority target with 12 gold points.</p>
        </div>
        """, unsafe_allow_html=True)
        
        img5 = viz_path / "comprehensive_summary_dashboard.png"
        if img5.exists():
            col = st.columns([1])[0]
            col.image(str(img5))
            #st.image(str(img5), use_container_width=True)
        else:
            st.error("❌ comprehensive_summary_dashboard.png not found")
        
        st.markdown("---")
        
        # Key Findings Summary
        st.markdown("<h2>Key Findings Summary</h2>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class='info-card'>
            <h3 style='color: #628b35;'>Performance</h3>
            <ul>
                <li>100% detection rate of the known gold occurrences</li>
                <li>25/25 gold points detected</li>
                <li>3 high priority clusters</li>
                <li>Excellent model validation</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class='info-card'>
            <h3 style='color: #628b35;'>Coverage</h3>
            <ul>
                <li>80.3% anomalous area</li>
                <li>601,662 priority pixels</li>
                <li>5 distinct clusters</li>
                <li>Moderate terrain (13.09°)</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class='info-card'>
            <h3 style='color: #628b35;'>Recommendations</h3>
            <ul>
                <li>Most focus should be on Cluster 0 during exploration</li>
                <li>Field validation by geologists is recommended</li>
                <li>Geochemical sampling should be carried out to confirm the model results</li>
                <li>Geophysical surveys recommended too</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        st.success("""
        ### Priority Recommendations:
        
        1. **Immediate Action:** Focus on Cluster 0 (Has 12 gold points, 31% of area)
        2. **Secondary Targets:** Also explore Clusters 1 and 2
        3. **Field Validation:** Ground truth the 80.3% anomalous zones
        4. **Sampling Protocol:** Collect geochemical samples in high priority clusters
        5. **Geophysical Survey:** Confirm structural controls with magnetics or Airborne surveys
        """)

# RUN ANALYSIS PAGE
elif page == "RUN ANALYSIS":
    st.markdown("<h1>Run Analysis on Your Data</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-card'>
    <h3>Upload Your Data for Gold Anomaly Detection</h3>
    <p style='line-height: 1.8;'>
    Upload your stacked features raster (TIF) and gold occurrences (GPKG). The model will train fresh 
    on your data and generate prospectivity maps, analysis and visualizations for you..
    </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.info("""
    **Required Files:**
    - **Stacked Features TIF:** Multi-band raster with stacked features of your Area of Interest.
    - **Gold Occurrences GPKG:** Vector file with known gold locations.
    - **Important:** Ensure that both the files have the same CRS (Coordinate System), otherwise the results will be gabbage.
    
    **Note:** Model trains fresh on uploaded data (takes upto 1 minutes)
    """)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Upload Stacked Features (TIF)")
        uploaded_tif = st.file_uploader(
            "Choose stacked features tif file",
            type=['tif', 'tiff'],
            help="Multi-band GeoTIFF"
        )
        
        if uploaded_tif:
            st.success(f"Uploaded: {uploaded_tif.name} ({uploaded_tif.size / (1024*1024):.2f} MB)")
    
    with col2:
        st.markdown("### Upload Gold Occurrences (GPKG)")
        uploaded_gpkg = st.file_uploader(
            "Choose gold occurences gpkg file",
            type=['gpkg'],
            help="GeoPackage with gold points"
        )
        
        if uploaded_gpkg:
            st.success(f"Uploaded: {uploaded_gpkg.name} ({uploaded_gpkg.size / 1024:.2f} KB)")
    
    st.markdown("---")
    
    if uploaded_tif and uploaded_gpkg:
        st.success("Both files uploaded successfully!")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("Run Analysis", use_container_width=True):
                
                with st.spinner("Training model and generating analysis..."):
                    
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp_tif:
                            tmp_tif.write(uploaded_tif.read())
                            tif_path = tmp_tif.name
                        
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.gpkg') as tmp_gpkg:
                            tmp_gpkg.write(uploaded_gpkg.read())
                            gpkg_path = tmp_gpkg.name
                        
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        status_text.text("Initializing model...")
                        progress_bar.progress(10)
                        detector = GoldAnomalyDetector()
                        
                        status_text.text("Training and analyzing...")
                        progress_bar.progress(30)
                        results = detector.train_and_predict(tif_path, gpkg_path)
                        
                        progress_bar.progress(100)
                        status_text.text("Analysis complete!")
                        
                        st.session_state.prediction_results = results
                        
                        os.unlink(tif_path)
                        os.unlink(gpkg_path)
                        
                        st.success("Analysis Complete!")
                        st.balloons()
                        
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        st.exception(e)
    
    else:
        st.warning("Please upload both files to run analysis")
    
    if st.session_state.prediction_results is not None:
        results = st.session_state.prediction_results
        
        st.markdown("---")
        st.markdown("<h2>Analysis Results</h2>", unsafe_allow_html=True)
        
        st.markdown("### Detection Performance")
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric("Detection Rate", f"{results['detection_rate']:.1f}%")
        col2.metric("Gold Points", f"{results['gold_points_detected']}/{results['gold_points_total']}")
        col3.metric("Anomalous Area", f"{results['anomalous_percentage']:.1f}%")
        col4.metric("PCA Variance", f"{results['pca_variance_explained']:.1f}%")
        
        st.markdown("---")
        
        st.markdown("### Cluster Statistics")
        cluster_df = pd.DataFrame(results['cluster_statistics'])
        st.dataframe(cluster_df, use_container_width=True)
        
        st.markdown("---")
        
        # VISUALIZATIONS
        st.markdown("### Main Detection Results")
        st.image(results['visualizations']['main_detection'])
        
        st.markdown("---")
        st.markdown("### Cluster Anomaly Statistics")
        st.image(results['visualizations']['cluster_stats'])
        
        st.markdown("---")
        st.markdown("### PCA Feature Contributions")
        st.markdown("""
        <div class='info-card'>
        <p><b>What this shows:</b> Which original features contribute most to each principal component. 
        Features with high loadings are most responsible for gold detection.</p>
        </div>
        """, unsafe_allow_html=True)
        st.image(results['visualizations']['pca_contributions'])
        
        st.markdown("---")
        st.markdown("### Terrain Analysis")
        if results['visualizations'].get('terrain_analysis'):
            st.markdown("""
            <div class='info-card'>
            <p><b>What this shows:</b> Terrain characteristics of the study area.</p>
            <p><b>Terrain Classification:</b> {}</p>
            </div>
            """.format(results.get('terrain_classification', 'Unknown')), unsafe_allow_html=True)
            st.image(results['visualizations']['terrain_analysis'])
            
            if 'slope_stats' in results:
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Mean Slope", f"{results['slope_stats']['Mean']:.1f}°")
                col2.metric("Median Slope", f"{results['slope_stats']['Median']:.1f}°")
                col3.metric("Min Slope", f"{results['slope_stats']['Min']:.1f}°")
                col4.metric("Max Slope", f"{results['slope_stats']['Max']:.1f}°")
        else:
            st.info("Terrain analysis not available (slope data not found)")
        
        st.markdown("---")
        st.markdown("### Comprehensive Summary Dashboard")
        st.markdown("""
        <div class='info-card'>
        <p><b>What this shows:</b> An integrated executive summary combining all key metrics - 
        detection performance, cluster distribution, gold concentration, exploration priorities, 
        PCA variance, and actionable recommendations.</p>
        </div>
        """, unsafe_allow_html=True)
        if results['visualizations'].get('comprehensive_summary'):
            st.image(results['visualizations']['comprehensive_summary'])
        
        st.markdown("---")
        st.markdown("### Recommendations")
        
        if results['detection_rate'] >= 70:
            st.success(f"""
            **EXCELLENT RESULTS!**
            
            Detection rate: {results['detection_rate']:.1f}% ({results['gold_points_detected']}/{results['gold_points_total']} points)
            
            **Next Steps:**
            - Focus on {results['anomalous_percentage']:.1f}% anomalous area
            - Prioritize {len([c for c in results['cluster_statistics'] if c['Status'] == 'HIGH PRIORITY'])} high-priority clusters
            - Conduct field validation and geochemical sampling
            """)
        else:
            st.warning(f"""
            **GOOD RESULTS ✓**
            
            Detection rate: {results['detection_rate']:.1f}%
            
            Consider field validation and feature refinement.
            """)
        
        st.markdown("---")
        st.markdown("### Download Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            csv = cluster_df.to_csv(index=False)
            st.download_button(
                label="Download Cluster Statistics (CSV)",
                data=csv,
                file_name="cluster_statistics.csv",
                mime="text/csv"
            )
        
        with col2:
            import json
            results_json = {
                'detection_rate': float(results['detection_rate']),
                'gold_points_detected': int(results['gold_points_detected']),
                'gold_points_total': int(results['gold_points_total']),
                'anomalous_percentage': float(results['anomalous_percentage']),
                'cluster_statistics': results['cluster_statistics']
            }
            json_str = json.dumps(results_json, indent=2)
            st.download_button(
                label="Download Full Report (JSON)",
                data=json_str,
                file_name="analysis_report.json",
                mime="application/json"
            )

# FOOTER
st.markdown("---")
st.markdown("""
<div style='background-color: #103713; padding: 2rem; border-radius: 10px; margin-top: 3rem;'>
    <div style='text-align: center; color: white;'>
        <h3 style='color: #fcfff5;'>🌍 GEO GUIDE UGANDA</h3>
        <p style='color: #fcfff5; font-size: 1.1rem;'>
            AI-Powered Mineral Prediction System<br>
            Kampala District, Uganda
        </p>
        <hr style='border: 1px solid #628b35; margin: 1.5rem 0;'>
        <p style='color: #e2dbd0; font-size: 0.9rem;'>
            <strong>Disclaimer:</strong> This is a screening tool. Validate results through 
            field work, geochemical sampling, and professional geological consultation.
        </p>
        <p style='color: #fcfff5; margin-top: 1rem;'>
            © 2025 Esther Chelsea | GEO GUIDE AI PROJECT<br>
            Built for Sustainable Mineral Exploration
        </p>
    </div>
</div>
""", unsafe_allow_html=True)
