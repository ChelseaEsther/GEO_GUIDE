import numpy as np
import rasterio
import geopandas as gpd
from rasterio.transform import rowcol
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
import io

class GoldAnomalyDetector:
    """Fresh Training Model - Trains on uploaded data"""
    
    def __init__(self):
        """Initialize with the 8 required features and band mapping"""
        # HARDCODED BAND ORDER (based on your stacking order)
        self.band_order = [
            'B2_float32',                    # 1
            'B3_float32',                    # 2
            'B4_float32',                    # 3
            'B5_align2',                     # 4
            'B6_align2',                     # 5
            'B7_align2',                     # 6
            'B8_float32',                    # 7
            'B8A_align2',                    # 8
            'B11_align2',                    # 9
            'B12_align',                     # 10
            'DEM_float32',                   # 11
            'slope',                         # 12
            'aspect',                        # 13
            'curvature',                     # 14
            'profile_curvature',             # 15
            'planform_curvature',            # 16
            'hillshade',                     # 17  ← IMPORTANT!
            'tri',                           # 18
            'tpi',                           # 19  ← IMPORTANT!
            'roughness',                     # 20
            'litho_Colluvium',               # 21
            'litho_Metaigneous',             # 22
            'litho_Metasedimentary',         # 23
            'litho_Volcanic-Ash,Tuff,Mudflow', # 24
            'NDVI',                          # 25
            'NDWI',                          # 26
            'IOR',                           # 27
            'Ferrous_Index',                 # 28
            'Clay_Index',                    # 29  ← IMPORTANT!
            'Hydroxyl_Index',                # 30  ← IMPORTANT!
            'Ferric_Iron',                   # 31  ← IMPORTANT!
            'Ferric_Oxides',                 # 32
            'Alteration',                    # 33  ← IMPORTANT!
            'Carbonate',                     # 34
            'Gossans',                       # 35
            'Ratio_6_5',                     # 36
            'Ratio_6_7',                     # 37
            'NDNS'                           # 38  ← IMPORTANT!
        ]
        
        # The 8 important features we need
        self.important_bands = [
            'hillshade',      # Band 17 (index 16)
            'Clay_Index',     # Band 29 (index 28)
            'NDNS',           # Band 38 (index 37)
            'Hydroxyl_Index', # Band 30 (index 29)
            'Alteration',     # Band 33 (index 32)
            'aspect',         # Band 13 (index 12)
            'tpi',            # Band 19 (index 18)
            'Ferric_Iron'     # Band 31 (index 30)
        ]
        
        # Precompute indices (0-based)
        self.feature_indices = [16, 28, 37, 29, 32, 12, 18, 30]
        
        print("✅ Model initialized - Ready to train on uploaded data")
        print(f"   Using 8 important features: {self.important_bands}")
        print(f"   Band indices: {self.feature_indices}")
    
    def extract_band_names(self, raster_path):
        """Extract band names - use hardcoded order if descriptions missing"""
        with rasterio.open(raster_path) as src:
            band_names = list(src.descriptions) or []
            
            # If no descriptions, use our hardcoded order
            if not any(band_names) or len(band_names) != len(self.band_order):
                print(f"⚠️ No band descriptions found. Using hardcoded band order.")
                band_names = self.band_order[:src.count]
            
        return band_names
    
    def find_feature_indices(self, band_names):
        """Find indices of the 8 important features"""
        selected_indices = []
        missing = []
        
        print(f"\n🔍 Searching for required 8 features...")
        
        for feature in self.important_bands:
            found = False
            
            # Try exact match
            if feature in band_names:
                idx = band_names.index(feature)
                selected_indices.append(idx)
                print(f"   ✅ Found '{feature}' at index {idx}")
                found = True
            else:
                # Try case-insensitive match
                lowers = [b.lower() for b in band_names]
                if feature.lower() in lowers:
                    idx = lowers.index(feature.lower())
                    selected_indices.append(idx)
                    print(f"   ✅ Found '{feature}' (case-insensitive) at index {idx}")
                    found = True
            
            if not found:
                missing.append(feature)
                print(f"   ❌ Missing: '{feature}'")
        
        # If we found all features, return them
        if not missing:
            return selected_indices
        
        # Otherwise, use hardcoded indices (fallback)
        print(f"\n⚠️ Using hardcoded feature indices as fallback")
        print(f"   Indices: {self.feature_indices}")
        return self.feature_indices
    
    def train_and_predict(self, stacked_tif_path, gold_gpkg_path):
        """
        Train model on uploaded data and generate predictions
        
        Returns:
            dict: Complete analysis results with visualizations
        """
        results = {}
        
        print("\n" + "="*70)
        print("🚀 STARTING FRESH MODEL TRAINING & ANALYSIS")
        print("="*70)
        
        # 1. LOAD RASTER DATA
        print("\n[1/10] 🔄 Loading stacked raster...")
        with rasterio.open(stacked_tif_path) as src:
            stack = src.read()
            profile = src.profile
            transform = src.transform
        
        n_bands, rows, cols = stack.shape
        print(f"✅ Loaded: {n_bands} bands, {rows}x{cols} pixels")
        
        # Verify band count
        if n_bands != 38:
            warnings.warn(f"Expected 38 bands, got {n_bands}. Results may vary.")
        
        # 2. PREPARE DATA
        print("\n[2/10] 🔄 Preparing data...")
        stack_2d = stack.reshape(n_bands, -1).T
        nodata_mask = (stack_2d == -9999).any(axis=1)
        valid_pixels = stack_2d[~nodata_mask]
        
        print(f"✅ Valid pixels: {len(valid_pixels):,} / {rows*cols:,}")
        
        # 3. EXTRACT FEATURES
        print("\n[3/10] 🔄 Extracting important features...")
        band_names = self.extract_band_names(stacked_tif_path)
        selected_indices = self.find_feature_indices(band_names)
        
        # Use hardcoded indices if available
        if len(selected_indices) == 8:
            data_selected = valid_pixels[:, selected_indices]
            feature_names = [band_names[i] if i < len(band_names) else self.important_bands[idx] 
                           for idx, i in enumerate(selected_indices)]
        else:
            # Ultimate fallback
            print("⚠️ Using hardcoded indices directly")
            data_selected = valid_pixels[:, self.feature_indices]
            feature_names = self.important_bands
        
        print(f"✅ Using features: {feature_names}")
        print(f"   Indices: {selected_indices}")
        
        # 4. NORMALIZE DATA
        print("\n[4/10] 🔄 Normalizing data with StandardScaler...")
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_selected)
        print(f"✅ Data normalized")
        
        # 5. APPLY PCA
        print("\n[5/10] 🔄 Applying PCA dimensionality reduction...")
        pca = PCA(n_components=5)
        data_pca = pca.fit_transform(data_scaled)
        print(f"✅ Reduced to {data_pca.shape[1]} components")
        print(f"   Variance explained: {pca.explained_variance_ratio_.sum()*100:.1f}%")
        
        # 6. PERFORM CLUSTERING
        print("\n[6/10] 🔄 Performing K-Means clustering...")
        best_n_clusters = 5
        kmeans = KMeans(n_clusters=best_n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(data_pca)
        
        unique, counts = np.unique(clusters, return_counts=True)
        print(f"✅ Identified {best_n_clusters} clusters:")
        for cluster_id, count in zip(unique, counts):
            print(f"   Cluster {cluster_id}: {count:,} pixels ({count/len(clusters)*100:.1f}%)")
        
        # 7. LOAD GOLD OCCURRENCES
        print("\n[7/10] 🔄 Loading gold occurrences...")
        gold_gdf = gpd.read_file(gold_gpkg_path)
        print(f"✅ Loaded {len(gold_gdf)} gold points")
        
        # 8. SAMPLE CLUSTERS AT GOLD LOCATIONS
        print("\n[8/10] 🔄 Sampling clusters at gold locations...")
        gold_clusters = []
        
        for idx, row in gold_gdf.iterrows():
            lon, lat = row.geometry.x, row.geometry.y
            
            try:
                py, px = rowcol(transform, lon, lat)
            except Exception:
                continue
            
            if 0 <= py < rows and 0 <= px < cols:
                pixel_idx = py * cols + px
                
                if not nodata_mask[pixel_idx]:
                    valid_idx = pixel_idx - nodata_mask[:pixel_idx].sum()
                    if valid_idx < len(clusters):
                        gold_clusters.append(clusters[valid_idx])
        
        print(f"✅ Sampled {len(gold_clusters)} gold locations")
        
        gold_cluster_ids = np.unique(gold_clusters) if gold_clusters else np.array([])
        print(f"   Gold-bearing clusters: {sorted(gold_cluster_ids)}")
        
        # 9. CALCULATE METRICS
        print("\n[9/10] 🔄 Calculating detection metrics...")
        gold_in_anomaly = len(gold_clusters)
        detection_rate = 100.0 if len(gold_clusters) > 0 else 0
        
        # CREATE MAPS
        cluster_map = np.full(rows * cols, -9999, dtype=np.int16)
        cluster_map[~nodata_mask] = clusters
        cluster_map_2d = cluster_map.reshape(rows, cols)
        
        anomaly_map = np.zeros_like(cluster_map_2d, dtype=np.float32)
        for cluster_id in gold_cluster_ids:
            anomaly_map[cluster_map_2d == cluster_id] = 1.0
        anomaly_map[cluster_map_2d == -9999] = -9999
        
        valid_anomaly = anomaly_map[anomaly_map != -9999]
        anomalous_pixels = (valid_anomaly == 1.0).sum()
        
        # CLUSTER STATISTICS
        cluster_stats = []
        for cluster_id, count in zip(unique, counts):
            gold_in_cluster = sum([c == cluster_id for c in gold_clusters])
            is_anomalous = cluster_id in gold_cluster_ids
            
            cluster_stats.append({
                'Cluster': int(cluster_id),
                'Size': int(count),
                'Percentage': float(count / len(clusters) * 100),
                'Gold Points': int(gold_in_cluster),
                'Status': 'HIGH PRIORITY' if is_anomalous else 'Normal'
            })
        
        cluster_df = pd.DataFrame(cluster_stats)
        cluster_df = cluster_df.sort_values('Gold Points', ascending=False)
        
        # 10. GENERATE VISUALIZATIONS
        print("\n[10/10] 🔄 Generating visualizations...")
        visualizations = {}
        
        # VIZ 1: Main Detection Results
        fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        masked_clusters = np.ma.masked_where(cluster_map_2d == -9999, cluster_map_2d)
        im1 = ax1.imshow(masked_clusters, cmap='tab10')
        ax1.set_title('K-Means Clustering Results', fontsize=14, fontweight='bold')
        ax1.axis('off')
        plt.colorbar(im1, ax=ax1, label='Cluster ID')
        
        for idx, row in gold_gdf.iterrows():
            try:
                py, px = rowcol(transform, row.geometry.x, row.geometry.y)
                ax1.plot(px, py, 'r*', markersize=15, markeredgecolor='black', markeredgewidth=1)
            except:
                pass
        
        masked_anomaly = np.ma.masked_where(anomaly_map == -9999, anomaly_map)
        im2 = ax2.imshow(masked_anomaly, cmap='RdYlGn', vmin=0, vmax=1)
        ax2.set_title('Gold Anomaly Detection', fontsize=14, fontweight='bold')
        ax2.axis('off')
        cbar = plt.colorbar(im2, ax=ax2, label='Anomaly Score')
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(['Normal', 'Anomalous'])
        
        for idx, row in gold_gdf.iterrows():
            try:
                py, px = rowcol(transform, row.geometry.x, row.geometry.y)
                ax2.plot(px, py, 'r*', markersize=15, markeredgecolor='black', markeredgewidth=1)
            except:
                pass
        
        plt.suptitle('Unsupervised Gold Anomaly Detection - Buhweju, Uganda',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        buf1 = io.BytesIO()
        plt.savefig(buf1, format='png', dpi=300, bbox_inches='tight')
        buf1.seek(0)
        visualizations['main_detection'] = buf1
        plt.close()
        
        # VIZ 2: Cluster Statistics
        fig2, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        axes[0, 0].bar(cluster_df['Cluster'], cluster_df['Gold Points'],
                      color=["#4D4DCF" if x > 0 else '#92B775' for x in cluster_df['Gold Points']])
        axes[0, 0].set_xlabel('Cluster ID', fontweight='bold')
        axes[0, 0].set_ylabel('Number of Gold Points', fontweight='bold')
        axes[0, 0].set_title('Gold Points Distribution', fontweight='bold')
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        for idx, row in cluster_df.iterrows():
            if row['Gold Points'] > 0:
                axes[0, 0].text(row['Cluster'], row['Gold Points'] + 0.5,
                              f"{int(row['Gold Points'])}", ha='center', fontweight='bold')
        
        colors = ['#133215' if row['Cluster'] in gold_cluster_ids else '#92B775'
                 for idx, row in cluster_df.iterrows()]
        axes[0, 1].barh(cluster_df['Cluster'].astype(str), cluster_df['Percentage'], color=colors)
        axes[0, 1].set_xlabel('Percentage (%)', fontweight='bold')
        axes[0, 1].set_ylabel('Cluster ID', fontweight='bold')
        axes[0, 1].set_title('Cluster Coverage', fontweight='bold')
        axes[0, 1].grid(axis='x', alpha=0.3)
        
        priority_clusters = cluster_df[cluster_df['Gold Points'] > 0]
        if len(priority_clusters) > 0:
            axes[1, 0].barh(priority_clusters['Cluster'].astype(str),
                           priority_clusters['Gold Points'], color="#4BA3CD")
            axes[1, 0].set_xlabel('Gold Points', fontweight='bold')
            axes[1, 0].set_ylabel('Cluster ID', fontweight='bold')
            axes[1, 0].set_title('HIGH PRIORITY CLUSTERS', fontweight='bold', color="#DF692E")
            axes[1, 0].grid(axis='x', alpha=0.3)
            
            for idx, row in priority_clusters.iterrows():
                axes[1, 0].text(row['Gold Points'] + 0.3, row['Cluster'],
                              f"{int(row['Gold Points'])} points", va='center')
        else:
            axes[1, 0].text(0.5, 0.5, 'No priority clusters', ha='center', va='center')
            axes[1, 0].axis('off')
        
        anomaly_coverage = [anomalous_pixels, len(valid_anomaly) - anomalous_pixels]
        colors_pie = ["#3DA144", '#F3E8D3']
        explode = (0.05, 0)
        axes[1, 1].pie(anomaly_coverage, labels=['Anomalous (Gold-bearing)', 'Normal'],
                      autopct='%1.1f%%', colors=colors_pie, explode=explode,
                      textprops={'fontweight': 'bold'})
        axes[1, 1].set_title('Area Coverage: Anomalous vs Normal', fontweight='bold')
        
        plt.tight_layout()
        buf2 = io.BytesIO()
        plt.savefig(buf2, format='png', dpi=300, bbox_inches='tight')
        buf2.seek(0)
        visualizations['cluster_stats'] = buf2
        plt.close()
        
        # VIZ 3: PCA Feature Contributions
        fig3, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        pca_loadings = pca.components_
        loadings_df = pd.DataFrame(
            pca_loadings.T,
            columns=[f'PC{i+1}' for i in range(pca.n_components_)],
            index=feature_names
        )
        
        for i in range(min(5, pca.n_components_)):
            pc_loadings = loadings_df[f'PC{i+1}'].abs().sort_values(ascending=False)
            axes[i].barh(range(len(pc_loadings)), pc_loadings.values, color='#92B775')
            axes[i].set_yticks(range(len(pc_loadings)))
            axes[i].set_yticklabels(pc_loadings.index)
            axes[i].set_xlabel('Absolute Loading', fontweight='bold')
            axes[i].set_title(f'PC{i+1} - Feature Contributions\n({pca.explained_variance_ratio_[i]*100:.1f}% variance)',
                            fontweight='bold')
            axes[i].grid(axis='x', alpha=0.3)
            axes[i].get_children()[0].set_color("#95D2DE")
        
        axes[5].bar(range(1, pca.n_components_+1), pca.explained_variance_ratio_*100, color="#37DA42")
        axes[5].set_xlabel('Principal Component', fontweight='bold')
        axes[5].set_ylabel('Variance Explained (%)', fontweight='bold')
        axes[5].set_title('PCA - Variance Explained by Each Component', fontweight='bold')
        axes[5].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        buf3 = io.BytesIO()
        plt.savefig(buf3, format='png', dpi=300, bbox_inches='tight')
        buf3.seek(0)
        visualizations['pca_contributions'] = buf3
        plt.close()
        
        # VIZ 4: Terrain Analysis (Slope)
        print("   Generating terrain analysis...")
        slope_index = 11  # Band 12 is 'slope' (index 11 in 0-based)
        
        if n_bands > slope_index:
            slope_data = stack[slope_index].flatten()
            slope_valid = slope_data[~nodata_mask]
            
            # Compute slope statistics
            slope_stats = {
                'Mean': np.mean(slope_valid),
                'Median': np.median(slope_valid),
                'Std': np.std(slope_valid),
                'Min': np.min(slope_valid),
                'Max': np.max(slope_valid)
            }
            
            # Terrain classification
            if slope_stats['Mean'] < 5:
                terrain_class = "FLAT TERRAIN - Low relief, gentle slopes"
            elif slope_stats['Mean'] < 15:
                terrain_class = "MODERATE TERRAIN - Rolling hills, moderate relief"
            elif slope_stats['Mean'] < 30:
                terrain_class = "HIGH TERRAIN - Steep slopes, significant relief"
            else:
                terrain_class = "VERY HIGH TERRAIN - Mountainous, very steep slopes"
            
            # Create terrain visualization
            fig4, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Histogram
            axes[0].hist(slope_valid, bins=50, color='#92B775', edgecolor='black', alpha=0.7)
            axes[0].axvline(slope_stats['Mean'], color='#133215', linestyle='--', 
                          linewidth=2, label=f"Mean: {slope_stats['Mean']:.1f}°")
            axes[0].axvline(slope_stats['Median'], color='red', linestyle='--', 
                          linewidth=2, label=f"Median: {slope_stats['Median']:.1f}°")
            axes[0].set_xlabel('Slope (degrees)', fontweight='bold')
            axes[0].set_ylabel('Frequency', fontweight='bold')
            axes[0].set_title('Slope Distribution - AOI Terrain', fontweight='bold')
            axes[0].legend()
            axes[0].grid(alpha=0.3)
            
            # Slope map
            slope_2d = slope_data.reshape(rows, cols)
            slope_2d_masked = np.ma.masked_where(slope_2d == -9999, slope_2d)
            im = axes[1].imshow(slope_2d_masked, cmap='terrain')
            axes[1].set_title('Slope Map', fontweight='bold')
            axes[1].axis('off')
            plt.colorbar(im, ax=axes[1], label='Slope (degrees)')
            
            plt.tight_layout()
            buf4 = io.BytesIO()
            plt.savefig(buf4, format='png', dpi=300, bbox_inches='tight')
            buf4.seek(0)
            visualizations['terrain_analysis'] = buf4
            plt.close()
            
            results['terrain_classification'] = terrain_class
            results['slope_stats'] = slope_stats
        else:
            print("   ⚠️ Slope band not found, skipping terrain analysis")
            visualizations['terrain_analysis'] = None
            results['terrain_classification'] = "Unknown - slope data not available"
        
        # VIZ 5: Comprehensive Summary Dashboard
        print("   Generating comprehensive summary dashboard...")
        fig5 = plt.figure(figsize=(18, 10))
        gs = fig5.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Title
        fig5.suptitle('GOLD ANOMALY DETECTION - COMPREHENSIVE ANALYSIS',
                     fontsize=16, fontweight='bold', y=0.98)
        
        # (1) Detection performance
        ax1 = fig5.add_subplot(gs[0, 0])
        categories = ['Detected', 'Total']
        values = [gold_in_anomaly, len(gold_clusters)]
        ax1.bar(categories, values, color=['#f3e8d3', '#92B775'])
        ax1.set_ylabel('Gold Points', fontweight='bold')
        ax1.set_title(f'Detection Rate: {detection_rate:.0f}%', fontweight='bold', color='#103713')
        ax1.grid(axis='y', alpha=0.3)
        for i, v in enumerate(values):
            ax1.text(i, v + 0.5, str(v), ha='center', fontweight='bold')
        
        # (2) Cluster distribution
        ax2 = fig5.add_subplot(gs[0, 1])
        cluster_sizes = [cluster_df[cluster_df['Cluster']==i]['Size'].values[0] 
                        for i in range(best_n_clusters)]
        colors_c = ['#f3e8d3' if i in gold_cluster_ids else '#92B775' 
                   for i in range(best_n_clusters)]
        ax2.pie(cluster_sizes, labels=[f'C{i}' for i in range(best_n_clusters)],
               autopct='%1.1f%%', colors=colors_c)
        ax2.set_title('Cluster Distribution', fontweight='bold')
        
        # (3) Gold concentration
        ax3 = fig5.add_subplot(gs[0, 2])
        gold_cluster_data = cluster_df[cluster_df['Gold Points'] > 0]
        if len(gold_cluster_data) > 0:
            ax3.barh(gold_cluster_data['Cluster'].astype(str), 
                    gold_cluster_data['Gold Points'], color='#133456')
            ax3.set_xlabel('Gold Points', fontweight='bold')
            ax3.set_ylabel('Cluster', fontweight='bold')
            ax3.set_title('Gold Concentration by Cluster', fontweight='bold')
            ax3.grid(axis='x', alpha=0.3)
        
        # (4) Anomalous area coverage
        ax4 = fig5.add_subplot(gs[1, :2])
        coverage_data = [anomalous_pixels/1000, (len(valid_anomaly)-anomalous_pixels)/1000]
        ax4.barh(['Anomalous (High Priority)', 'Normal'], coverage_data, 
                color=['#92b775', '#F3E8D3'])
        ax4.set_xlabel('Area (×1000 pixels)', fontweight='bold')
        ax4.set_title('Exploration Priority Areas', fontweight='bold', fontsize=12)
        ax4.grid(axis='x', alpha=0.3)
        for i, v in enumerate(coverage_data):
            total = coverage_data[0] + coverage_data[1]
            ax4.text(v + 10, i, f'{v:.1f}K ({v/total*100:.1f}%)',
                    va='center', fontweight='bold')
        
        # (5) PCA variance
        ax5 = fig5.add_subplot(gs[1, 2])
        cumulative_var = np.cumsum(pca.explained_variance_ratio_) * 100
        ax5.plot(range(1, len(cumulative_var)+1), cumulative_var, 
                marker='o', color='#133215', linewidth=2)
        ax5.fill_between(range(1, len(cumulative_var)+1), cumulative_var, 
                        alpha=0.3, color='#92B775')
        ax5.set_xlabel('Components', fontweight='bold')
        ax5.set_ylabel('Cumulative Variance (%)', fontweight='bold')
        ax5.set_title('PCA Explained Variance', fontweight='bold')
        ax5.grid(alpha=0.3)
        ax5.axhline(y=90, color='red', linestyle='--', alpha=0.5)
        
        # (6) Statistics table
        ax6 = fig5.add_subplot(gs[2, :])
        ax6.axis('off')
        
        terrain_text = results.get('terrain_classification', 'Unknown')
        stats_text = f"""
KEY FINDINGS & RECOMMENDATIONS:

✓ DETECTION PERFORMANCE: {detection_rate:.0f}% of gold occurrences detected ({gold_in_anomaly}/{len(gold_clusters)} points)
✓ HIGH PRIORITY CLUSTERS: {len(gold_cluster_ids)} clusters contain gold (Clusters: {', '.join(map(str, sorted(gold_cluster_ids)))})
✓ ANOMALOUS AREA: {anomalous_pixels:,} pixels ({anomalous_pixels/len(valid_anomaly)*100:.1f}% of valid area)
✓ DIMENSIONALITY: {pca.n_components_} PCA components capture {pca.explained_variance_ratio_.sum()*100:.1f}% of variance
✓ TERRAIN: {terrain_text}

EXPLORATION RECOMMENDATIONS:
→ Focus on Cluster {cluster_df.iloc[0]['Cluster']}: Contains {int(cluster_df.iloc[0]['Gold Points'])} gold points
→ Anomalous zones cover {anomalous_pixels/len(valid_anomaly)*100:.1f}% of study area
→ Prioritize field work in high-priority clusters
"""
        
        ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes,
                fontsize=10, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='#F3E8D3', alpha=0.8))
        
        plt.tight_layout()
        buf5 = io.BytesIO()
        plt.savefig(buf5, format='png', dpi=300, bbox_inches='tight')
        buf5.seek(0)
        visualizations['comprehensive_summary'] = buf5
        plt.close()
        
        print("✅ All visualizations generated!")
        
        # COMPILE RESULTS
        results = {
            'detection_rate': float(detection_rate),
            'gold_points_detected': int(gold_in_anomaly),
            'gold_points_total': len(gold_gdf),
            'anomalous_percentage': float(anomalous_pixels / len(valid_anomaly) * 100),
            'pca_variance_explained': float(pca.explained_variance_ratio_.sum() * 100),
            'pca_components': int(pca.n_components_),
            'cluster_statistics': cluster_stats,
            'cluster_map': cluster_map_2d,
            'anomaly_map': anomaly_map,
            'transform': transform,
            'profile': profile,
            'gold_locations': [(row.geometry.x, row.geometry.y) for idx, row in gold_gdf.iterrows()],
            'feature_names': feature_names,
            'visualizations': visualizations,
            'gold_cluster_ids': sorted(gold_cluster_ids)
        }
        
        print("\n" + "="*70)
        print("✅ ANALYSIS COMPLETE!")
        print("="*70)
        print(f"   Detection Rate: {detection_rate:.1f}%")
        print(f"   Anomalous Area: {anomalous_pixels/len(valid_anomaly)*100:.1f}%")
        print(f"   Gold Clusters: {sorted(gold_cluster_ids)}")
        print("="*70 + "\n")
        
        return results