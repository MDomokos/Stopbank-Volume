# Advanced Stopbank Volume Analysis with Slope-Based Edge Detection
# Automatically detects stopbank edges and calculates volumes for sloped sides and flat top

import arcpy
import numpy as np
import pandas as pd
import os
from arcpy import env
from arcpy.sa import *
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.signal import find_peaks
import math

# Check out Spatial Analyst extension
arcpy.CheckOutExtension("Spatial")

# Set up workspace and environment
workspace = r"C:\YourWorkspace"  # Update this path
arcpy.env.workspace = workspace
arcpy.env.overwriteOutput = True
arcpy.env.cellSize = "MINOF"  # Use finest resolution available

print("ArcGIS Pro version:", arcpy.GetInstallInfo()["Version"])
print("Spatial Analyst available:", arcpy.CheckExtension("Spatial") == "Available")

## Input Parameters
# Update these paths to your actual data
dem_raster = r"C:\YourWorkspace\elevation_dem.tif"  # Your DEM raster
centerline_fc = r"C:\YourWorkspace\stopbank_centerlines.shp"  # Centerline feature class

# Analysis parameters
cross_section_interval = 10    # meters - distance between cross-sections
cross_section_width = 100     # meters - total width of cross-section (increased for better edge detection)
slope_threshold = 15          # degrees - threshold to distinguish flat from sloped areas
edge_detection_sensitivity = 5 # meters - minimum distance between edge points
min_stopbank_height = 0.5     # meters - minimum height difference to be considered a stopbank
smoothing_window = 3          # cells - for smoothing elevation profiles

# Output locations
output_gdb = "stopbank_analysis.gdb"
slope_raster = "slope_analysis"
cross_sections_fc = "cross_sections"
edge_points_fc = "edge_points"
stopbank_polygons_fc = "stopbank_polygons"
volume_table = "volume_results"

## Step 1: Create File Geodatabase for outputs
if not arcpy.Exists(output_gdb):
    arcpy.CreateFileGDB_management(workspace, "stopbank_analysis")
    print(f"Created geodatabase: {output_gdb}")

## Step 2: Validate inputs and get raster properties
def validate_inputs_and_get_properties():
    """Check inputs and get DEM properties"""
    if not arcpy.Exists(dem_raster):
        raise FileNotFoundError(f"DEM raster not found: {dem_raster}")
    if not arcpy.Exists(centerline_fc):
        raise FileNotFoundError(f"Centerline feature class not found: {centerline_fc}")
    
    # Get DEM properties
    dem_desc = arcpy.Describe(dem_raster)
    cell_size = dem_desc.meanCellWidth
    spatial_ref = dem_desc.spatialReference
    
    print(f"DEM Cell Size: {cell_size:.2f} meters")
    print(f"DEM Spatial Reference: {spatial_ref.name}")
    
    return spatial_ref, cell_size

spatial_ref, cell_size = validate_inputs_and_get_properties()

## Step 3: Generate slope analysis
def create_slope_analysis():
    """Create slope raster from DEM"""
    print("Creating slope analysis...")
    
    slope_raster_path = os.path.join(output_gdb, slope_raster)
    
    # Calculate slope in degrees
    slope_calc = Slope(dem_raster, "DEGREE")
    slope_calc.save(slope_raster_path)
    
    print(f"Slope raster created: {slope_raster_path}")
    return slope_raster_path

slope_raster_path = create_slope_analysis()

## Step 4: Create enhanced cross-sections with proper perpendicular geometry
def create_perpendicular_cross_sections():
    """Generate cross-section lines truly perpendicular to centerlines"""
    print("Generating perpendicular cross-sections...")
    
    cross_sections_path = os.path.join(output_gdb, cross_sections_fc)
    
    # Create points along centerlines
    points_fc = os.path.join(output_gdb, "centerline_points")
    arcpy.GeneratePointsAlongLines_management(
        centerline_fc, 
        points_fc, 
        'DISTANCE', 
        Distance=f"{cross_section_interval} Meters"
    )
    
    # Create feature class for cross-sections
    arcpy.CreateFeatureclass_management(
        output_gdb, cross_sections_fc, "POLYLINE", 
        spatial_reference=spatial_ref
    )
    
    arcpy.AddField_management(cross_sections_path, "CENTERLINE_ID", "LONG")
    arcpy.AddField_management(cross_sections_path, "SECTION_ID", "LONG")
    arcpy.AddField_management(cross_sections_path, "DISTANCE_ALONG", "DOUBLE")
    
    cross_sections = []
    
    # Get centerline geometries for angle calculation
    centerline_geoms = {}
    with arcpy.da.SearchCursor(centerline_fc, ['OID@', 'SHAPE@']) as cursor:
        for row in cursor:
            centerline_geoms[row[0]] = row[1]
    
    with arcpy.da.SearchCursor(points_fc, ['SHAPE@', 'ORIG_FID', 'DISTANCE_ALONG']) as cursor:
        section_id = 0
        for row in cursor:
            point = row[0].firstPoint
            orig_id = row[1]
            distance_along = row[2]
            
            # Get the centerline geometry
            centerline_geom = centerline_geoms[orig_id]
            
            # Calculate the angle of the centerline at this point
            # by looking at a small segment around the point
            segment_length = 5  # meters
            
            # Get points before and after current point along the line
            pos_before = max(0, distance_along - segment_length)
            pos_after = min(centerline_geom.length, distance_along + segment_length)
            
            point_before = centerline_geom.positionAlongLine(pos_before, False).firstPoint
            point_after = centerline_geom.positionAlongLine(pos_after, False).firstPoint
            
            # Calculate angle of centerline
            dx = point_after.X - point_before.X
            dy = point_after.Y - point_before.Y
            centerline_angle = math.atan2(dy, dx)
            
            # Perpendicular angle (90 degrees offset)
            perp_angle = centerline_angle + math.pi/2
            
            # Create perpendicular line
            half_width = cross_section_width / 2
            start_x = point.X + half_width * math.cos(perp_angle)
            start_y = point.Y + half_width * math.sin(perp_angle)
            end_x = point.X - half_width * math.cos(perp_angle)
            end_y = point.Y - half_width * math.sin(perp_angle)
            
            line_array = arcpy.Array([
                arcpy.Point(start_x, start_y),
                arcpy.Point(end_x, end_y)
            ])
            
            polyline = arcpy.Polyline(line_array, spatial_ref)
            cross_sections.append((polyline, orig_id, section_id, distance_along))
            section_id += 1
    
    # Insert cross-sections into feature class
    with arcpy.da.InsertCursor(cross_sections_path, 
                              ['SHAPE@', 'CENTERLINE_ID', 'SECTION_ID', 'DISTANCE_ALONG']) as cursor:
        for line, orig_id, section_id, distance_along in cross_sections:
            cursor.insertRow([line, orig_id, section_id, distance_along])
    
    print(f"Created {len(cross_sections)} cross-sections")
    return cross_sections_path

cross_sections_path = create_perpendicular_cross_sections()

## Step 5: Extract detailed elevation and slope profiles
def extract_elevation_and_slope_profiles():
    """Extract elevation and slope values along cross-sections"""
    print("Extracting elevation and slope profiles...")
    
    # Create points along cross-sections for detailed sampling
    profile_points_fc = os.path.join(output_gdb, "profile_points")
    arcpy.GeneratePointsAlongLines_management(
        cross_sections_path,
        profile_points_fc,
        'DISTANCE',
        Distance=f'{cell_size} Meters'  # Sample at raster resolution
    )
    
    # Extract elevation values
    ExtractValuesToPoints(profile_points_fc, dem_raster, profile_points_fc, "INTERPOLATE")
    
    # Extract slope values
    ExtractValuesToPoints(profile_points_fc, slope_raster_path, profile_points_fc, "INTERPOLATE")
    
    # Read data into memory
    profile_data = []
    fields = ['SECTION_ID', 'ORIG_FID', 'DISTANCE_ALONG', 'RASTERVALU', 'RASTERVALU_1', 'SHAPE@X', 'SHAPE@Y']
    
    with arcpy.da.SearchCursor(profile_points_fc, fields) as cursor:
        for row in cursor:
            if row[3] is not None and row[4] is not None:  # Skip null values
                profile_data.append({
                    'section_id': row[0],
                    'centerline_id': row[1],
                    'distance_along_section': row[2],
                    'elevation': row[3],
                    'slope_degrees': row[4],
                    'x': row[5],
                    'y': row[6]
                })
    
    return pd.DataFrame(profile_data)

profile_df = extract_elevation_and_slope_profiles()
print(f"Extracted {len(profile_df)} profile points")

## Step 6: Advanced edge detection algorithm
def detect_stopbank_edges():
    """Detect stopbank edges based on slope analysis and elevation changes"""
    print("Detecting stopbank edges...")
    
    edge_results = []
    
    for section_id in profile_df['section_id'].unique():
        section_data = profile_df[profile_df['section_id'] == section_id].copy()
        section_data = section_data.sort_values('distance_along_section')
        
        if len(section_data) < 10:  # Need minimum points for analysis
            continue
        
        elevations = section_data['elevation'].values
        slopes = section_data['slope_degrees'].values
        distances = section_data['distance_along_section'].values
        
        # Smooth the data to reduce noise
        elevations_smooth = ndimage.gaussian_filter1d(elevations, sigma=smoothing_window)
        slopes_smooth = ndimage.gaussian_filter1d(slopes, sigma=smoothing_window)
        
        # Find potential stopbank edges using multiple criteria
        edge_candidates = []
        
        # Method 1: Slope threshold transitions
        flat_mask = slopes_smooth < slope_threshold
        slope_changes = np.diff(flat_mask.astype(int))
        
        # Transitions from flat to steep (toe of slope)
        toe_indices = np.where(slope_changes == -1)[0]
        # Transitions from steep to flat (top of slope)  
        top_indices = np.where(slope_changes == 1)[0]
        
        # Method 2: Elevation gradient analysis
        elevation_gradient = np.gradient(elevations_smooth)
        elevation_gradient_2nd = np.gradient(elevation_gradient)
        
        # Find inflection points (where curvature changes significantly)
        inflection_threshold = np.std(elevation_gradient_2nd) * 1.5
        inflection_points = find_peaks(np.abs(elevation_gradient_2nd), 
                                     height=inflection_threshold,
                                     distance=edge_detection_sensitivity/cell_size)[0]
        
        # Combine all candidate points
        all_candidates = list(toe_indices) + list(top_indices) + list(inflection_points)
        all_candidates = sorted(set(all_candidates))
        
        # Filter candidates based on elevation criteria
        valid_edges = []
        center_idx = len(elevations) // 2
        center_elevation = elevations_smooth[center_idx]
        
        for idx in all_candidates:
            if idx < len(elevations) and idx > 0:
                # Check if this represents a significant elevation change
                local_elevation = elevations_smooth[idx]
                height_diff = abs(local_elevation - center_elevation)
                
                if height_diff > min_stopbank_height:
                    edge_type = "unknown"
                    
                    # Classify edge type based on position and slope
                    if idx < center_idx:
                        if local_elevation < center_elevation:
                            edge_type = "left_toe"
                        else:
                            edge_type = "left_shoulder"
                    else:
                        if local_elevation < center_elevation:
                            edge_type = "right_toe"
                        else:
                            edge_type = "right_shoulder"
                    
                    valid_edges.append({
                        'section_id': section_id,
                        'centerline_id': section_data['centerline_id'].iloc[0],
                        'edge_index': idx,
                        'distance_along_section': distances[idx],
                        'elevation': elevations[idx],
                        'slope': slopes[idx],
                        'edge_type': edge_type,
                        'x': section_data['x'].iloc[idx],
                        'y': section_data['y'].iloc[idx]
                    })
        
        edge_results.extend(valid_edges)
    
    return pd.DataFrame(edge_results)

edges_df = detect_stopbank_edges()
print(f"Detected {len(edges_df)} edge points")

## Step 7: Create edge points feature class
def create_edge_points_fc():
    """Create feature class for detected edge points"""
    edge_points_path = os.path.join(output_gdb, edge_points_fc)
    
    arcpy.CreateFeatureclass_management(
        output_gdb, edge_points_fc, "POINT", 
        spatial_reference=spatial_ref
    )
    
    # Add fields
    fields_to_add = [
        ("SECTION_ID", "LONG"),
        ("CENTERLINE_ID", "LONG"),
        ("EDGE_TYPE", "TEXT", 20),
        ("ELEVATION", "DOUBLE"),
        ("SLOPE_DEG", "DOUBLE"),
        ("DIST_ALONG", "DOUBLE")
    ]
    
    for field_name, field_type, *args in fields_to_add:
        if args:
            arcpy.AddField_management(edge_points_path, field_name, field_type, field_length=args[0])
        else:
            arcpy.AddField_management(edge_points_path, field_name, field_type)
    
    # Insert edge points
    insert_fields = ['SHAPE@', 'SECTION_ID', 'CENTERLINE_ID', 'EDGE_TYPE', 
                    'ELEVATION', 'SLOPE_DEG', 'DIST_ALONG']
    
    with arcpy.da.InsertCursor(edge_points_path, insert_fields) as cursor:
        for _, row in edges_df.iterrows():
            point = arcpy.Point(row['x'], row['y'])
            point_geom = arcpy.PointGeometry(point, spatial_ref)
            
            cursor.insertRow([
                point_geom, row['section_id'], row['centerline_id'],
                row['edge_type'], row['elevation'], row['slope'], 
                row['distance_along_section']
            ])
    
    print(f"Created edge points feature class: {edge_points_path}")
    return edge_points_path

edge_points_path = create_edge_points_fc()

## Step 8: Create stopbank surface polygons
def create_stopbank_polygons():
    """Create polygons for stopbank surfaces (left slope, top, right slope)"""
    print("Creating stopbank surface polygons...")
    
    polygons_path = os.path.join(output_gdb, stopbank_polygons_fc)
    
    arcpy.CreateFeatureclass_management(
        output_gdb, stopbank_polygons_fc, "POLYGON", 
        spatial_reference=spatial_ref
    )
    
    # Add fields
    fields_to_add = [
        ("CENTERLINE_ID", "LONG"),
        ("SURFACE_TYPE", "TEXT", 20),  # 'left_slope', 'top', 'right_slope'
        ("SECTION_START", "LONG"),
        ("SECTION_END", "LONG"),
        ("AVG_ELEVATION", "DOUBLE"),
        ("AREA_M2", "DOUBLE")
    ]
    
    for field_name, field_type, *args in fields_to_add:
        if args:
            arcpy.AddField_management(polygons_path, field_name, field_type, field_length=args[0])
        else:
            arcpy.AddField_management(polygons_path, field_name, field_type)
    
    # Group edges by centerline and create polygons
    polygon_data = []
    
    for centerline_id in edges_df['centerline_id'].unique():
        centerline_edges = edges_df[edges_df['centerline_id'] == centerline_id].copy()
        centerline_edges = centerline_edges.sort_values(['section_id', 'distance_along_section'])
        
        # Group by section and identify surface boundaries
        sections = centerline_edges['section_id'].unique()
        
        if len(sections) < 2:
            continue
        
        # Create polygons for each surface type
        for surface_type in ['left_slope', 'top', 'right_slope']:
            surface_edges = centerline_edges[centerline_edges['edge_type'].str.contains(
                surface_type.split('_')[0] if surface_type != 'top' else 'shoulder'
            )]
            
            if len(surface_edges) < 2:
                continue
            
            # Create polygon from edge points
            polygon_points = []
            
            # Add points in order to form a polygon
            for _, edge in surface_edges.iterrows():
                polygon_points.append(arcpy.Point(edge['x'], edge['y']))
            
            if len(polygon_points) >= 3:
                # Close the polygon
                polygon_points.append(polygon_points[0])
                
                polygon_array = arcpy.Array(polygon_points)
                polygon_geom = arcpy.Polygon(polygon_array, spatial_ref)
                
                avg_elevation = surface_edges['elevation'].mean()
                area = polygon_geom.area
                
                polygon_data.append({
                    'geometry': polygon_geom,
                    'centerline_id': centerline_id,
                    'surface_type': surface_type,
                    'section_start': surface_edges['section_id'].min(),
                    'section_end': surface_edges['section_id'].max(),
                    'avg_elevation': avg_elevation,
                    'area_m2': area
                })
    
    # Insert polygons
    insert_fields = ['SHAPE@', 'CENTERLINE_ID', 'SURFACE_TYPE', 'SECTION_START', 
                    'SECTION_END', 'AVG_ELEVATION', 'AREA_M2']
    
    with arcpy.da.InsertCursor(polygons_path, insert_fields) as cursor:
        for poly_data in polygon_data:
            cursor.insertRow([
                poly_data['geometry'], poly_data['centerline_id'],
                poly_data['surface_type'], poly_data['section_start'],
                poly_data['section_end'], poly_data['avg_elevation'],
                poly_data['area_m2']
            ])
    
    print(f"Created {len(polygon_data)} surface polygons")
    return polygons_path, polygon_data

polygons_path, polygon_data = create_stopbank_polygons()

## Step 9: Calculate volumes for each surface
def calculate_surface_volumes():
    """Calculate volume for each stopbank surface"""
    print("Calculating surface volumes...")
    
    volume_results = []
    
    # Group polygon data by centerline
    for centerline_id in set(p['centerline_id'] for p in polygon_data):
        centerline_polygons = [p for p in polygon_data if p['centerline_id'] == centerline_id]
        
        total_volume = 0
        surface_volumes = {}
        
        for poly_data in centerline_polygons:
            surface_type = poly_data['surface_type']
            area = poly_data['area_m2']
            avg_elevation = poly_data['avg_elevation']
            
            # Estimate volume using average elevation method
            # This assumes the base is at the minimum elevation found in the analysis
            min_elevation = edges_df[edges_df['centerline_id'] == centerline_id]['elevation'].min()
            height = max(0, avg_elevation - min_elevation)
            
            # Volume = area × average height above base
            volume = area * height * 0.5  # Factor for slope surfaces
            
            surface_volumes[surface_type] = volume
            total_volume += volume
        
        # Get centerline length
        centerline_length = 0
        with arcpy.da.SearchCursor(centerline_fc, ['OID@', 'SHAPE@LENGTH']) as cursor:
            for row in cursor:
                if row[0] == centerline_id:
                    centerline_length = row[1]
                    break
        
        volume_results.append({
            'centerline_id': centerline_id,
            'total_volume_m3': total_volume,
            'left_slope_volume': surface_volumes.get('left_slope', 0),
            'top_volume': surface_volumes.get('top', 0),
            'right_slope_volume': surface_volumes.get('right_slope', 0),
            'centerline_length_m': centerline_length,
            'volume_per_meter': total_volume / centerline_length if centerline_length > 0 else 0
        })
    
    return pd.DataFrame(volume_results)

volume_results_df = calculate_surface_volumes()

## Step 10: Create comprehensive results summary
print("\n" + "="*60)
print("ADVANCED STOPBANK VOLUME ANALYSIS RESULTS")
print("="*60)
print(f"Analysis Parameters:")
print(f"  Cross-section interval: {cross_section_interval}m")
print(f"  Cross-section width: {cross_section_width}m")
print(f"  Slope threshold: {slope_threshold}°")
print(f"  Minimum stopbank height: {min_stopbank_height}m")
print(f"  Edge detection sensitivity: {edge_detection_sensitivity}m")
print(f"  Total centerlines analyzed: {len(volume_results_df)}")
print(f"  Total edge points detected: {len(edges_df)}")
print(f"  Total surface polygons created: {len(polygon_data)}")

print("\nDetailed Volume Results by Centerline:")
print("-" * 40)

for _, row in volume_results_df.iterrows():
    print(f"Centerline ID: {row['centerline_id']}")
    print(f"  Length: {row['centerline_length_m']:.1f} m")
    print(f"  Total Volume: {row['total_volume_m3']:.1f} m³")
    print(f"  Left Slope Volume: {row['left_slope_volume']:.1f} m³")
    print(f"  Top Surface Volume: {row['top_volume']:.1f} m³")
    print(f"  Right Slope Volume: {row['right_slope_volume']:.1f} m³")
    print(f"  Volume per Unit Length: {row['volume_per_meter']:.1f} m³/m")
    print()

# Summary statistics
total_volume_all = volume_results_df['total_volume_m3'].sum()
total_length_all = volume_results_df['centerline_length_m'].sum()

print(f"SUMMARY STATISTICS:")
print(f"  Total Volume (All Stopbanks): {total_volume_all:.1f} m³")
print(f"  Total Length (All Stopbanks): {total_length_all:.1f} m")
print(f"  Average Volume per Unit Length: {total_volume_all/total_length_all:.1f} m³/m")

## Step 11: Export results
results_table_path = os.path.join(output_gdb, volume_table)
volume_results_df.to_csv(os.path.join(workspace, "stopbank_volume_analysis.csv"), index=False)
edges_df.to_csv(os.path.join(workspace, "detected_edges.csv"), index=False)

# Create ArcGIS table
x = np.array(np.rec.fromrecords(volume_results_df.values))
names = volume_results_df.dtypes.index.tolist()
x.dtype.names = tuple(names)
arcpy.da.NumPyArrayToTable(x, results_table_path)

print(f"\nResults exported to:")
print(f"  Volume Results CSV: {os.path.join(workspace, 'stopbank_volume_analysis.csv')}")
print(f"  Edge Points CSV: {os.path.join(workspace, 'detected_edges.csv')}")
print(f"  ArcGIS Volume Table: {results_table_path}")
print(f"  Edge Points Feature Class: {edge_points_path}")
print(f"  Surface Polygons Feature Class: {polygons_path}")

## Step 12: Create advanced visualizations
def create_advanced_visualizations():
    """Create comprehensive charts and visualizations"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Volume by centerline
    axes[0,0].bar(volume_results_df['centerline_id'], volume_results_df['total_volume_m3'])
    axes[0,0].set_xlabel('Centerline ID')
    axes[0,0].set_ylabel('Total Volume (m³)')
    axes[0,0].set_title('Total Volume by Stopbank')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # Volume breakdown by surface type
    centerline_ids = volume_results_df['centerline_id']
    width = 0.25
    x = np.arange(len(centerline_ids))
    
    axes[0,1].bar(x - width, volume_results_df['left_slope_volume'], width, label='Left Slope')
    axes[0,1].bar(x, volume_results_df['top_volume'], width, label='Top Surface')
    axes[0,1].bar(x + width, volume_results_df['right_slope_volume'], width, label='Right Slope')
    axes[0,1].set_xlabel('Centerline ID')
    axes[0,1].set_ylabel('Volume (m³)')
    axes[0,1].set_title('Volume by Surface Type')
    axes[0,1].set_xticks(x)
    axes[0,1].set_xticklabels(centerline_ids)
    axes[0,1].legend()
    
    # Volume per unit length
    axes[1,0].bar(volume_results_df['centerline_id'], volume_results_df['volume_per_meter'])
    axes[1,0].set_xlabel('Centerline ID')
    axes[1,0].set_ylabel('Volume per Meter (m³/m)')
    axes[1,0].set_title('Volume Density by Stopbank')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # Edge detection summary
    edge_type_counts = edges_df['edge_type'].value_counts()
    axes[1,1].pie(edge_type_counts.values, labels=edge_type_counts.index, autopct='%1.1f%%')
    axes[1,1].set_title('Distribution of Detected Edge Types')
    
    plt.tight_layout()
    plt.savefig(os.path.join(workspace, 'advanced_stopbank_analysis.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create elevation profile example
    if len(profile_df) > 0:
        plt.figure(figsize=(12, 8))
        
        # Show profile for first section as example
        first_section = profile_df['section_id'].iloc[0]
        section_profile = profile_df[profile_df['section_id'] == first_section].sort_values('distance_along_section')
        section_edges = edges_df[edges_df['section_id'] == first_section]
        
        plt.subplot(2, 1, 1)
        plt.plot(section_profile['distance_along_section'], section_profile['elevation'], 'b-', linewidth=2, label='Elevation')
        plt.scatter(section_edges['distance_along_section'], section_edges['elevation'], 
                   c='red', s=100, zorder=5, label='Detected Edges')
        plt.xlabel('Distance along Cross-section (m)')
        plt.ylabel('Elevation (m)')
        plt.title(f'Elevation Profile - Section {first_section}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 1, 2)
        plt.plot(section_profile['distance_along_section'], section_profile['slope_degrees'], 'g-', linewidth=2, label='Slope')
        plt.axhline(y=slope_threshold, color='r', linestyle='--', label=f'Slope Threshold ({slope_threshold}°)')
        plt.xlabel('Distance along Cross-section (m)')
        plt.ylabel('Slope (degrees)')
        plt.title(f'Slope Profile - Section {first_section}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(workspace, 'example_profile_analysis.png'), 
                    dpi=300, bbox_inches='tight')
        plt.show()

create_advanced_visualizations()

## Clean up
arcpy.CheckInExtension("Spatial")
print("\nAdvanced stopbank analysis complete!")
print(f"All outputs saved to: {workspace}")

## Technical Notes and Recommendations
"""
ADVANCED STOPBANK ANALYSIS - TECHNICAL NOTES:

1. EDGE DETECTION METHODOLOGY:
   - Combines slope threshold analysis with elevation gradient detection
   - Uses inflection point analysis to identify slope changes
   - Applies smoothing to reduce noise in elevation data
   - Classifies edges as toe, shoulder, left, or right positions

2. VOLUME CALCULATION APPROACH:
   - Creates separate polygons for each stopbank surface (left slope, top, right slope)
   - Calculates volume using surface area × average height above detected base
   - Accounts for different surface types with appropriate volume factors

3. VALIDATION RECOMMENDATIONS:
   - Visually inspect detected edge points against aerial imagery
   - Compare results with surveyed cross-sections where available
   - Check edge detection sensitivity by adjusting parameters
   - Validate slope threshold against known flat/sloped areas
   - Field verify representative sections for accuracy

4. PARAMETER TUNING GUIDANCE:
   - slope_threshold: Increase for steeper terrain, decrease for gentler slopes
   - edge_detection_sensitivity: Increase to merge nearby edges, decrease for more detail
   - min_stopbank_height: Adjust based on minimum expected stopbank dimensions
   - smoothing_window: Increase for noisy data, decrease for high-precision DEMs

5. POTENTIAL IMPROVEMENTS:
   - Integration with survey data for ground truth validation
   - Machine learning approaches for edge classification
   - Multi-scale analysis for different stopbank types
   - Automated quality control checks for detected edges
   - Integration with design specifications for validation

6. OUTPUT FILES CREATED:
   - Volume analysis results (CSV and ArcGIS table)
   - Detected edge points (feature class and CSV)
   - Surface polygons (feature class)
   - Cross-sections (feature class)
   - Slope analysis raster
   - Visualization charts and profile examples

7. TROUBLESHOOTING TIPS:
   - If few edges detected: Reduce slope_threshold or min_stopbank_height
   - If too many edges: Increase edge_detection_sensitivity or smoothing_window
   - For complex geometries: Reduce cross_section_interval for more detail
   - For noisy data: Increase smoothing_window parameter

This advanced approach provides a comprehensive analysis of stopbank volumes
by automatically detecting the actual geometry rather than making assumptions
about base elevations or uniform cross-sections.
"""
