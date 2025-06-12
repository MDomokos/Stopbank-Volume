# Advanced Stopbank Volume Analysis

A comprehensive Python script for calculating stopbank/levy volumes using ArcPy with automatic edge detection based on slope analysis.

## Overview

This Jupyter notebook script automatically detects stopbank edges from DEM data and centerline features, then calculates accurate volumes for different stopbank surfaces. Unlike traditional approaches that assume base elevations, this method uses sophisticated slope analysis to identify where the ground is flat versus sloped, automatically detecting stopbank boundaries.

## Features

### 🔍 **Automatic Edge Detection**
- **Multi-Method Approach**: Combines slope threshold analysis, elevation gradient detection, and inflection point analysis
- **Intelligent Classification**: Automatically identifies left slope, top surface, and right slope areas
- **Noise Reduction**: Applies smoothing filters to handle noisy elevation data
- **Quality Control**: Validates detected edges using multiple criteria

### 📐 **Accurate Geometry Analysis**
- **True Perpendicular Cross-Sections**: Uses trigonometry to create properly oriented cross-sections
- **High-Resolution Sampling**: Samples elevation data at DEM resolution for maximum accuracy
- **Surface-Based Volume Calculation**: Calculates separate volumes for each stopbank surface

### 📊 **Comprehensive Outputs**
- **Detailed Volume Breakdown**: Results by surface type (left slope, top, right slope) for each centerline
- **Multiple Export Formats**: CSV files, ArcGIS tables, and feature classes
- **Visual Validation**: Charts showing elevation profiles and detected edges
- **Quality Metrics**: Statistics on edge detection success and analysis parameters

## Requirements

### Software
- **ArcGIS Pro** or **ArcMap** with Spatial Analyst extension
- **Python 3.x** with Jupyter notebook support
- **ArcPy** (included with ArcGIS)

### Python Libraries
```bash
pip install pandas numpy matplotlib scipy
```

### Input Data
- **DEM Raster**: High-resolution digital elevation model (.tif, .img, etc.)
- **Centerline Features**: Polyline feature class representing stopbank centerlines (.shp, feature class)

## Installation & Setup

1. **Clone or download** the notebook script
2. **Install required Python libraries** (see requirements above)
3. **Update file paths** in the script to point to your data:
   ```python
   dem_raster = r"C:\YourWorkspace\elevation_dem.tif"
   centerline_fc = r"C:\YourWorkspace\stopbank_centerlines.shp"
   workspace = r"C:\YourWorkspace"
   ```
4. **Ensure ArcGIS licensing** is available (Spatial Analyst extension required)

## Configuration Parameters

### Key Analysis Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `cross_section_interval` | 10m | Distance between cross-sections along centerlines |
| `cross_section_width` | 100m | Total width of each cross-section |
| `slope_threshold` | 15° | Threshold to distinguish flat from sloped areas |
| `edge_detection_sensitivity` | 5m | Minimum distance between detected edges |
| `min_stopbank_height` | 0.5m | Minimum height difference to be considered a stopbank |
| `smoothing_window` | 3 | Amount of smoothing applied to elevation data |

### Parameter Tuning Guidelines

- **For Steeper Terrain**: Increase `slope_threshold` (20-25°)
- **For Gentler Slopes**: Decrease `slope_threshold` (10-12°)
- **For Noisy Data**: Increase `smoothing_window` (5-7)
- **For High-Precision DEMs**: Decrease `smoothing_window` (1-2)
- **For Detailed Analysis**: Reduce `cross_section_interval` (5m)
- **For Broad Overview**: Increase `cross_section_interval` (20-50m)

## Usage

### Running the Analysis

1. **Open Jupyter Notebook** with ArcGIS Python environment
2. **Load the script** and update parameters as needed
3. **Run all cells** sequentially
4. **Review outputs** and validation charts

### Expected Runtime
- **Small projects** (< 1km centerlines): 2-5 minutes
- **Medium projects** (1-10km centerlines): 5-20 minutes  
- **Large projects** (> 10km centerlines): 20+ minutes

## Output Files

### Generated Files and Locations

| Output | Format | Description |
|--------|---------|-------------|
| `stopbank_volume_analysis.csv` | CSV | Detailed volume results by centerline |
| `detected_edges.csv` | CSV | All detected edge points with classifications |
| `stopbank_analysis.gdb` | File GDB | Contains all spatial outputs |
| `advanced_stopbank_analysis.png` | Image | Comprehensive analysis charts |
| `example_profile_analysis.png` | Image | Sample elevation and slope profiles |

### Feature Classes Created

- **cross_sections**: Generated cross-section lines
- **edge_points**: Detected stopbank edge points
- **stopbank_polygons**: Surface polygons for volume calculation
- **volume_results**: Analysis results table

### Raster Outputs

- **slope_analysis**: Slope raster derived from DEM

## Results Interpretation

### Volume Results Structure

Each centerline will have the following volume components:
- **Left Slope Volume**: Volume of the left-side sloped surface
- **Top Volume**: Volume of the flat top surface
- **Right Slope Volume**: Volume of the right-side sloped surface
- **Total Volume**: Sum of all surface volumes
- **Volume per Unit Length**: Total volume divided by centerline length

### Quality Indicators

- **Edge Points Detected**: Number of edge points found (more indicates better detection)
- **Surface Polygons Created**: Number of surface polygons generated
- **Edge Type Distribution**: Balance of different edge types detected

## Validation & Quality Control

### Recommended Validation Steps

1. **Visual Inspection**
   - Overlay edge points on aerial imagery
   - Check cross-sections against known stopbank locations
   - Verify surface polygons cover expected areas

2. **Parameter Sensitivity Testing**
   - Run analysis with different slope thresholds
   - Test various smoothing window sizes
   - Compare results for consistency

3. **Field Validation**
   - Survey representative cross-sections
   - Compare calculated volumes with design specifications
   - Validate edge detection against ground truth

### Troubleshooting Common Issues

| Issue | Likely Cause | Solution |
|-------|--------------|----------|
| Few edges detected | Slope threshold too high | Reduce `slope_threshold` to 10-12° |
| Too many edges | Sensitivity too low | Increase `edge_detection_sensitivity` |
| Noisy results | Elevation data noisy | Increase `smoothing_window` to 5-7 |
| Missing small features | Height threshold too high | Reduce `min_stopbank_height` |
| Irregular cross-sections | Interval too large | Reduce `cross_section_interval` |

## Advanced Features

### Edge Detection Methodology

The script uses three complementary methods:

1. **Slope Threshold Analysis**: Identifies transitions between flat and steep areas
2. **Elevation Gradient Analysis**: Finds significant elevation changes
3. **Inflection Point Detection**: Locates curvature changes in elevation profiles

### Volume Calculation Approach

- **Surface-Specific Calculation**: Each surface type calculated independently
- **Automatic Base Detection**: Uses minimum elevation from slope analysis
- **Realistic Volume Factors**: Applies appropriate factors for slope vs. flat surfaces

## Technical Notes

### Spatial Reference Requirements
- DEM and centerlines should have matching spatial reference systems
- Script will warn if coordinate systems don't match
- Consider reprojecting data to a projected coordinate system for accurate measurements

### Data Quality Considerations
- **DEM Resolution**: Higher resolution DEMs (1m or better) provide best results
- **Centerline Accuracy**: Centerlines should accurately represent stopbank alignment
- **Data Currency**: Ensure DEM represents current conditions

### Performance Optimization
- Use local file geodatabases for better performance
- Consider tiling large datasets for processing
- Ensure adequate system memory for large raster processing

## Support & Troubleshooting

### Common Error Messages

- **"Spatial Analyst extension not available"**: Check ArcGIS licensing
- **"DEM raster not found"**: Verify file path and format
- **"Memory error during processing"**: Reduce analysis extent or increase system memory

### Getting Help

1. Check parameter settings against recommendations
2. Verify input data quality and formats
3. Review ArcGIS Pro/ArcMap error messages
4. Test with smaller dataset to isolate issues

## Version History

- **v1.0**: Initial release with basic volume calculation
- **v2.0**: Added advanced edge detection and surface classification
- **v2.1**: Enhanced visualization and validation features

## License

This script is provided as-is for educational and research purposes. Please ensure compliance with your organization's software licensing requirements for ArcGIS products.

## Contributing

Suggestions for improvements and bug reports are welcome. Consider testing the script with various stopbank types and DEM qualities to help improve the edge detection algorithms.

---

**Note**: This tool provides estimates based on available DEM data. For critical engineering applications, results should be validated with survey-grade measurements and professional engineering review.
