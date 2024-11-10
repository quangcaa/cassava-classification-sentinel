import pandas as pd
import xgboost as xgb
import rasterio
import numpy as np
from rasterio.transform import from_origin
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

# load best model
best_model = xgb.XGBClassifier()
best_model.load_model('best_xgboost_model.json')

# load satellite image
satellite_image_path = 'data/tif/TanHoi_final.tif'

try:
    with rasterio.open(satellite_image_path) as src:
        band_count = src.count
        print(f"Number of bands in the image: {band_count}")
        
        band_names = ['B4', 'B3', 'B2', 'B8', 'B11', 'B12', 'NDVI', 'EVI', 'SAVI', 'NDWI', 'NBR', 'CIgreen']
        if band_count != len(band_names):
            print(f"Warning: Number of bands in image ({band_count}) doesn't match band_names list ({len(band_names)})")
        
        mask = src.dataset_mask()
        
        image_data = []
        for band in range(1, band_count + 1):
            band_data = src.read(band)
            if band_data is None:
                raise ValueError(f"Failed to read band {band}")
            band_data = np.where(mask, band_data, np.nan)
            image_data.append(band_data)
        
        image_data = np.array(image_data)
        transform = src.transform
        crs = src.crs
        
        print(f"Image data shape: {image_data.shape}")

except rasterio.errors.RasterioIOError as e:
    print(f"Error opening the image file: {e}")
    raise
except Exception as e:
    print(f"Unexpected error: {e}")
    raise

# prepare data for prediction
rows, cols = image_data[0].shape
data = []
coords = []
valid_pixels = []

no_data_value = src.nodata
if no_data_value is not None:
    print(f"No data value: {no_data_value}")

for row in range(rows):
    for col in range(cols):
        pixel_values = []
        is_valid = True
        
        for band in range(min(len(band_names), band_count)):
            value = image_data[band][row, col]
            if np.isnan(value) or (no_data_value is not None and value == no_data_value):
                is_valid = False
                break
            pixel_values.append(value)
        
        if is_valid:
            while len(pixel_values) < len(band_names):
                pixel_values.append(0)
            
            data.append(pixel_values)
            coords.append(src.xy(row, col))
            valid_pixels.append((row, col))

print(f"Total valid pixels: {len(valid_pixels)}")

# create DataFrame for prediction
df = pd.DataFrame(data, columns=band_names)

# add coordinates
df['long'] = [coord[0] for coord in coords]
df['lat'] = [coord[1] for coord in coords]

print("\nDataFrame info:")
print(df.info())
print("\nSample of DataFrame:")
print(df.head())
print("\nBasic statistics:")
print(df.describe())

# verify no null values
null_counts = df.isnull().sum()
if null_counts.any():
    print("\nWarning: Null values found in columns:")
    print(null_counts[null_counts > 0])

model_features = [ 'long', 'lat','B4', 'B3', 'B2', 'B8', 'B11', 'B12', 'NDVI', 'EVI', 'SAVI', 'NDWI', 'NBR', 'CIgreen']
df = df[model_features]

# Predict only for valid pixels
try:
    predictions = best_model.predict(df)
    print("Predictions successful!")
except Exception as e:
    print(f"Error during prediction: {e}")
    raise

# create empty prediction map
prediction_map = np.zeros((rows, cols), dtype=np.uint8)

# fill in predictions only for valid pixels
for (row, col), pred in zip(valid_pixels, predictions):
    prediction_map[row, col] = pred

# create image with green=cassava, white=non-cassava, black=invalid/no-data
classified_image = np.zeros((rows, cols, 3), dtype=np.uint8)
classified_image[prediction_map == 1] = [0, 255, 0]    
classified_image[prediction_map == 0] = [255, 255, 255] 

# save as GeoTIFF
output_tif_path = 'classified_map.tif'

try:
    with rasterio.open(
        output_tif_path,
        'w',
        driver='GTiff',
        height=rows,
        width=cols,
        count=3,
        dtype=rasterio.uint8,
        crs=crs,
        transform=transform,
    ) as dst:
        for i in range(3):
            dst.write(classified_image[:, :, i], i + 1)
    print(f"Classified map saved successfully as {output_tif_path}")
except Exception as e:
    print(f"Error saving classified map: {e}")
    raise