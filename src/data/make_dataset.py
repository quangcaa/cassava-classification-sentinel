import geopandas as gpd
import rasterio
import pandas as pd
from sklearn.model_selection import train_test_split

# load san and non-san data
san_data = pd.read_csv('data/processed/07_11_san_dat_fixed.csv')
non_san_data = pd.read_csv('data/processed/07_11_non_san_dat_fixed.csv')

# combine
data = pd.concat([san_data, non_san_data], ignore_index=True)

# create GeoDataFrame
gdf = gpd.GeoDataFrame(
    data,
    geometry=gpd.points_from_xy(data['long'], data['lat']),
    crs="EPSG:4326"
)

# sentinel tif path
rgb_path = "data/tif/Sentinel2_TanHoi_RGB.tif"
ndvi_path = "data/tif/Sentinel2_TanHoi_NDVI.tif"

# open rgb
with rasterio.open(rgb_path) as src_rgb:
    # transform to CRS
    gdf = gdf.to_crs(src_rgb.crs)
    
    # open ndvi
    with rasterio.open(ndvi_path) as src_ndvi:
        # check CRS === rgb
        assert src_ndvi.crs == src_rgb.crs, "CRS of RGB and NDVI images do not match"
         
        print(f"Number of bands in RGB image: {src_rgb.count}")
        
        # create list for training data
        training_data = []

        # iterate through each point in gdf and get the corresponding pixel values
        for idx, row in gdf.iterrows():
            point = row.geometry  # get the point (coordinates)

            # get row, column indices of the pixel corresponding to the coordinates
            row_idx_rgb, col_idx_rgb = src_rgb.index(point.x, point.y)
            row_idx_ndvi, col_idx_ndvi = src_ndvi.index(point.x, point.y)
            
            # read the pixel values at this location for all bands
            pixel_values_rgb = src_rgb.read([1, 2, 3], window=((row_idx_rgb, row_idx_rgb+1), (col_idx_rgb, col_idx_rgb+1))).flatten()
            pixel_values_ndvi = src_ndvi.read(1, window=((row_idx_ndvi, row_idx_ndvi+1), (col_idx_ndvi, col_idx_ndvi+1))).flatten()
            
            # store the pixel values along with coordinates and label
            training_data.append({
                'long': point.x,
                'lat': point.y,
                'label': row['label'],
                **{f'rgb_band_{i+1}': pixel_values_rgb[i] for i in range(len(pixel_values_rgb))},
                **{f'ndvi_band_{i+1}': pixel_values_ndvi[i] for i in range(len(pixel_values_ndvi))}
            })

# create DataFrame from the training data
df_training = pd.DataFrame(training_data)

# split training (80%) , test (20%)
X = df_training.drop(columns=['label'])
y = df_training['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# save dataset
train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)

train_data.to_csv('data/train/train_data.csv', index=False)
test_data.to_csv('data/test/test_data.csv', index=False)

print("Đã tạo thành công các bộ dữ liệu: train_data.csv, test_data.csv")
