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
sentinel_path = "data/tif/TanHoi_final.tif"
band_names = ['B4', 'B3', 'B2', 'B8', 'B11', 'B12', 'NDVI', 'EVI', 'SAVI', 'NDWI', 'NBR', 'CIgreen']

# open rgb
with rasterio.open(sentinel_path) as src:
    # transform to CRS
    gdf = gdf.to_crs(src.crs)

    # create list for training data
    training_data = []
    
    # Duyệt qua từng điểm và lấy giá trị pixel tương ứng từ ảnh vệ tinh
    for idx, row in gdf.iterrows():
        point = row.geometry  # lấy tọa độ điểm (long, lat)

        # Tìm chỉ số hàng và cột của pixel tương ứng với tọa độ điểm
        row_idx, col_idx = src.index(point.x, point.y)
        
        # Đọc giá trị pixel cho tất cả các băng phổ và chỉ số tại vị trí này
        pixel_values = src.read(window=((row_idx, row_idx+1), (col_idx, col_idx+1))).flatten()
        
        # Lưu trữ các giá trị pixel cùng với nhãn và tọa độ
        training_data.append({
            'long': point.x,
            'lat': point.y,
            'label': row['label'],  # gán nhãn cho sắn hoặc không sắn
            **{band_names[i]: pixel_values[i] for i in range(len(pixel_values))}
        })

# create df from training_data
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
