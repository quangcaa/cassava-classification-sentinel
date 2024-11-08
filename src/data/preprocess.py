import pandas as pd

# True: ko phan san
# load csv
non_san_df = pd.read_csv('data/raw/07_11_san_data.csv')

# drop not necessary columns
non_san_df = non_san_df.drop(columns=['OID_', 'CID'])

# rename columns
non_san_df = non_san_df.rename(columns={'POINT_X': 'long', 'POINT_Y': 'lat'})

# add label
non_san_df['label'] = 'True'

# save the modified df back to csv
non_san_df.to_csv('data/processed/07_11_san_dat_fixed.csv', index=False)


# False: ko phan san
# load csv
non_san_df = pd.read_csv('data/raw/07_11_non_san_data.csv')

# drop not necessary columns
non_san_df = non_san_df.drop(columns=['OID_','Name','FolderPath','SymbolID','AltMode','Base','Snippet','PopupInfo','HasLabel','LabelID', 'POINT_Z'])

# rename columns
non_san_df = non_san_df.rename(columns={'POINT_X': 'long', 'POINT_Y': 'lat'})

# add label
non_san_df['label'] = 'False'

# save the modified df back to csv
non_san_df.to_csv('data/processed/07_11_non_san_dat_fixed.csv', index=False)
