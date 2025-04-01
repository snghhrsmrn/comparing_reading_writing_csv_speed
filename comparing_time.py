import pandas as pd
import random
import numpy as np
import polars as pl
import dask.dataframe as dd
import time
import string
import os

def generate_test_data(rows):
    return pd.DataFrame({
        'row_id' : range(rows),
    'float_col' : np.random.randn(rows),
    'int_col' : np.random.randint(0, 1000000, rows),
    'str_col' : np.random.choice(random.choices(string.ascii_lowercase, k=10), rows)
    })

# creating dataset for testing
num_rows = 20000000
print(f"Generating data with {num_rows} rows")
test_data = generate_test_data(num_rows)
print("Generating data completed")

# creating a directory for dask output
dask_output_dir = 'dask_data_dir'
os.makedirs(dask_output_dir, exist_ok=True)

# store benchmark results
results = {
    'library': ['Pandas', 'Dask', 'Polars'],
    'write_time': [],
    'read_time': []
}

# benchmark pandas writing
start_time = time.time()
test_data.to_csv('pandas_data.csv', index=False)
pandas_write_time = time.time() - start_time
results['write_time'].append(pandas_write_time)
print(f"Pandas write time: {pandas_write_time}")

# benchmark pandas reading
start_time = time.time()
pd.read_csv('pandas_data.csv')
pandas_read_time = time.time() - start_time
results['read_time'].append(pandas_read_time)
print(f"Pandas read time: {pandas_read_time}")

# benchmark dask writing
start_time = time.time()
ddf = dd.from_pandas(test_data, npartitions=4)
ddf.to_csv(os.path.join(dask_output_dir, 'part-*.csv'), index=False)
dask_write_time = time.time() - start_time
results['write_time'].append(dask_write_time)
print(f"Dask write time: {dask_write_time}")

# benchmark dask reading
start_time = time.time()
ddf = dd.read_csv(os.path.join(dask_output_dir, 'part-*.csv'))
len(ddf.compute())
dask_read_time = time.time() - start_time
results['read_time'].append(dask_read_time)
print(f"Dask read time: {dask_read_time}")

# benchmark polars writing
start_time = time.time()
pl_df = pl.from_pandas(test_data)
pl_df.write_csv('polars_data.csv')
polars_write_time = time.time() - start_time
results['write_time'].append(polars_write_time)
print(f"Polars write time: {polars_write_time}")

# benchmark polars reading
start_time = time.time()
pl.read_csv('polars_data.csv')
polars_read_time = time.time() - start_time
results['read_time'].append(polars_read_time)
print(f"Polars read time: {polars_read_time}")


print(results)