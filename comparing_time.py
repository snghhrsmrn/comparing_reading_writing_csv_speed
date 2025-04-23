import pandas as pd
import dask.dataframe as dd
import polars as pl
import numpy as np
import random
import string
import os
import time
import matplotlib.pyplot as plt


def generate_test_data(rows):
    """Generate random test data with specified number of rows"""

    return pd.DataFrame({
        'row_id': range(rows),
        'float_col': np.random.randn(rows),
        'int_col': np.random.randint(0, 1000000, rows),
        'str_col': [''.join(random.choices(string.ascii_lowercase + string.ascii_uppercase, k=10)) for _ in range(rows)]
    })


def run_benchmarks(test_data):
    """Run benchmarks for Pandas, Dask and Polars"""

    # create a directory for dask output
    dask_output_dir = 'dask_output_dir'
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
    print(f"Pandas write time: {pandas_write_time:.4f}s")

    # benchmark pandas reading
    start_time = time.time()
    pd.read_csv('pandas_data.csv')
    pandas_read_time = time.time() - start_time
    results['read_time'].append(pandas_read_time)
    print(f"Pandas read time: {pandas_read_time:.4f}s")

    # benchmark dask writing
    start_time = time.time()
    ddf = dd.from_pandas(test_data, npartitions=4)
    ddf.to_csv(os.path.join(dask_output_dir, 'part-*.csv'), index=False)
    dask_write_time = time.time() - start_time
    results['write_time'].append(dask_write_time)
    print(f"Dask write time: {dask_write_time:.4f}s")

    # benchmark dask reading
    start_time = time.time()
    ddf = dd.read_csv(os.path.join(dask_output_dir, 'part-*.csv'))
    len(ddf.compute())  # force computation
    dask_read_time = time.time() - start_time
    results['read_time'].append(dask_read_time)
    print(f"Dask reading time: {dask_read_time:.4f}s")

    # benchmark polars writing
    start_time = time.time()
    pl_df = pl.from_pandas(test_data)
    pl_df.write_csv('polars_data.csv')
    polars_write_time = time.time() - start_time
    results['write_time'].append(polars_write_time)
    print(f"Polars writing time: {polars_write_time:.4f}s")

    # benchmark polars reading
    start_time = time.time()
    pl.read_csv('polars_data.csv')
    polars_read_time = time.time() - start_time
    results['read_time'].append(polars_read_time)
    print(f"Polars reading time: {polars_read_time:.4f}s")

    return results


def visualize_results(results):
    """Create visualizations of the benchmark results"""
    
    # create x position for plotting
    x = np.arange(len(results['library']))

    # writing performance chart
    plt.figure(figsize=(10, 6))
    plt.bar(x, results['write_time'], width=0.6)
    plt.xlabel('Library')
    plt.ylabel('Time(s)')
    plt.title('Time each library takes to write to csv (Lower is better)')
    plt.xticks(x, results['library'])
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig('csv_writing_performance.png')
    print("\nSaved write performance chart to csv_writing_performance.png")

    # reading performance chart
    plt.figure(figsize=(10, 6))
    plt.bar(x, results['read_time'], width=0.6, color='orange')
    plt.xlabel('Library')
    plt.ylabel('Time(s)')
    plt.title('Time each library takes to read from csv (Lower is better)')
    plt.xticks(x, results['library'])
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig('csv_reading_performance.png')
    print("\nSaved read performance chart to csv_reading_performance.png")

    # create a combined chart for comparison
    plt.figure(figsize=(12, 7))
    width = 0.35

    plt.bar(x - width/2, results['write_time'], width, label='Write Time')
    plt.bar(x - width/2, results['read_time'], width, label='Read Time')

    plt.xlabel('Library')
    plt.ylabel('Time(s)')
    plt.title('CSV Reading and Writing erformance')
    plt.xticks(x, results['library'])
    plt.legend()
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig('csv_benchmark_combined.png')
    print("\nSaved combined chart to csv_benchmark_combined.png")


def save_results(results):
    """Save benchmark results to csv file and display them"""
    
    # save results to a dataframe
    results_df = pd.DataFrame({
        'Library': results['library'],
        'Write Time(s)': results['write_time'],
        'Read Time(s)': results['read_time']
    })
    print("\nBenchmark results:")
    print(results_df)
    results_df.to_csv('benchmark_results.csv', index=False)


def main():
    """Main function to run benchmark test"""

    # number of rows to generate
    num_of_rows = 10000000  # 10 million

    # create test data
    print(f"Generating data with {num_of_rows} rows")
    test_data = generate_test_data(num_of_rows)
    print("Data generation completed")

    # run benchmarks
    results = run_benchmarks(test_data)

    # save results
    visualize_results(results)
    save_results(results)

if __name__ == "__main__":
    main()