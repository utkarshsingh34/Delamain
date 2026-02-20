"""Clean and filter NYC TLC trip data to Manhattan zones.

Reads raw parquet files, filters to Manhattan, computes derived columns,
removes outliers, and outputs processed parquet files.
"""
