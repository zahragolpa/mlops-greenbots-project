import pandas as pd
import os
import datetime

from prefect import task,flow


class PreprocessData:
    def __init__(self):
        pass
    @task
    def load_data(path='', separator=';'):
        """
        Load data from a file.

        Parameters:
        - path: File path.
        - separator: Delimiter used in the file.

        Returns:
        - df: DataFrame containing the loaded data.
        """
        _, ext = os.path.splitext(path)
        ext = ext.lower()[1:]
        if ext == 'csv':
            return pd.read_csv(path, sep=separator)
        elif ext in ['xlsx', 'xls']:
            return pd.read_excel(path, sep=separator)
        else:
            raise ValueError("Unsupported file extension. Only 'csv', 'xlsx', and 'xls' are supported.")

    def parse_info(df=None):
        """
        Parse information about the DataFrame.

        Parameters:
        - df: DataFrame.

        Returns:
        - info_dict: Dictionary containing information about each column.
        """
        info_dict = {}
        for column_name, dtype in df.dtypes.items():
            non_null_count = df[column_name].notnull().sum()
            info_dict[column_name] = {'nonNullCount': non_null_count, 'Dtype': str(dtype)}
        return info_dict
    
    @task
    def remove_outliers(df=None, columns_list=None, quantiles_list=None):
        """
        Remove outliers from the DataFrame.

        Parameters:
        - df: DataFrame.
        - columns_list: List of columns to consider.
        - quantiles_list: List of quantiles for outlier detection.

        Returns:
        - df: DataFrame after removing outliers.
        """
        for column, quantile in zip(columns_list, quantiles_list):
            low_quantile = df[column].quantile(quantile)
            high_quantile = df[column].quantile(1 - quantile)
            df = df[(df[column] >= low_quantile) & (df[column] <= high_quantile)]
        return df

    @task
    def get_statistics(df=None):
        """
        Get descriptive statistics for the DataFrame.

        Parameters:
        - df: DataFrame.

        Returns:
        - result_dict: Dictionary containing statistics for each column.
        """
        result_dict = df.describe().to_dict()
        return result_dict
    
    @task
    def bin_age(df=None, age_edges_list=None, age_labels_list=None, is_day=True):
        """
        Bin the age column in the DataFrame.

        Parameters:
        - df: DataFrame.
        - age_edges_list: List of bin edges for age.
        - age_labels_list: List of labels for age bins.
        - is_day: Boolean indicating whether age should be converted to years.

        Returns:
        - df: DataFrame with age bin column added.
        """
        if is_day:
            df['age'] = (df['age'] / 365).round().astype('int')
        df['age_group'] = pd.cut(df['age'], bins=age_edges_list, labels=age_labels_list, include_lowest=True,
                                  right=True)
        return df
    
    @task
    def calculate_bmi(df=None, weight_col='weight', height_col='height'):
        """
        Calculate BMI and bin it in the DataFrame.

        Parameters:
        - df: DataFrame.
        - weight_col: Column name for weight.
        - height_col: Column name for height.

        Returns:
        - df: DataFrame with BMI column added and binned.
        """
        df['bmi'] = df[weight_col] / ((df[height_col] / 100) ** 2)
        df['bmi'] = pd.cut(df['bmi'], bins=6, labels=range(6), right=True, include_lowest=True)
        return df

    @task
    def calculate_map(df=None, ap_lo_col='ap_lo', ap_hi_col='ap_hi'):
        """
        Calculate Mean Arterial Pressure (MAP) and bin it in the DataFrame.

        Parameters:
        - df: DataFrame.
        - ap_lo_col: Column name for diastolic blood pressure.
        - ap_hi_col: Column name for systolic blood pressure.

        Returns:
        - df: DataFrame with MAP column added and binned.
        """
        df['map'] = ((2 * df[ap_lo_col]) + df[ap_hi_col]) / 3
        df['map'] = pd.cut(df['map'], bins=6, labels=range(6), right=True, include_lowest=True)
        return df
    
    @task
    def drop_columns(df=None, columns_to_drop=None):
        """
        Drop columns from the DataFrame.

        Parameters:
        - df: DataFrame.
        - columns_to_drop: List of column names to drop.

        Returns:
        - df: DataFrame after dropping specified columns.
        """
        try:
            df = df.drop(columns_to_drop, axis=1)
            print("Columns dropped successfully.")
        except KeyError as e:
            print(f"Error: One or more columns to drop do not exist in the DataFrame: {e}")
        return df
    
    @task
    def save_data(df=None, path=''):
        """
        Save the DataFrame to a file.

        Parameters:
        - df: DataFrame.
        - path: File path to save the DataFrame.
        """
        df.to_csv(path, index=False)

    @flow(name='Process file')
    def process_file(origin_path='', destination_dir=''):
        """
        Perform data preprocessing and save the processed DataFrame to a file.

        Parameters:
        - origin_path: Path to the original data file.
        - destination_path: Path to save the processed data file.
        """
        self = PreprocessData
        data_df = self.load_data(origin_path)
        print("Data loaded successfully.")
        print("Data information:")
        print(self.parse_info(data_df))
        data_df = self.remove_outliers(df=data_df, columns_list=['height', 'weight', 'ap_hi', 'ap_lo'],
                                       quantiles_list=[0.025, 0.025, 0.025, 0.025])
        print("Outliers removed.")
        print("Descriptive statistics after outlier removal:")
        print(self.get_statistics(data_df))
        data_df = self.bin_age(df=data_df, age_edges_list=[30, 35, 40, 45, 50, 55, 60, 65],
                               age_labels_list=[0, 1, 2, 3, 4, 5, 6])
        print("Age binned.")
        data_df = self.calculate_bmi(df=data_df)
        print("BMI calculated and binned.")
        data_df = self.calculate_map(df=data_df)
        print("MAP calculated and binned.")
        data_df = self.drop_columns(df=data_df, columns_to_drop=['height', 'weight', 'ap_hi', 'ap_lo', 'age'])
        print("Columns dropped.")
         # Generate filename with current date and time
        current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        destination_path = os.path.join(destination_dir, f"clean_data_{current_datetime}.csv")

        self.save_data(df=data_df, path=destination_path)
        print("Data saved successfully.")


if __name__ == '__main__':
    preprocessor = PreprocessData()
    preprocessor.process_file(origin_path='data/raw/cardio_train.csv',
                              destination_dir='data/processed/')
