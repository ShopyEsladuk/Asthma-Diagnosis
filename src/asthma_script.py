import numpy as np
import pandas as pd
import re
from sklearn.preprocessing import MinMaxScaler

# Data Loading

def load_data_from_github(link):
    """
    Reads asthma dataset given its online link
    :param link: link to the DataSet
    :return: the loaded DataSet
    """
    try:
        asthma_data = pd.read_csv(link)
        return asthma_data
    except Exception as e:
        print(f"Error Loading data {e}")
        return None

# Data Preprocessing

def split_words(columns):
    """
    Splits the words in the names of columns in a DataFrame
    :param columns: names of columns in DataFrame
    :return: list of splitted by '_' words in columns names
    """
    try:
        splitted_words = []
        for column in columns:
            if column[0].islower():
                splitted_words.append(column)
            else:
                words = "_".join(re.findall(r'[A-Z][^A-Z]*', column))
                splitted_words.append(words)
        return splitted_words
    except Exception as e:
        print(f"Error splitting words: {e}")
        return columns

def data_cleaning(asthma_data):
    """
    Cleans the asthma DataSet
    :param asthma_data: DataSet containing information about asthma patients
    :return: the cleaned DataSet
    """
    try:
        # No doctor's name mentioned because it contains confidential info
        asthma_data = asthma_data.drop(columns=["DoctorInCharge"])
    except:
        print("No 'DoctorInCharge' column found")

    asthma_data.columns = split_words(asthma_data.columns)
    asthma_data.columns = asthma_data.columns.str.lower()
    try:
        asthma_data = asthma_data.rename(
            columns={
                "patient_i_d": "patient_id",
                'b_m_i': 'bmi',
                'lung_function_f_e_v1': 'lung_function_fev1',
                'lung_function_f_v_c': 'lung_function_fvc',
            }
        )
    except Exception as e:
        print(f"Error connecting abbreviations: {e}")

    # Absolutely no dependence found on gender in asthma diagnosis
    try:
        asthma_data = asthma_data.drop(columns=["gender"])
    except:
        print("No 'gender' (after lowercasing) column found")

    try:
        asthma_data = asthma_data.drop(columns=["patient_id"])
    except:
        print("No 'patiend_id' (after splitting and lowercasing words) column found")

    return asthma_data

# Finding Dependencies in Asthma Diagnosis

def scale_data(asthma_data):
    """
    Scales values in a given DataFrame using MinMaxScaler with feature range [0, 1]
    :param asthma_data: DataFrame containing information about asthma patients
    :return: DataFrame with scaled values
    """
    try:
        scaler = MinMaxScaler(feature_range = (0, 1))
        asthma_data_scaled = scaler.fit_transform(asthma_data)
        asthma_data_scaled = pd.DataFrame(asthma_data_scaled, columns = asthma_data.columns)
        return asthma_data_scaled
    except Exception as e:
        print(f"Error scaling values: {e}")
        return None

def extract_means(asthma_data_scaled):
    """
    Splits DataFrame into positive and negative tested patients. Gets the mean of each feature and compares them.

    *Getting the mean of column with boolean values (0 and 1) gives us the ratio of people with 1 value
    :param asthma_data_scaled: scaled DataFrame with asthma patients information
    :return: DataFrame with comparison of means of values between DataFrames with positive or negative diagnosis
    """
    try:
        # Positive for asthma
        positive_diagnosis_means = asthma_data_scaled[asthma_data_scaled.diagnosis == 1].describe().T[["mean"]]

        # Negative for asthma
        negative_diagnosis_means = asthma_data_scaled[asthma_data_scaled.diagnosis == 0].describe().T[["mean"]]

        comparison_means = positive_diagnosis_means.rename(columns = {"mean" : "mean_positive"})
        comparison_means["mean_negative"] = negative_diagnosis_means["mean"]
        comparison_means["difference"] = comparison_means.mean_positive.values - comparison_means.mean_negative.values

        comparison_means = comparison_means.drop(index = "diagnosis")
        comparison_means = comparison_means.sort_values("mean_positive", ascending = False)

        return comparison_means
    except KeyError as e:
        print(f"Key error: {e}")
    except Exception as e:
        print(f"Error extracting means: {e}")
        return None

def main():
    url = "https://raw.githubusercontent.com/ShopyEsladuk/Asthma-Diagnosis/main/data/asthma_disease_data.csv"
    asthma_data = load_data_from_github(url)
    if asthma_data is not None:
        processed_data = data_cleaning(asthma_data)
        scaled_data = scale_data(processed_data)
        comparison_means = extract_means(scaled_data)
        print(comparison_means)

if __name__ == "__main__":
    main()