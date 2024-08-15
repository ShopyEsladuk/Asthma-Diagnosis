import numpy as np
import pandas as pd
import asthma_script

def test_load_data_from_github():
    """
    Tests load_data_from_github function from asthma_script
    """
    test_data = asthma_script.load_data_from_github("Wrong link")
    assert test_data is None

def test_split_words():
    """
    Tests split_words function from asthma_script
    """
    some_words = ["DataScience", "datascience", "GitHubRepo"]
    target = ["Data_Science", "datascience", "Git_Hub_Repo"]
    result = asthma_script.split_words(some_words)
    assert result == target

def test_data_cleaning():
    """
    Tests data_cleaning function from asthma_script
    """
    test_data = pd.read_csv("https://raw.githubusercontent.com/ShopyEsladuk/Asthma-Diagnosis/main/data/asthma_disease_data.csv")
    target_data = pd.read_csv("https://raw.githubusercontent.com/ShopyEsladuk/Asthma-Diagnosis/main/data/asthma_data_preprocessed.csv")
    target_data = target_data.drop(columns = ["gender", "patient_id"])
    result = asthma_script.data_cleaning(test_data)
    
    assert np.array_equal(target_data.columns, result.columns)

def test_scale_data():
    """
    Tests scale_data function from asthma_script
    """
    test_data = pd.DataFrame({
    'F1': [20.0, 5, 0],
    'F2': [12.0, 12, 0],
    'F3': [0, 0.5, 1]
    })
    target_data = pd.DataFrame({
    'F1': [1, 0.25, 0],
    'F2': [1.0, 1, 0],
    'F3': [0, 0.5, 1]
    })

    result = asthma_script.scale_data(test_data)
    assert result.equals(target_data)

def test_extract_mean():
    """
    Tests extract_mean function from asthma_script
    """
    test_data = pd.DataFrame({
    'F1': [1, 1, 0, 0, 1, 0],
    'F2': [1, 1, 1, 1, 1, 1],
    'F3': [1, 1, 1, 0, 0, 0],
    'diagnosis': [1, 1, 1, 1, 0, 0]
    })
    target_data = pd.DataFrame({
    "mean_positive": [0.5, 1, 0.75],
    "mean_negative": [0.5, 1, 0],
    "difference": [0, 0, 0.75]
    }, 
    index = ["F1", "F2", "F3"]
    )
    target_data = target_data.sort_values("mean_positive", ascending = False)
    result = asthma_script.extract_means(test_data)
    
    assert result.equals(target_data)

def main():
    test_load_data_from_github()
    print("load_data_from_github passed unit test")
    test_split_words()
    print("split_words passed unit test")
    test_data_cleaning()
    print("data_cleaning passed unit test")
    test_scale_data()
    print("scale_data passed unit test")
    test_extract_mean()
    print("extract_mean passed unit test")
    print("everything passed unit tests")

if __name__ == "__main__":
    main()