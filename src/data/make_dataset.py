# make_dataset.py
import pathlib
import yaml
import sys
import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(data_path):
    # Load your dataset from a given path
    df = pd.read_csv(data_path)
    return df

def split_data(df, test_split, seed):
    # Split the dataset into train and test sets
    train, test = train_test_split(df, test_size=test_split, random_state=seed)
    return train, test

def save_data(train, test, output_path):
    # Save the split datasets to the specified output path
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    train.to_csv(output_path + '/train.csv', index=False)
    test.to_csv(output_path + '/test.csv', index=False)

def main():
    
    # below code is their to navigate into folder structure.
    # /src/data
    curr_dir = pathlib.Path(__file__) 
    # /src/NYC_TAXI
    home_dir = curr_dir.parent.parent.parent 
    # parameter file present in home directory with the name of '/params.yaml'
    params_file = home_dir.as_posix() + '/params.yaml'
    # reading params.yaml file and pick up the parameter which is relavent to make_dataset.py file
    params = yaml.safe_load(open(params_file))["make_dataset"]
    
    # The value .\data\raw\nyc_taxi.xls is assigned to sys.argv[1] in your Python script 
    # (make_dataset.py)  because you explicitly passed it as a command-line argument when 
    # executing the Python script from the command line.
    input_file = sys.argv[1]
    # data present in home directory and input_file is path of raw data
    data_path = home_dir.as_posix() + input_file
    # this is the path where we save train, test processed data.
    output_path = home_dir.as_posix() + '/data/processed' 
    
    # loading data
    data = load_data(data_path)
    train_data, test_data = split_data(data, params['test_split'], params['seed'])
    # saving train test split into output_path i.e., /data/processed.
    save_data(train_data, test_data, output_path)

if __name__ == "__main__":
    main()