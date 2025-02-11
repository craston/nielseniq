# Open Food Facts

## Installation
### Clone the repository
```
git clone https://github.com/craston/nielseniq.git
cd nielseniq
```

### Setting up the environment
#### Option 1: Using the devcontainer in VScode

#### Option 2: Using virtualenv
```
virtualenv .env
source .env/bin/activate
pip install -r requirements.txt
```

## Part One
Assuming the open food facts dataset (`.csv file`) has already been downloaded.
```
cd part_one
```
The `eda.ipynb` notebook contains the results of the EDA.

## Part Two
For this I have used Typer for the entrypoints into the code. Currently, there are two entrypoints:
* for splitting the dataset into train and test; 
```
python part_two/cli.py split run --input-csv path_to_csv_file
```
* for running the training script
```
python part_two/cli.py train run --csv-file path_to_csv_file --image-fold
er path_to_image_folder --config-file path_to_yaml_file
```
## Part Three
The answers are presented in the `part_three/discussion.md` file.