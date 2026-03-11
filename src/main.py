from pathlib import Path
from feature_selection import*

# Convert CSV data into array format for easy access during Feature Selection
def load_data(input_path):
    # Stores every record of CSV
    data = []

    with open(input_path, "r") as file:
        for line in file:
            record = []
            values = line.strip().split()

            for i in range(len(values)):
                record.append(float(values[i]))

            data.append(record)

    return data

# Datsets: Small(52) & Large(38)
def main():
    # data_set = input('Enter Data Set FILENAME: ')
    # input_path = Path('../data/') / data_set
    input_path = Path('../data/personalized_datasets/')

    if (int(input('Enter 1 to choose Small Dataset: ')) == 1):
        load_path = input_path / 'CS170_Small_DataSet__52.txt'
        data_arr = load_data(load_path)
        print(f"Using file: {load_path}")

        forward_selection(data_arr)

    else:
        load_path = input_path / 'CS170_Large_DataSet__38.txt'
        data_arr = load_data(load_path)
        print(f"Using file: {load_path}")

        forward_selection(data_arr)


if __name__ == "__main__":
    main()
