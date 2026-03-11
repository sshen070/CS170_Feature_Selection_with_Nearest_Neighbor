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
                record.append(values[i])

            data.append(record)

    return data

def main():
    data_set = input('Enter Data Set FILENAME: ')
    input_path = Path('../data/') / data_set

    print(f"Using file: {input_path}")

    data_arr = load_data(input_path)
    # print(data_arr)
    # print(len(data_arr))

    forward_selection(data_arr)
    

if __name__ == "__main__":
    main()
