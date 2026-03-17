from pathlib import Path
import time
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

    user_input = int(input('Enter 1 to choose Small Dataset: 2 for Large Dataset: '))
    if (user_input == 1):
        load_path = input_path / 'CS170_Small_DataSet__52.txt'
        data_arr = load_data(load_path)
        print(f"Using file: {load_path}\n")

        user_input = int(input('Enter 1 for Forward Selection; 2 for Backwards Elimination: '))
        
        if (user_input == 1):
            # Forward Selection Timing
            start_time = time.time()
            selected_sets, selected_accuracy = forward_selection(data_arr)
            forward_duration = time.time() - start_time

            visualize_feature_selection(selected_sets, selected_accuracy)
            print(f"Forward Selection completed in {forward_duration:.2f} seconds. Best Accuracy: {max(selected_accuracy)}\n")

        elif (user_input == 2):
            # Backward Elimination Timing 
            start_time = time.time()
            selected_sets, selected_accuracy = backward_elimination(data_arr)
            backward_duration = time.time() - start_time

            visualize_feature_selection(selected_sets, selected_accuracy)
            print(f"Backward Elimination completed in {backward_duration:.2f} seconds. Best Accuracy: {max(selected_accuracy)}\n")
        
    elif (user_input == 2):
        load_path = input_path / 'CS170_Large_DataSet__38.txt'
        data_arr = load_data(load_path)
        print(f"Using file: {load_path}")
        
        user_input = int(input('Enter 1 for Forward Selection; 2 for Backwards Elimination: '))

        if (user_input == 1):
            # Forward Selection Timing
            start_time = time.time()
            selected_sets, selected_accuracy = forward_selection(data_arr)
            forward_duration = time.time() - start_time

            visualize_feature_selection(selected_sets, selected_accuracy)
            print(f"Forward Selection completed in {forward_duration:.2f} seconds. Best Accuracy: {max(selected_accuracy)}\n")

        elif (user_input == 2):
            # Backward Elimination Timing 
            start_time = time.time()
            selected_sets, selected_accuracy = backward_elimination(data_arr)
            backward_duration = time.time() - start_time

            visualize_feature_selection(selected_sets, selected_accuracy)
            print(f"Backward Elimination completed in {backward_duration:.2f} seconds. Best Accuracy: {max(selected_accuracy)}\n")

if __name__ == "__main__":
    main()
