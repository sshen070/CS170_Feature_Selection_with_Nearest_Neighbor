import matplotlib.pyplot as plt
import random as rand
import math

def forward_selection(data_arr):

    # First column is the cluster record belongs to
    num_features = len(data_arr[0]) - 1

    # Keep track of the features that we plan to include in our final selection
    current_set = set()

    # Keep track of global variables for optimal subset & accuracy across all features
    best_overall_accuracy = 0 
    best_set = set()

    # Track the selected set over all iterations & corresponding accuracies    
    selected_sets = []
    selected_accuracy = []

    print(f'\nThis dataset has a total of {num_features} features, with {len(data_arr)} instances.')
    
    # Test with all features for references
    full_set = set()
    for i in range(1, num_features + 1):
        full_set.add(i)
    print(f'Running nearest neighbor with all {num_features} features, using “leaving-one-out” evaluation, I get an accuracy of {leave_one_out_cross_validation(data_arr, full_set, 0)}')
    print('Beginning Forward Selection\n')


    # Determine empty set stats before Forward Selection
    empty_set_accuracy = leave_one_out_cross_validation(data_arr, current_set, 0)
    print(f'\tUsing feature(s) {current_set} accuracy is {empty_set_accuracy}\n')

    selected_sets.append(current_set.copy())
    selected_accuracy.append(empty_set_accuracy)

    best_overall_accuracy = empty_set_accuracy

    # Reduce computational time for large dataset
    half_way = int(num_features/2)

    # N iterations (1 per level of parsing)
    for i in range(num_features):

        feature_to_add_at_this_level = None

        # Keep track of the best potential features based current iteration (non-global)
        best_accuracy_curr_level = 0

        for k in range(1, num_features + 1):
            
            # Do not repeat features
            if k not in current_set:

                potential_set = current_set.copy()
                potential_set.add(k)
                
                accuracy = leave_one_out_cross_validation(
                    data_arr, potential_set, best_accuracy_curr_level
                )

                if (accuracy > best_accuracy_curr_level):
                    best_accuracy_curr_level = accuracy
                    feature_to_add_at_this_level = k
            
                print(f'\tUsing features(s) {potential_set} accuracy is {accuracy}')
        
        current_set.add(feature_to_add_at_this_level)

        # Append best feature set & accuracy at the level --> Return for analysis
        selected_sets.append(current_set.copy())
        selected_accuracy.append(best_accuracy_curr_level)

        if best_accuracy_curr_level < best_overall_accuracy:
            print('\n(Warning, Accuracy has decreased! Continuing search in case of local maxima)')

        else:
            
            # Keep track of optimal set of features & accuracy
            best_set = current_set.copy()
            best_overall_accuracy = best_accuracy_curr_level
            print('\n')

        print(f'Feature set {current_set} was best, accuracy is {best_accuracy_curr_level}\n')

    print(f'Finished Forward Selection! The best feature subset is {best_set}, which has an accuracy of {best_overall_accuracy}')

    return selected_sets, selected_accuracy


def backward_elimination(data_arr):

    # First column is the cluster record belongs to
    num_features = len(data_arr[0]) - 1

    # Start with full set & move backwards
    current_set = set()
    for i in range(1, num_features + 1):
        current_set.add(i)

    # Keep track of global variables for optimal subset & accuracy across all features
    best_overall_accuracy = 0 
    best_set = current_set.copy()

    # Track the selected set over all iterations & corresponding accuracies    
    selected_sets = []
    selected_accuracy = []

    print(f'\nThis dataset has a total of {num_features} features, with {len(data_arr)} instances.')
    
    # Test with all features for references
    print(f'Running nearest neighbor with all {num_features} features, using “leaving-one-out” evaluation, I get an accuracy of {leave_one_out_cross_validation(data_arr, current_set, 0)}')
    print('Beginning Backward Elimination\n')

    # Reduce computational time for large dataset
    quarter_way = 3 * int(num_features/4)

    # Continue to run until only one feature remains (no more feature elimination can be performed)
    while(len(current_set) > 1):

        feature_to_remove_at_this_level = None

        # Keep track of the best potential features based current iteration (non-global)
        best_accuracy_curr_level = 0

        for k in current_set:
            potential_set = current_set.copy()
            potential_set.remove(k)
            
            accuracy = leave_one_out_cross_validation(
                data_arr, potential_set, best_accuracy_curr_level
            )

            if (accuracy > best_accuracy_curr_level):
                best_accuracy_curr_level = accuracy
                feature_to_remove_at_this_level = k
        
            print(f'\tUsing features(s) {potential_set} accuracy is {accuracy}')
        
        current_set.remove(feature_to_remove_at_this_level)

        # Append best feature set & accuracy at the level --> Return for analysis
        selected_sets.append(current_set.copy())
        selected_accuracy.append(best_accuracy_curr_level)


        if best_accuracy_curr_level < best_overall_accuracy:
            print('\n(Warning, Accuracy has decreased! Continuing search in case of local maxima)')

        else:
            
            # Keep track of optimal set of features & accuracy
            best_set = current_set.copy()
            best_overall_accuracy = best_accuracy_curr_level
            print('\n')

        print(f'Feature set {current_set} was best, accuracy is {best_accuracy_curr_level}\n')


    # Determine empty set stats at the end of Backward Elimination
    empty_set = set()
    empty_set_accuracy = leave_one_out_cross_validation(data_arr, empty_set, 0)
    print(f'\tUsing feature(s) {empty_set} accuracy is {empty_set_accuracy}\n')

    selected_sets.append(empty_set)
    selected_accuracy.append(empty_set_accuracy)

    if (empty_set_accuracy > best_overall_accuracy):
        best_set = empty_set
        best_overall_accuracy = empty_set_accuracy

    print(f'Finished Backward Elimination! The best feature subset is {best_set}, which has an accuracy of {best_overall_accuracy}')

    return selected_sets, selected_accuracy


def leave_one_out_cross_validation(data_arr, potential_set, best_accuracy_curr_level):
    
    # If potential_set is null (cross validation edge case)
    if (not potential_set):
        
        # Keep track of each of the different labels and occurances with a dict
        cluster_counter = {}

        # Record every label in data set
        for i in range(len(data_arr)):
        
            object_label = data_arr[i][0]
            if (object_label in cluster_counter):
                cluster_counter[object_label] += 1

            else:
                cluster_counter[object_label] = 1

        largest_count = 0

        for key in cluster_counter:
            if (cluster_counter[key] > largest_count):
                largest_count = cluster_counter[key]

        return largest_count/len(data_arr)

    # Early abandoning optimization
    number_correctly_classified = 0
    tot_mistakes = 0
    max_allowed_mistakes = int((1 - best_accuracy_curr_level) * len(data_arr))

    # Iterate for all for the whole length of the dataset (for each object)
    for i in range (len(data_arr)):
        
        object_label = data_arr[i][0]

        # Track closest neigbor & label
        nearest_neighbor_distance = float('inf')
        nearest_neighbor_label = None
        
        # For each object find the nearest neighbor (compare with every other object)
        for k in range (len(data_arr)):
            
            # Comparing an object with itself will return 0 (we must find the closest neighbor)
            if (k != i):
                
                # Calculate the Euclidian Distance from point i to every other point k (only based on features in the potential_set)
                total_diff = 0
                for feature in potential_set:
                    total_diff += pow(data_arr[i][feature] - data_arr[k][feature], 2)
                
                total_dist = math.sqrt(total_diff)

                # Is the distance from the previously calculated neighbor closer?
                if (total_dist < nearest_neighbor_distance):
                    nearest_neighbor_distance = total_dist
                    nearest_neighbor_label = data_arr[k][0]

        if (nearest_neighbor_label == object_label):
            number_correctly_classified += 1
        else:
            tot_mistakes += 1

        if (tot_mistakes > max_allowed_mistakes):
            return -1
        
    # Find the avg amount of correctly classified items for all elements between the n features
    num_items = len(data_arr)
    return number_correctly_classified/num_items


def visualize_feature_selection(selected_sets, selected_accuracy):
    
    # Convert sets to sorted strings for labeling
    x_labels = [str(sorted(s)) for s in selected_sets]
    tot_sets = int(len(selected_accuracy))

    accuracy_percent = []
    for i in range(tot_sets):
        accuracy_percent.append(selected_accuracy[i] * 100)
        
    plt.figure(figsize=(12,6))
    
    # Bar graph
    bars = plt.bar(range(tot_sets), accuracy_percent, color='skyblue')
    
    # Label x-axis with selected feature sets
    plt.xticks(ticks=range(tot_sets), labels=x_labels, rotation=45, ha='right')
    
    plt.xlabel("Selected Feature Sets")
    plt.ylabel("Accuracy (%)")
    plt.title("Selection Algorithm Performance")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.ylim(0,100)

    # Label percentages on each bar & expand visualization for x-ticks sets to be seen
    plt.bar_label(bars, fmt="%.2f%%", padding=3)
    plt.tight_layout()

    plt.show()