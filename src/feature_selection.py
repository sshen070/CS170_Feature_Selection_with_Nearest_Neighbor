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

    print('Beginning Forward Selection\n')
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

        if best_accuracy_curr_level < best_overall_accuracy:
            print('\n(Warning, Accuracy has decreased! Continuing search in case of local maxima)')

        else:
            
            # Keep track of optimal set of features & accuracy
            best_set = current_set.copy()
            best_overall_accuracy = best_accuracy_curr_level
            print('\n')

        print(f'Feature set {current_set} was best, accuracy is {best_accuracy_curr_level}\n')

    print(f'Finished Forward Selection! The best feature subset is {best_set}, which has an accuracy of {best_overall_accuracy}')


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

    print('Beginning Backward Elimination\n')

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

        if best_accuracy_curr_level < best_overall_accuracy:
            print('\n(Warning, Accuracy has decreased! Continuing search in case of local maxima)')

        else:
            
            # Keep track of optimal set of features & accuracy
            best_set = current_set.copy()
            best_overall_accuracy = best_accuracy_curr_level
            print('\n')

        print(f'Feature set {current_set} was best, accuracy is {best_accuracy_curr_level}\n')

    print(f'Finished Backward Elimination! The best feature subset is {best_set}, which has an accuracy of {best_overall_accuracy}')



def leave_one_out_cross_validation(data_arr, potential_set, best_accuracy_curr_level):
    
    number_correctly_classified = 0

    # Early abandoning optimization
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