import random as rand

# Implement forward selection & backward elimination search algorithm
def forward_selection(data_arr):

    # First column is the cluster record belongs to
    num_features = len(data_arr[0]) - 1

    # Keep track of the features that we plan to include in our final selection
    current_set = set()

    # Keep track of global variables for optimal subset & accuracy across all features
    best_overall_accuracy = 0 
    best_set = set()

    print('Beginning Search\n')
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
                    data_arr, current_set, k
                )

                if (accuracy > best_accuracy_curr_level):
                    best_accuracy_curr_level = accuracy
                    feature_to_add_at_this_level = k
            
                print(f'\tUsing features(s) {potential_set} accuracy is {accuracy}')
        
        current_set.add(feature_to_add_at_this_level)

        if best_accuracy_curr_level < best_overall_accuracy:
            print('\n(Warning, Accuracy has decreased! Continuing search in case of local maxima)')
            
        else:
            
            # Keep track of optimal set & accuracy
            best_set = current_set.copy()
            best_overall_accuracy = best_accuracy_curr_level
            print('\n')

        print(f'Feature set {current_set} was best, accuracy is {best_accuracy_curr_level}\n')

    print(f'Finished search! The best feature subset is {best_set}, which has an accuracy of {best_overall_accuracy}')


# Unimplemented ~ Testing search
def leave_one_out_cross_validation(data_arr, current_set, k):
    return rand.random()