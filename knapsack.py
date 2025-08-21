import numpy as np

def knapsack_dp(values, weights, num_items, max_capacity):
    dp_table = np.zeros((num_items + 1, max_capacity + 1), dtype=np.float32)
    decision_table = np.zeros((num_items + 1, max_capacity + 1), dtype=np.int8)

    for item_idx in range(1, num_items + 1):
        item_weight = weights[item_idx - 1]
        item_value = values[item_idx - 1]
        for capacity in range(max_capacity + 1):
            if item_weight <= capacity:
                value_including = item_value + dp_table[item_idx - 1, capacity - item_weight]
                value_excluding = dp_table[item_idx - 1, capacity]
                if value_including > value_excluding:
                    dp_table[item_idx, capacity] = value_including
                    decision_table[item_idx, capacity] = 1
                else:
                    dp_table[item_idx, capacity] = value_excluding
            else:
                dp_table[item_idx, capacity] = dp_table[item_idx - 1, capacity]

    selected_items = []
    remaining_capacity = max_capacity

    for item_idx in range(num_items, 0, -1):
        if decision_table[item_idx, remaining_capacity] == 1:
            selected_items.append(item_idx - 1)
            remaining_capacity -= weights[item_idx - 1]

    selected_items.sort()
    return selected_items
