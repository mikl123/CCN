"""
This script includes functions to count/estimate the number of combinations that
violate any constraint (out of 2**N possible combinations)
"""

from typing import List, Tuple
import random

import numpy as np

NUM_CLASSES = 12

STRING_CONSTRAINTS = [
    "A0,A1-A2",
    "A1,A2-A3",
    "A3,!A4-A5",
]

def get_constraint_masks_and_patterns(
    n: int, constraints: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    constraint_masks = np.zeros((len(constraints), n), dtype=int)
    violation_patterns = np.zeros((len(constraints), n), dtype=int)

    for i in range(len(constraints)):
        constraint = constraints[i]
        constraint_mask = np.zeros((1, n))
        violation_pattern = np.zeros((1, n))
        body, head = constraint.split("-")
        body = body.split(",")

        head_ind = int(head.split("A")[1])
        constraint_mask[0, head_ind] = 1

        for cl in body:  # cl = class
            negation, cl_index = cl.split("A")
            cl_index = int(cl_index)

            constraint_mask[0, cl_index] = 1

            if negation == "": # if there's no exclamation mark before "A", then the variable is equal to "" (empty string)
                violation_pattern[0, cl_index] = 1

        constraint_masks[i, :] = constraint_mask
        violation_patterns[i, :] = violation_pattern
    
    return constraint_masks, violation_patterns
    

def bf_approach(n: int, constraints: List[str]) -> int:
    """
    Checks every combination out of possible 2^n and counts the number of
    combinations that violate any constraint
    """
    k = 0

    constraint_masks, violation_patterns = get_constraint_masks_and_patterns(n, constraints)

    for i in range(2**n):
        combination = np.expand_dims(
            np.array(
                list(
                    format(i, f"0{n}b")
                ),
                dtype=int),
            0
        )
        masked = constraint_masks * combination
        
        violates_constraints = (masked == violation_patterns).sum(axis=1) == n
        
        violates_any_constraint = violates_constraints.sum() > 0

        if violates_any_constraint:
            k += 1

    return k

def naive_approach(n: int, constraints: List[str], number_of_trials: int = 100, seed: int = 42) -> int:
    """
    Picks a random combination and checks whether it violates any constraints,
    repeats it number_of_trials times
    """
    k = 0

    constraint_masks, violation_patterns = get_constraint_masks_and_patterns(n, constraints)

    np.random.seed(seed)
    # TODO vectorize it, so the for-loop is not needed
    for i in range(number_of_trials):
        combination = np.random.randint(0, 2, size=(1, n))
        masked = constraint_masks * combination
        
        violates_constraints = (masked == violation_patterns).sum(axis=1) == n
        
        violates_any_constraint = violates_constraints.sum() > 0

        if violates_any_constraint:
            k += 1

    k = int(k / number_of_trials * (2**n))

    return k


def karp_luby_approach(n: int, constraints: List[str], number_of_trials: int = 100, seed: int = 42) -> int:
    """
    Implements the Karp-Luby DNF counting problem (also referred as a case of the union of sets problem).
    See more here: https://www.math.cmu.edu/~af1p/Teaching/MCC17/Papers/KLM.pdf
    """
    constraint_masks, violation_patterns = get_constraint_masks_and_patterns(n, constraints)
    constraint_masks = constraint_masks.astype(bool)

    # Transform the shape from (num_constraints, num_classes) to (1, num_classes, num_constraints)
    constraint_masks_transformed = np.transpose(np.expand_dims(constraint_masks, 1), axes=(1, 2, 0))
    violation_patterns_transformed = np.transpose(np.expand_dims(violation_patterns, 1), axes=(1, 2, 0))


    constraint_cardinalities = np.zeros(shape=(len(constraints)))
    total_cardinality = 0
    for i in range(len(constraints)):
        constraint_size = constraint_masks[i, :].sum()
        constraint_cardinalities[i] = 2**(n - constraint_size)
        total_cardinality += constraint_cardinalities[i]
    constraint_weights =  constraint_cardinalities / total_cardinality

    np.random.seed(seed)

    combinations = np.random.randint(0, 2, size=(number_of_trials, n))
    
    random.seed(seed)
    chosen_constraints = random.choices(
        population=range(len(constraints)),
        weights=constraint_weights,
        k=number_of_trials
    )
    chosen_constraints = np.array(chosen_constraints)

    # This makes the chosen constraints violated in each of the trials. Trust me xD
    combinations[constraint_masks[chosen_constraints, :]] = violation_patterns[chosen_constraints, :][constraint_masks[chosen_constraints, :]]

    combinations = np.expand_dims(combinations, 2)
    combinations_masked = combinations * constraint_masks_transformed

    violates_constraints = (combinations_masked == violation_patterns_transformed).sum(axis=1) == n
    first_violated_constraint = np.argmax(violates_constraints, axis=1)
    
    # Y is a term from the paper
    Y_mean = int(total_cardinality * (first_violated_constraint == chosen_constraints).sum() / number_of_trials)

    return Y_mean


print(bf_approach(NUM_CLASSES, STRING_CONSTRAINTS))
print(naive_approach(NUM_CLASSES, STRING_CONSTRAINTS, number_of_trials=10_000))
print(karp_luby_approach(NUM_CLASSES, STRING_CONSTRAINTS, number_of_trials=10_000))