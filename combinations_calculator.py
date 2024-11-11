"""
The matrices produced in this module are described in section 4.1 Constraint Module of the paper.
"""

import numpy as np
import itertools
import resource
import signal
import concurrent.futures
from collections import defaultdict
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix

######
# Note: Since in practice all the datasets had only one stratum, we implemented the code for one stratum
#####


def createIs(file_path, num_classes):
    """
    Generates two matrices, Iplus and Iminus, which correspond to the presence or absence of literals in the constraint bodies.
    """

    # Matrix with indices for positive literals
    Iplus = []
    # Matrix with indeces for negative literals
    Iminus = []
    with open(file_path, 'r') as f:
        for line in f:
            split_line = line.split()
            assert split_line[2] == ':-'
            iplus = np.zeros(num_classes)
            iminus = np.zeros(num_classes)
            for item in split_line[3:]:
                if 'n' in item:
                    index = int(item[1:])
                    iminus[index] = 1
                else:
                    index = int(item)
                    iplus[index] = 1
            Iplus.append(iplus)
            Iminus.append(iminus)
    Iplus = np.array(Iplus)
    Iminus = np.array(Iminus)
    return Iplus, Iminus


def createM(file_path, num_classes):
    """
    Matrix M maps the head of each constraint to a binary representation.
    Each row corresponds to a specific constraint with its associated target class.
    """
    M = []
    with open(file_path, 'r') as f:
        for line in f:
            split_line = line.split()
            assert split_line[2] == ':-'
            m = np.zeros(num_classes)
            index = int(split_line[1])
            m[index] = 1
            M.append(m)
    M = np.array(M).transpose()
    return M


def find_connected_components(Iplus, Iminus, M, num_classes):
    # Create an adjacency matrix for the graph based on constraints
    adjacency_matrix = np.zeros((num_classes, num_classes))
    
    for row in range(len(Iplus)):
        involved_classes = np.where((Iplus[row] + Iminus[row] + M[row]) > 0)[0]
        for i in involved_classes:
            for j in involved_classes:
                if i != j:
                    adjacency_matrix[i, j] = 1
                    adjacency_matrix[j, i] = 1
    
    # Use connected components to find groups of interconnected classes
    graph = csr_matrix(adjacency_matrix)
    n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
    
    components = defaultdict(list)
    for i, component_id in enumerate(labels):
        components[component_id].append(i)
    
    return list(components.values())

def generate_combinations_for_component(component, Iplus, Iminus, M):
    num_classes = len(component)
    all_combinations = list(itertools.product([0, 1], repeat=num_classes))
    valid_combinations = []
    
    for combination in all_combinations:
        combination = np.array(combination)
        valid = True
        
        for i, (iplus, iminus) in enumerate(zip(Iplus, Iminus)):
            if not np.any(iplus[component]) and not np.any(iminus[component]):
                continue
            
            constr = combination - M[i][component]
            if np.all(constr[iplus[component] == 1] == 1) and np.all(combination[iminus[component] == 1] == 0):
                if not np.all(combination[M[i][component] == 1] == 1):
                    valid = False
                    break
        if valid:
            valid_combinations.append(combination)
    
    return np.array(valid_combinations)

def generate_combinations(file_path, num_classes):
    Iplus, Iminus = createIs(file_path, num_classes)
    M = createM(file_path, num_classes).transpose()
    
    components = find_connected_components(Iplus, Iminus, M, num_classes)
    
    component_combinations = []
    for component in components:
        valid_combinations = generate_combinations_for_component(component, Iplus, Iminus, M)
        component_combinations.append(valid_combinations)
    
    print("Component combinations: ", len(component_combinations))
    print("Together combinations: ", sum(len(x) for x in component_combinations))
    return sum(len(x) for x in component_combinations)
    
    # Combine combinations across all components
    # final_combinations = np.array(list(itertools.product(*component_combinations)))
    # return final_combinations.reshape(-1, num_classes)

def set_resource_limits(cpu_time_limit, memory_limit):
    resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))

def handle_timeout(signum, frame):
    raise TimeoutError("Execution time exceeded the limit")

def process_dataset(dataset):
    try:
        print(f"Processing dataset: {dataset['file_path']}")
        valid_combinations = generate_combinations(dataset["file_path"], dataset["num_classes"])
        print(f"Num of valid combinations for {dataset['file_path']}: {valid_combinations}")
        return valid_combinations
    except TimeoutError as e:
        print(f"TimeoutError for {dataset['file_path']}: {e}")
    except MemoryError as e:
        print(f"MemoryError for {dataset['file_path']}: {e}")
    except Exception as e:
        print(f"An error occurred for {dataset['file_path']}: {e}")
        
        
if __name__ == "__main__":
    datasets = [
        {"file_path": "./data/arts/arts_constraints.txt", "num_classes": 26},
        {"file_path": "./data/business/business_constraints.txt", "num_classes": 30},
        {"file_path": "./data/cal500/cal500_constraints.txt", "num_classes": 174}, 
        {"file_path": "./data/emotions/emotions_constraints.txt", "num_classes": 6},
        {"file_path": "./data/enron/enron_constraints.txt", "num_classes": 53},
        {"file_path": "./data/genbase/genbase_constraints.txt", "num_classes": 27},
        {"file_path": "./data/image/image_constraints.txt", "num_classes": 5}, 
        {"file_path": "./data/medical/medical_constraints.txt", "num_classes": 45},
        {"file_path": "./data/rcv1subset1/rcv1subset1_constraints.txt", "num_classes": 101},  
        {"file_path": "./data/rcv1subset2/rcv1subset2_constraints.txt", "num_classes": 101}, 
        {"file_path": "./data/rcv1subset3/rcv1subset3_constraints.txt", "num_classes": 101}, 
        {"file_path": "./data/rcv1subset4/rcv1subset4_constraints.txt", "num_classes": 101},
        {"file_path": "./data/rcv1subset5/rcv1subset5_constraints.txt", "num_classes": 101},
        {"file_path": "./data/scene/scene_constraints.txt", "num_classes": 6},
        {"file_path": "./data/science/science_constraints.txt", "num_classes": 40},
        {"file_path": "./data/yeast/yeast_constraints.txt", "num_classes": 14},
    ]
    
    np.set_printoptions(threshold=np.inf)
    
    # Set resource limits
    cpu_time_limit = 120 
    memory_limit = 10 * 1024 * 1024 * 1024  # 5 GB
    
    set_resource_limits(cpu_time_limit, memory_limit)
    
    # Set signal handler for timeout
    signal.signal(signal.SIGXCPU, handle_timeout)
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_dataset, dataset) for dataset in datasets]
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                print(result)
            except Exception as e:
                print(f"An error occurred: {e}")

