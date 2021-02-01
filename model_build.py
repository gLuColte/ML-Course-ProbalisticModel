'''
COMP9418 Assignment 2

This file is model builder that learns from given initial data, then store the model into model.pkl.

Note, some of the function is based on tutorial.

Name: Kan-Lin Lu    zID: z3417618

'''

from __future__ import division
from __future__ import print_function

# Allowed libraries
import numpy as np
import pandas as pd
from itertools import product, combinations
from collections import OrderedDict as odict
from graphviz import Digraph
from tabulate import tabulate

#Global functions to be used ------------------------------------------------------------------
def printFactor(f):
    """
    argument
    `f`, a factor to print on screen
    """
    # Create a empty list that we will fill in with the probability table entries
    table = list()

    # Iterate over all keys and probability values in the table
    for key, item in f['table'].items():
        # Convert the tuple to a list to be able to manipulate it
        k = list(key)
        # Append the probability value to the list with key values
        k.append(item)
        # Append an entire row to the table
        table.append(k)
    # dom is used as table header. We need it converted to list
    dom = list(f['dom'])
    # Append a 'Pr' to indicate the probabity column
    dom.append('Pr')
    print(tabulate(table,headers=dom,tablefmt='orgtbl'))

# Notice for initiation
data_filename = 'data.csv'
print(f"Begin building model based on {data_filename}...")

# Dataframe set up ---------------------------------------------------------------------------
# First we read the ground truth data:
raw_df = pd.read_csv(data_filename, index_col=0)

# We split it into motion_df and trianing_df, former is same format as input for each round, latter is ground truth
# Dataframe for Storing data (Note this is different to ground truth training data):
# Motion
motion_df = pd.DataFrame(columns=['reliable_sensor1','reliable_sensor2','reliable_sensor3','reliable_sensor4',
                                'unreliable_sensor1','unreliable_sensor2','unreliable_sensor3','unreliable_sensor4',
                                'robot1','robot2','door_sensor1','door_sensor2','door_sensor3','door_sensor4','time',
                                'electricity_price'])
# Copying the dataframe
motion_df = raw_df.loc[:, motion_df.columns]
motion_df.set_index('time', inplace=True)

# Training:
training_df = pd.DataFrame(columns=['time', 'r1','r2','r3','r4','r5','r6','r7','r8','r9','r10','r11','r12','r13','r14','r15',
                                    'r16','r17','r18','r19','r20','r21','r22','r23','r24','r25','r26','r27','r28','r29',
                                    'r30','r31','r32','r33','r34','r35','c1','c2','c3','c4','o1','outside'])
# Copying the datarame
training_df = raw_df.loc[:, training_df.columns]
training_df.set_index('time', inplace=True)

# Area Dataframe:
# With Neighbours we restrict ourself to direct door neighbour, unless specified, see report
# Influence is the designated room based on DAG created

area_df = pd.DataFrame(columns=['type','number', 'door_sensors', 'motion_sensors', 'robot_sensors', 'neighbour_room',
                                'corridor', 'open_area', 'outside', 'influence'])


#Rooms:
area_df.loc[len(area_df)] = ['Room', 1, [], ['unreliable_sensor3'], [], [2, 3], [], [],[], ['r7']]
area_df.loc[len(area_df)] = ['Room', 2, [], [], [], [1, 4], [], [],[], ['r4', 'r1']]
area_df.loc[len(area_df)] = ['Room', 3, [], [], [], [1, 7], [], [],[], ['r1','r7']]
area_df.loc[len(area_df)] = ['Room', 4, [], [], [], [2, 8], [], [],[], ['r1','r8']]
area_df.loc[len(area_df)] = ['Room', 5, [], ['reliable_sensor2'], [], [9, 6], [3], [],[], []]
area_df.loc[len(area_df)] = ['Room', 6, [], [], [], [5], [3], [],[], ['c3','r5']]
area_df.loc[len(area_df)] = ['Room', 7, [], [], [], [3], [1], [],[], ['c2','c1']]
area_df.loc[len(area_df)] = ['Room', 8, ['door_sensor1'], [], [], [4, 9], [], [],[], ['r9','r5']]
area_df.loc[len(area_df)] = ['Room', 9, ['door_sensor1'], [], [], [5, 8, 13], [], [],[], ['r5']]
area_df.loc[len(area_df)] = ['Room', 10, [], [], [], [], [3], [],[], ['c3','r16']]
area_df.loc[len(area_df)] = ['Room', 11, [], [], [], [], [3], [],[], ['c3','r16']]
area_df.loc[len(area_df)] = ['Room', 12, [], [], [], [22], [], [],[1], ['r22','r25']]
area_df.loc[len(area_df)] = ['Room', 13, [], [], [], [9,24], [], [],[], ['r8','r9']]
area_df.loc[len(area_df)] = ['Room', 14, [], [], [], [24], [], [],[], ['r24','r13']]
area_df.loc[len(area_df)] = ['Room', 15, [], [], [], [], [3], [],[], ['c3','r16']]
area_df.loc[len(area_df)] = ['Room', 16, [], ['reliable_sensor1'], [], [], [3], [],[], []]
area_df.loc[len(area_df)] = ['Room', 17, [], [], [], [], [3], [],[], ['c3', 'r16']]
area_df.loc[len(area_df)] = ['Room', 18, [], [], [], [], [3], [],[], ['c3', 'r16']]
area_df.loc[len(area_df)] = ['Room', 19, [], [], [], [], [3], [],[], ['c3', 'r16']]
area_df.loc[len(area_df)] = ['Room', 20, [], [], [], [], [3], [],[], ['c3', 'r16']]
area_df.loc[len(area_df)] = ['Room', 21, [], [], [], [], [3], [],[], ['c3', 'r16']]
area_df.loc[len(area_df)] = ['Room', 22, [], [], [], [12,25], [], [],[], ['r26','c1']]
area_df.loc[len(area_df)] = ['Room', 23, [], [], [], [24], [], [],[], ['r24','r13']]
area_df.loc[len(area_df)] = ['Room', 24, [], ['unreliable_sensor4'], [], [23,13,14], [], [],[], ['r13','r9']]
area_df.loc[len(area_df)] = ['Room', 25, [], ['reliable_sensor3'], [], [22,26], [1], [],[], []]
area_df.loc[len(area_df)] = ['Room', 26, ['door_sensor3'], [], [], [27,25], [], [],[], ['r25']]
area_df.loc[len(area_df)] = ['Room', 27, ['door_sensor3'], [], [], [26,32], [], [],[], ['r26','r25']]
area_df.loc[len(area_df)] = ['Room', 28, [], [], [], [], [4], [],[], ['r35','c4']]
area_df.loc[len(area_df)] = ['Room', 29, [], [], [], [30], [4], [],[], ['o1','c4']]
area_df.loc[len(area_df)] = ['Room', 30, [], [], [], [29], [], [],[], ['r29','c4']]
area_df.loc[len(area_df)] = ['Room', 31, [], ['reliable_sensor4'], [], [32], [], [],[], []]
area_df.loc[len(area_df)] = ['Room', 32, [], [], [], [33,27,31], [], [],[], ['r27','r31']]
area_df.loc[len(area_df)] = ['Room', 33, [], [], [], [32], [], [],[], ['r32','r31','r27']]
area_df.loc[len(area_df)] = ['Room', 34, [], [], [], [], [2], [],[], ['c4','c2']]
area_df.loc[len(area_df)] = ['Room', 35, ['door_sensor4'], [], [], [], [4], [],[], ['o1','c4']]

#Corridors:
area_df.loc[len(area_df)] = ['Corridor', 1, ['door_sensor2'], [], [], [7,25], [2], [],[], ['r25']]
area_df.loc[len(area_df)] = ['Corridor', 2, ['door_sensor2'], [], [], [34], [1,4], [],[], ['c1']]
area_df.loc[len(area_df)] = ['Corridor', 3, [], ['unreliable_sensor2'], [], [5,6,10,11,15,16,17,18,19,20,21], [], [1],[], ['o1','r5']]
area_df.loc[len(area_df)] = ['Corridor', 4, ['door_sensor4'], [], [], [28,29,35], [2], [1],[], ['c2','c1']]

#Open-Space:
area_df.loc[len(area_df)] = ['Open-Area', 1, [], ['unreliable_sensor1'], [], [], [3,4], [],[], ['c4','c2']]

#Outside:
area_df.loc[len(area_df)] = ['Outside', 1, [], [], [], [12], [], [],[], ['r12','r22']]

#Iteration Order
iteration_order = ['r5', 'r25', 'r31', 'r16',   # First Layer
                   'r9', 'r26', 'c1',   # Second Layer
                   'r8', 'r27', 'r22', 'c2', # Third Layer
                   'r13', 'r32', 'r12', 'c4', 'r7', # Fourth Layer
                   'r24', 'r33', 'outside', 'r34', 'o1','r1',   # Fifth Layer
                   'r14', 'r23', 'c3', 'r29', 'r35', 'r3', 'r4',    # Sixth Layer
                   'r6', 'r30', 'r28', 'r2', 'r10', 'r11', 'r15', 'r17', 'r18', 'r19', 'r20', 'r21' # Seventh Layer
                   ]

# HMM - dataframe, the dataframe consiste of transition, emission and outcome space tables:
# Note we introduced a node call infleunce, see report, mainly looking at the neighbour room and check if there is
# increase or decrease in number. It is highly dependent on given structure, therefore, iteration order is essential
hmm_df = pd.DataFrame(columns=['area', 'influence', 'transition_table', 'emission_table', 'outComeSpace'])

# Write the area name into hmm_df
def area_name(x):
    if x.type == 'Outside' :
        return 'outside'
    else:
        return x.type[0].lower() + str(x.number)

# Applying to all Dataframe
hmm_df['area'] = area_df.apply(lambda x: area_name(x), axis=1)

# Write the influence based on graph
hmm_df['influence'] = area_df['influence']

# Write transition table
def transition_table(x,ground_truth_df):
    # First find the column
    if x.type == 'Outside' :
        column_name =  'outside'
    else:
        column_name =  x.type[0].lower() + str(x.number)

    # Locate the stats
    column_stats = ground_truth_df[column_name].to_list()

    # Raw for incrementing
    # Note: p here represents people, where p stands for True, and not_p stands for False
    # Hence p_to_not_p = from people to no people
    transition_raw = {
        'p_to_not_p' : 0,
        'p_to_p' : 0,
        'not_p_to_p' : 0,
        'not_p_to_not_p' : 0,
    }

    # Iterating over every 2 data point to find the transition
    for index in range(0,len(column_stats)):
        # If index is zero, meaning at initial, ignore
        if index == 0:
            continue
        else:
            # Write States:
            # Previous
            if column_stats[index-1] == 0:
                previous_state = False
            elif column_stats[index-1] != 0:
                previous_state = True
            # Current
            if column_stats[index] == 0:
                current_state = False
            elif column_stats[index] != 0:
                current_state = True

            # Incrementation
            if  previous_state and current_state:
                transition_raw['p_to_p']+=1
            elif previous_state and not current_state:
                transition_raw['p_to_not_p']+=1
            elif not previous_state and not current_state:
                transition_raw['not_p_to_not_p'] +=1
            elif not previous_state and current_state:
                transition_raw['not_p_to_p'] +=1

    # Define raw probailities
    # We Add Smoothening
    transition_prob = {
        'p_to_not_p' : (transition_raw['p_to_not_p'] +5000) / ((transition_raw['p_to_not_p'] + transition_raw['p_to_p']) + 5000*4),
        'p_to_p' : (transition_raw['p_to_p'] +5000) / ((transition_raw['p_to_not_p'] + transition_raw['p_to_p']) + 5000*4),
        'not_p_to_p' : (transition_raw['not_p_to_p'] +5000) / ((transition_raw['not_p_to_not_p'] + transition_raw['not_p_to_p']) +5000*4),
        'not_p_to_not_p' : (transition_raw['not_p_to_not_p'] +5000) / ((transition_raw['not_p_to_not_p'] + transition_raw['not_p_to_p']) +5000*4),
    }

    """
    # We explore the Affect of influence: 
    # Weak 
    Weak Smooethning with Alpha = 1:
    | r5_t-1   | r5_t   |        Pr |
    |----------+--------+-----------|
    | true     | true   | 0.616314  |
    | true     | false  | 0.383686  |
    | false    | true   | 0.0613823 |
    | false    | false  | 0.938618  |
    Given:
    {'reliable_sensor2_t': 'motion', 'r5_t-1': 'false'}
    Inference: 
    | r5_t   |      Pr |
    |--------+---------|
    | true   | 0.82533 |
    | false  | 0.17467 |
    
    # Medium
    Medium Smoothening with Alpha = 1000:
    | r5_t-1   | r5_t   |       Pr |
    |----------+--------+----------|
    | true     | true   | 0.277996 |
    | true     | false  | 0.260217 |
    | false    | true   | 0.185698 |
    | false    | false  | 0.484759 |
    Given:
    {'reliable_sensor2_t': 'motion', 'r5_t-1': 'false'}
    Inference: 
    | r5_t   |       Pr |
    |--------+----------|
    | true   | 0.446487 |
    | false  | 0.553513 |
    
    # Strong
    Strong Smoothening with Alpha = 5000:
    | r5_t-1   | r5_t   |       Pr |
    |----------+--------+----------|
    | true     | true   | 0.255964 |
    | true     | false  | 0.252176 |
    | false    | true   | 0.232317 |
    | false    | false  | 0.314559 |

    Given:
    {'reliable_sensor2_t': 'motion', 'r5_t-1': 'false'}
    Inference: 
    | r5_t   |        Pr |
    |--------+-----------|
    | true   | 0.901085  |
    | false  | 0.0989146 |

    """

    # Convert it to table of same format as tutorial
    if x.type != 'Outside':
        area_name = x.type[0].lower() + str(x.number)
    else:
        area_name = 'outside'

    # Transition Table
    transition_table = {
        'dom': (area_name+'_t-1', area_name+'_t'),
        'table': odict([
            (('true', 'true'), transition_prob['p_to_p']),
            (('true', 'false'), transition_prob['p_to_not_p']),
            (('false', 'true'), transition_prob['not_p_to_p']),
            (('false', 'false'), transition_prob['not_p_to_not_p']),
        ])
    }

    # Return the transition Table
    return transition_table

# Applying to all Dataframe
hmm_df['transition_table'] = area_df.apply(lambda x: transition_table(x, training_df), axis=1)

# Write the outcomeSpace
def outcomeSpace_construct(x):
    # First find the area name, and store for easier use latter
    if x.type != 'Outside':
        area_name = x.type[0].lower() + str(x.number)
    else:
        area_name = 'outside'

    # Initialize:
    outComeSpace = {
        area_name+'_t-1': ('true', 'false'),
        area_name+ '_t': ('true', 'false'),
        'robot1_t' : ('motion', 'no motion'), # Assuming Robot sensor moves into all area
        'robot2_t' : ('motion', 'no motion'), # Assuming Robot sensor moves into all area
    }

    # Sensor List:
    for _ in x['door_sensors']:
        outComeSpace[_ + '_t'] = ('motion', 'no motion')

    for _ in x['motion_sensors']:
        outComeSpace[_+'_t'] = ('motion', 'no motion')

    # Influence List
    for _ in x['influence']:
        outComeSpace[_+'_state_t'] = ('decrease', 'increase', 'consistent')

    # Return
    return outComeSpace

# Applying to all Dataframe
hmm_df['outComeSpace'] = area_df.apply(lambda x: outcomeSpace_construct(x), axis=1)

# Write emission table
def emission_table_construct(x, ground_truth, data):
    # We want to construct:
    emission_table = {}

    # Tutorial Functions:
    def prob(factor, *entry):
        return factor['table'][entry]
    def transposeGraph(G):
        GT = dict((v, []) for v in G)
        for v in G:
            for w in G[v]:
                GT[w].append(v)
        return GT
    def allEqualThisIndex(dict_of_arrays, **fixed_vars):
        # base index is a boolean vector, everywhere true
        first_array = dict_of_arrays[list(dict_of_arrays.keys())[0]]
        index = np.ones_like(first_array, dtype=np.bool_)
        for var_name, var_val in fixed_vars.items():
            index = index & (np.asarray(dict_of_arrays[var_name])==var_val)
        return index
    def estProbTable(data, var_name, parent_names, outcomeSpace):
        var_outcomes = outcomeSpace[var_name]
        parent_outcomes = [outcomeSpace[var] for var in (parent_names)]
        # cartesian product to generate a table of all possible outcomes
        all_parent_combinations = product(*parent_outcomes)
        prob_table = odict()
        for i, parent_combination in enumerate(all_parent_combinations):
            parent_vars = dict(zip(parent_names, parent_combination))
            parent_index = allEqualThisIndex(data, **parent_vars)
            possibilities = sum([len(x) for x in outcomeSpace.values()])
            for var_outcome in var_outcomes:
                var_index = (np.asarray(data[var_name])==var_outcome)
                # Apply Additive Smoothening
                prob_table[tuple(list(parent_combination)+[var_outcome])] = ((var_index & parent_index).sum()+1)/ (parent_index.sum() + possibilities)

        return {'dom': tuple(list(parent_names)+[var_name]), 'table': prob_table}
    def join(f1, f2, outcomeSpace):
        # First, we need to determine the domain of the new factor. It will be union of the domain in f1 and f2
        # But it is important to eliminate the repetitions
        common_vars = list(f1['dom']) + list(set(f2['dom']) - set(f1['dom']))

        # We will build a table from scratch, starting with an empty list. Later on, we will transform the list into a odict
        table = list()

        # Here is where the magic happens. The product iterator will generate all combinations of varible values
        # as specified in outcomeSpace. Therefore, it will naturally respect observed values
        for entries in product(*[outcomeSpace[node] for node in common_vars]):
            # We need to map the entries to the domain of the factors f1 and f2
            entryDict = dict(zip(common_vars, entries))
            f1_entry = (entryDict[var] for var in f1['dom'])
            f2_entry = (entryDict[var] for var in f2['dom'])
            # Insert your code here
            p1 = prob(f1, *f1_entry)           # Use the fuction prob to calculate the probability in factor f1 for entry f1_entry
            p2 = prob(f2, *f2_entry)           # Use the fuction prob to calculate the probability in factor f2 for entry f2_entry
            # Create a new table entry with the multiplication of p1 and p2
            table.append((entries, p1 * p2))
        return {'dom': tuple(common_vars), 'table': odict(table)}
    def p_joint(outcomeSpace, cond_tables):
        node_list = list(outcomeSpace.keys())
        for index in range(1, len(node_list)):
            if index == 1:
                p = join(cond_tables[node_list[index-1]], cond_tables[node_list[index]], outcomeSpace)
            else:
                p = join(p, cond_tables[node_list[index]], outcomeSpace)
        return p

    # Graph Checking:
    def graph_check(input_graph, output_name):
        from graphviz import Source
        dot = Digraph(engine="neato", comment='Direct graph example')
        dot.attr(overlap="false", splines="true")
        for v in input_graph.keys():
            dot.node(str(v))
        for v in input_graph.keys():
            for w in input_graph[v]:
                dot.edge(str(v), str(w))
        dot.format = 'png'
        dot.render(filename = output_name)

    # Robot Sensor Data Pre-Processing functions:
    def robot_sensor_room_data(x):
        return x.split(',')[0].replace('(', '').replace("'", "")
    def robot_sensor_count_data(x):
        if int(x.split(',')[1].replace(')', '')) != 0:
            return 'motion'
        else:
            return 'no motion'

    # Pre Processing ---------------------------------------------------------------------------------------
    # First we convert ground truth into true or false
    outcome_map_truth = {True: 'true', False: 'false'}
    # We map the ground truth to True or False, note ground truth is dataframe with people count
    new_groundTruth = ground_truth.astype('bool').replace(outcome_map_truth)

    # As we are considering influence, we also greate a dataframe call state_df
    # We need to also create a column to append onto ground truth that provide 'increase', 'decrease', 'consistent'
    tmp_state_df = pd.DataFrame(index=ground_truth.index,columns=[x + '_state' for x in ground_truth.columns])

    #Iterate over dataframe, as we can not use apply, since we are using every 2 rows:
    first_row_indicator = True # Special case for first row
    for row in ground_truth.itertuples():
        if first_row_indicator:
            previous_row = [0 for x in ground_truth.columns]
            first_row_indicator = False
        # Locate current row
        current_row = [_ for _ in row[1:]]
        # Subtract Current row and Previous Row
        subtraction = list(map(int.__sub__, current_row, previous_row))
        # Now we find an update row
        updated_row = []
        for _ in subtraction:
            # If subtraction has > 0 output means there is an increase in number
            if _ > 0 :
                updated_row.append('increase')
            # If subtraction == 0 output means there is no change in number of people
            elif _ == 0:
                updated_row.append('consistent')
            # If subtraction < 0 output means there is a decrease in number
            elif _ < 0:
                updated_row.append('decrease')
        # Now we store the state into tmporary dataframe
        tmp_state_df.loc[row[0]] = updated_row
        # For next row
        previous_row = current_row

    # We merge the 2 dataframe
    new_groundTruth = new_groundTruth.join(tmp_state_df)

    # We also split into non-robot, and robot data:
    outcome_map_motion = {True: 'motion', False: 'no motion'} # Motion represents there is people, No motion represents there is none
    # We clean robot data into various dataframes, stored based on room in a dictionary
    robot_data_overall = data[['robot1', 'robot2']]
    # Robot1
    robot_data_overall['robot1_room'] = robot_data_overall['robot1'].apply(lambda x: robot_sensor_room_data(x))
    robot_data_overall['robot1_count'] = robot_data_overall['robot1'].apply(lambda x: robot_sensor_count_data(x))
    # Robot2
    robot_data_overall['robot2_room'] = robot_data_overall['robot2'].apply(lambda x: robot_sensor_room_data(x))
    robot_data_overall['robot2_count'] = robot_data_overall['robot2'].apply(lambda x: robot_sensor_count_data(x))

    # Non Robot Data:
    non_robot_data = data[[x for x in data.columns if x not in ['robot1', 'robot2', 'electricity_price']]]
    non_robot_data['door_sensor1'] = data['door_sensor1'].astype('bool').replace(outcome_map_motion)
    non_robot_data['door_sensor2'] = data['door_sensor2'].astype('bool').replace(outcome_map_motion)
    non_robot_data['door_sensor3'] = data['door_sensor3'].astype('bool').replace(outcome_map_motion)
    non_robot_data['door_sensor4'] = data['door_sensor4'].astype('bool').replace(outcome_map_motion)

    # Then we establish the outcomeSpace and Graph, note emission dose not rely on t-1:
    # We create a new outcome Space so we dont overwrite the original
    reduced_outcomeSpace = {}
    for _, val in x.outComeSpace.items():
        if '-1' in _:
            continue #Ignore
        elif '_t' in _:
            reduced_outcomeSpace[_[:-2]] = val # We remove_t here for latter operation
        else:
            reduced_outcomeSpace[_] = val

    # Now we are getting somewhere, we store the domain
    emission_table['dom'] = tuple([x + '_t' for x in reduced_outcomeSpace.keys()])

    # Now we establish Graph
    # In theory, it should be a DAG with a single parent to sensors and influences
    graph = {}
    motion_sensor, door_sensor = '',''
    influence = []
    for _ in reduced_outcomeSpace.keys():
        if  _ == x.area:
            graph[_] = [y for y in reduced_outcomeSpace.keys() if y!=_]
            continue # Go to next
        elif 'reliable' in _ :
            motion_sensor = _
        elif 'door' in _:
            door_sensor = _
        elif 'robot' not in _:
            influence.append(_)
        graph[_] = []

    # Plot for checking
    if x.area == 'r9':
        graph_check(graph, 'emission_construct_eval')
    # Tranpose the graph and store for easier operation
    graphT = transposeGraph(graph)

    # Now we construct the CPT for each node given the data
    conditional_probs = {}

    # First we find prior:
    # P(People)
    conditional_probs[x.area] = estProbTable(new_groundTruth, x.area, [], reduced_outcomeSpace)

    # P( Motion Sensor | People )
    # Find the sensor conditional probabilities in the room:
    if motion_sensor != '':
       conditional_probs[motion_sensor] = estProbTable(new_groundTruth.join(non_robot_data), motion_sensor, graphT[motion_sensor], reduced_outcomeSpace)

    # P ( Door Sensor | People)
    # Find the Sensor conditional probabilities in the room:
    if door_sensor != '':
       conditional_probs[door_sensor] = estProbTable(new_groundTruth.join(non_robot_data), door_sensor, graphT[door_sensor], reduced_outcomeSpace)

    # P ( Robot Sensor | People)
    # Robot 1 We need to consider both Robot 1 and robot 2
    # We take a smaller dataframe
    rb1_tmp_df = robot_data_overall.loc[robot_data_overall['robot1_room'] == x.area ]['robot1_count']
    rb1_tmp_df = rb1_tmp_df.to_frame().rename({'robot1_count':'robot1'}, axis='columns')
    # Then we take the join of rb1 with new ground truth
    conditional_probs['robot1'] = estProbTable(rb1_tmp_df.join(new_groundTruth), 'robot1', graphT['robot1'], reduced_outcomeSpace)

    # Robot 2 We take a smaller dataframe
    rb2_tmp_df = robot_data_overall.loc[robot_data_overall['robot2_room'] == x.area ]['robot2_count']
    rb2_tmp_df = rb2_tmp_df.to_frame().rename({'robot2_count':'robot2'}, axis='columns')
    # Then we take the join of rb1 with new ground truth
    conditional_probs['robot2'] = estProbTable(rb2_tmp_df.join(new_groundTruth), 'robot2', graphT['robot2'], reduced_outcomeSpace)

    # P( Influences | People )
    # Influences:
    for _ in influence:
        conditional_probs[_] = estProbTable(new_groundTruth.join(non_robot_data), _, graphT[_], reduced_outcomeSpace)

    # Find Joint Probability:
    joint_p = p_joint(reduced_outcomeSpace, conditional_probs)

    # Now Convert it to emission Table, where p is given:
    # P( Join ) / P ( People )
    # We make a copy:
    tmp_table = joint_p.copy()
    # Now we iterate over joint probability
    for _, val in joint_p['table'].items():
        # Find the entires of P( People )
        area_val = (_[0],)
        # Divide the entry
        tmp_table['table'][_] =tmp_table['table'][_]/conditional_probs[x.area]['table'][area_val]
    emission_table['table'] = tmp_table['table']

    """
    # For testing Purpose we look at a single node
    if x.area == 'r5':
        print("Joint")
        printFactor(joint_p)
        print("Emission")
        printFactor(emission_table)
        print("Conditional")
        for _, factor in conditional_probs.items():
            printFactor(factor)
        evi_dict = {'reliable_sensor2':'motion'}
        test = emission_table.copy()
        test['dom'] = joint_p['dom']
        printFactor(query(test, reduced_outcomeSpace, x.area, evi_dict))
        sys.exit()
    """
    return emission_table
hmm_df['emission_table'] = hmm_df.apply(lambda x: emission_table_construct(x, training_df, motion_df), axis=1)

# Let's try to reduce speed:
def merge_columns(x):
    return {0 : x.outComeSpace,1: x.transition_table,2: x.emission_table,}
hmm_df['merged'] = hmm_df.apply(lambda x: merge_columns(x), axis=1)

# Save Model as Pickle
hmm_df.to_pickle('model.pkl')

print("Model Has finished been built and stored in model.pkl. Execute solution.py for further stage.")



