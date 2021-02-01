'''
COMP9418 Assignment 2

This file is for execution mainly grabbing model from model.pkl and make inference.

Note, some of the function is based on tutorial.

Name: Kan-Lin Lu    zID: z3417618

'''
# Make division default to floating-point, saving confusion
from __future__ import division
from __future__ import print_function

# Allowed libraries
import time, sys
import pandas as pd
from itertools import product
from collections import OrderedDict as odict
from tabulate import tabulate

###################################
# Global Variables:

# Globale Previous State, represents whether there is people or not,
# False = No People, True = People
# Note this is initialized to 'false' for initial state, except outside is true
previous_overall_state = {'r5': 'false', 'r25': 'false', 'r31': 'false', 'r16': 'false', 'r9': 'false', 'r26': 'false',
                          'c1': 'false', 'r8': 'false', 'r27': 'false', 'r22': 'false', 'c2': 'false', 'r13': 'false',
                          'r32': 'false', 'r12': 'false', 'c4': 'false', 'r7': 'false', 'r24': 'false', 'r33': 'false',
                          'outside': 'true', 'r34': 'false', 'o1': 'false', 'r1': 'false', 'r14': 'false', 'r23': 'false',
                          'c3': 'false', 'r29': 'false', 'r35': 'false', 'r3': 'false', 'r4': 'false', 'r6': 'false',
                          'r30': 'false', 'r28': 'false', 'r2': 'false', 'r10': 'false', 'r11': 'false', 'r15': 'false',
                          'r17': 'false', 'r18': 'false', 'r19': 'false', 'r20': 'false', 'r21': 'false'}

# Order for iterating based on structure, refer to report
iteration_order = ['r5', 'r25', 'r31', 'r16',   # First Layer
                   'r9', 'r26', 'c1',   # Second Layer
                   'r8', 'r27', 'r22', 'c2', # Third Layer
                   'r13', 'r32', 'r12', 'c4', 'r7', # Fourth Layer
                   'r24', 'r33', 'outside', 'r34', 'o1','r1',   # Fifth Layer
                   'r14', 'r23', 'c3', 'r29', 'r35', 'r3', 'r4',    # Sixth Layer
                   'r6', 'r30', 'r28', 'r2', 'r10', 'r11', 'r15', 'r17', 'r18', 'r19', 'r20', 'r21' # Seventh Layer
                   ]

# Read the model as dataframe
read_model = pd.read_pickle('model.pkl')
# Store into dictionary, 0 = outcomeSpace, 1 = transition Table 2 = emission Table:
model_dict = { x: read_model.loc[read_model['area'] == x]['merged'].values[0] for x in iteration_order }

# Light Room List:
light_rooms = [ 'r' +str(x) for x in range(1, 36)]

###################################

# Functions to be used from tutorial:
def prob(factor, *entry):
    """
    argument
    `factor`, a dictionary of domain and probability values,
    `entry`, a list of values, one for each variable in the same order as specified in the factor domain.

    Returns p(entry)
    """

    return factor['table'][entry]     # insert your code here, 1 line
def join(f1, f2, outcomeSpace):
    """
    argument
    `f1`, first factor to be joined.
    `f2`, second factor to be joined.
    `outcomeSpace`, dictionary with the domain of each variable

    Returns a new factor with a join of f1 and f2
    """

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
def marginalize(f, var, outcomeSpace):
    """
    argument
    `f`, factor to be marginalized.
    `var`, variable to be summed out.
    `outcomeSpace`, dictionary with the domain of each variable

    Returns a new factor f' with dom(f') = dom(f) - {var}
    """

    # Let's make a copy of f domain and convert it to a list. We need a list to be able to modify its elements
    new_dom = list(f['dom'])

    new_dom.remove(var)            # Remove var from the list new_dom by calling the method remove(). 1 line
    table = list()                 # Create an empty list for table. We will fill in table from scratch. 1 line
    for entries in product(*[outcomeSpace[node] for node in new_dom]):
        s = 0;                     # Initialize the summation variable s. 1 line

        # We need to iterate over all possible outcomes of the variable var
        for val in outcomeSpace[var]:
            # To modify the tuple entries, we will need to convert it to a list
            entriesList = list(entries)
            # We need to insert the value of var in the right position in entriesList
            entriesList.insert(f['dom'].index(var), val)

            p = prob(f, *tuple(entriesList))     # Calculate the probability of factor f for entriesList. 1 line
            s = s + p                            # Sum over all values of var by accumulating the sum in s. 1 line

        # Create a new table entry with the multiplication of p1 and p2
        table.append((entries, s))
    return {'dom': tuple(new_dom), 'table': odict(table)}
def evidence(var, e, outcomeSpace):
    """
    argument
    `var`, a valid variable identifier.
    `e`, the observed value for var.
    `outcomeSpace`, dictionary with the domain of each variable

    Returns dictionary with a copy of outcomeSpace with var = e
    """
    newOutcomeSpace = outcomeSpace.copy()      # Make a copy of outcomeSpace
    newOutcomeSpace[var] = (e,)                # Replace the domain of variable var with a tuple with a single element e. 1 line
    return newOutcomeSpace
def normalize(f):
    """
    argument
    `f`, factor to be normalized.

    Returns a new factor f' as a copy of f with entries that sum up to 1
    """
    table = list()
    sum = 0
    for k, p in f['table'].items():
        sum = sum + p
    for k, p in f['table'].items():
        table.append((k, p/sum))
    return {'dom': f['dom'], 'table': odict(table)}
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
def query(p, outcomeSpace, q_vars, q_evi):
    pm = p.copy()  # make a copy first of the probability tables
    outSpace = outcomeSpace.copy()  # make a copy of the outcome space

    # We first set the evidence - note it can be multiple
    for var, evi in q_evi.items():
        outSpace = evidence(var, evi, outSpace)  # update outSpaceto given evidence

    # then we eliminate hideden variables NOT in the query
    for var in outSpace:
        if not var in q_vars:
            pm = marginalize(pm, var, outSpace)
    return normalize(pm)

# Get Action function:
def get_action(sensor_data):

    # Global Variables
    global previous_overall_state, iteration_order, read_model, light_rooms

    # Timing
    #print("Iteration Started: ")
    #start_time = time.time()

    # To avoid complexity, we first update robot sensor data to robot1_loc, robot1_val and stoed into sensor data dictionary
    robot1_loc, robot1_val = 'robot1_loc', 'robot1_val'
    if sensor_data['robot1'] != None:
        sensor_data[robot1_loc] = sensor_data['robot1'].split(',')[0].replace("'", "").replace("(", "")
        sensor_data[robot1_val] = int(sensor_data['robot1'].split(',')[1].replace("'", "").replace(")", ""))
    robot2_loc, robot2_val = 'robot2_loc', 'robot2_val'
    if sensor_data['robot2'] != None:
        sensor_data[robot2_loc] = sensor_data['robot2'].split(',')[0].replace("'", "").replace("(", "")
        sensor_data[robot2_val] = int(sensor_data['robot2'].split(',')[1].replace("'", "").replace(")", ""))

    # Then we initialize a current state dictionary in recording whether there is people or not in each area
    current_overall_state = {x: '' for x in iteration_order}

    # Now we iterate based on above iteration order
    for _ in iteration_order:
        # First read the respective outcomeSpace, transition table and emission table from dataframe
        outcomeSpace = model_dict[_][0]
        transition_table = model_dict[_][1]
        emission_table = model_dict[_][2]

        # Think of it as Bayesian network, we got 2 probability table
        joint_et = join(emission_table, transition_table, outcomeSpace)

        # Evidence:
        # Then we construct our evidence of current_t:
        q_evi = {}
        # We iterate through outcomeSpace to find the possible evidence:
        for evi in outcomeSpace.keys():
            # To make it faster we store to name of sensor to var_name
            if 'reliable' in evi or 'door' in evi:
                var_name = evi[:-2]
            else:
                var_name = evi.split('_')[0]

            # Possible evidence:
            if _ in var_name:
                continue  # Ignore own state
            # Non Robot Sensors:
            if 'reliable' in evi and sensor_data[var_name] != None:
                # Store into evidence dictionary
                q_evi[evi] = sensor_data[evi[:-2]]
            elif 'door' in evi and sensor_data[var_name] != None:
                if sensor_data[evi[:-2]] > 0:
                    q_evi[evi] = 'motion' # If greater then means motion
                else:
                    q_evi[evi] = 'no motion' # Else there was no motion
            # Robot Sensors:
            elif 'robot' in evi and (evi[:-2] + '_loc' in sensor_data.keys()) and (sensor_data[evi[:-2] + '_loc'] == _):
                if sensor_data[evi[:-2] + '_val'] > 0:
                    q_evi[evi] = 'motion' # If greater then means motion
                else:
                    q_evi[evi] = 'no motion'  # Else there was no motion
            # Room State:
            elif 'state' in evi:
                if current_overall_state[var_name] == previous_overall_state[var_name]:
                    # Equal means consistent
                    q_evi[evi] = 'consistent'
                elif current_overall_state[var_name] == 'true' and previous_overall_state[var_name] == 'false':
                    # True to False means decreased
                    q_evi[evi] = 'decrease'
                elif current_overall_state[var_name]  == 'false' and previous_overall_state[var_name] == 'true':
                    # False to True means increased
                    q_evi[evi] = 'increase'

        # Now we also add last state as evidence:
        q_evi[_+'_t-1'] = previous_overall_state[_]

        # Now we query:
        query_with_evi = query(joint_et, outcomeSpace, _ + '_t', q_evi)
        # Find the Maximum, in other words, most probable explain
        max_prob = ('', 0)
        for entry, value in query_with_evi['table'].items():
            if value > max_prob[1]:
                max_prob = (entry, value)

        # Store into current overall State
        current_overall_state[_] = max_prob[0][0]

        """
        # For debuggin Purpose:
        if _ == 'r9':
            print("Emission Table -------")
            printFactor(emission_table)
            print("Transition Table -------")
            printFactor(transition_table)
            print("Joint Table -------")
            printFactor(joint_et)
            print("Evidence -------")
            print(q_evi)
            print("OutcomeSpace -------")
            print(outcomeSpace)
            # Now we query the joint_et table
            printFactor(query(joint_et, outcomeSpace, _+'_t', q_evi))
            sys.exit()
        """

    # Now we completed Iterating:
    # We need to convert it to light switch:
    actions_dict = {}
    for _ in light_rooms:
        if current_overall_state[_] == 'false':
            actions_dict['lights' + _.replace('r', '')] = 'off'
        elif current_overall_state[_] == 'true':
            actions_dict['lights' + _.replace('r', '')] = 'on'

    # Store overall states
    previous_overall_state = current_overall_state

    # Timing
    #print(f"Total {time.time() - start_time} seconds")

    # Returning the actions
    return actions_dict