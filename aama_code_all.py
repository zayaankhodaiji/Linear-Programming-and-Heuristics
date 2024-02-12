#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 22:21:23 2023

@author: mirandama
"""

import pandas as pd
import numpy as np
import pulp as pl
import math
import random
from scipy.spatial.distance import cdist
from copy import deepcopy

random.seed(123)
###---------------Question3----------------------------------------------------

# Step 1: Measure distance
# load data from Excel files
customer = pd.read_excel('AAMA_CustomersData.xlsx')
facility = pd.read_excel('facilities.xlsx')

# calculate Haversine distances between each pair of coordinates
def haversine(coord1, coord2):
    R = 6371  # radius of the Earth in kilometers

    lat1, lon1 = coord1
    lat2, lon2 = coord2

    dLat = np.radians(lat2 - lat1)
    dLon = np.radians(lon2 - lon1)

    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)

    a = np.sin(dLat / 2) ** 2 + np.sin(dLon / 2) ** 2 * np.cos(lat1) * np.cos(lat2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    d = R * c

    return d

# extract latitude and longitude values from loc1 and loc2 DataFrames
loc1 = list(zip(customer['LAT'], customer['LONG']))
loc2 = list(zip(facility['Latitude'], facility['Longitude']))

# calculate distances between each pair of coordinates
distances = np.zeros((len(loc1), len(loc2)))

for i, coord1 in enumerate(loc1):
    for j, coord2 in enumerate(loc2):
        distances[i, j] = haversine(coord1, coord2)

# print distances matrix
print(distances)


# customer demand
demand = customer.loc[customer.index[:], customer.columns[-1]]

# facility capacity
capacity = facility.loc[facility.index[:], facility.columns[3]]


# Step 2: Greedy Adaptive Algorithm Implementation
# initialize allocation matrix
allocation = np.zeros((len(customer), len(facility)))

# sort customer demand from high to low
sorted_customer = customer.sort_values(by='Demand', ascending=False)


shortlist = []

while len(sorted_customer) > 0:
     
    random_k = random.randint(1, len(sorted_customer))
    # add top k number of customers into shortlist
    for _ in range(random_k):
        shortlist.append(sorted_customer.iloc[:1])
        sorted_customer.drop(sorted_customer.index[0], inplace=True)
      
    while len(shortlist) > 0:    
        
        random_cust = random.randint(0,len(shortlist)-1)
        
        # allocate the customer to facility j with shortest distance
        for i in range(len(facility)):
            smallest_facility = distances[shortlist[random_cust].index].argsort()[0][i]
            # if no capacity in facility j, allocate the customer to facility with second shortest distance 
            if float(shortlist[random_cust]["Demand"])<= capacity[smallest_facility]:
                allocation[shortlist[random_cust].index[0]][smallest_facility] = 1
                # update facility capacity
                capacity[smallest_facility] -= float(shortlist[random_cust]["Demand"])
                
                del shortlist[random_cust]
                
                break


# Step 3: Print total cost
total_cost = 0 
for i in range(len(customer)):
    for j in range(len(facility)): 
        total_cost += allocation[i][j]*distances[i][j]*demand[i]
print("Total cost_GA: ", total_cost)


###---------------Question4----------------------------------------------------

capacity_new
## Random First Improvement Algorithm without any condition change

# Step 1:
min_cost = deepcopy(distances)
min_allocation_cost = np.min(min_cost, axis=1)
number_reallocations = 0
# Step 2: 

cost = distances 
iteration = 0
max_ite = 99999

# Step 3: Local search
non_empty_solution = True
while iteration <= max_ite and non_empty_solution:
    # randomly select an allocated customer
    allocated_customers = np.argwhere(allocation == 1)
    if len(allocated_customers) == 0:
        non_empty_solution = False
        continue
    random_customer = np.random.choice(allocated_customers[:, 0])
    random_facility = np.argwhere(allocation[random_customer] == 1)[0][0]
    
    # set distance as current allocation cost
    current_allocation_cost = cost[random_customer][random_facility]
    for k in range(len(min_allocation_cost)):
        if current_allocation_cost > min_allocation_cost[k]:
    # try to allocate customer to a different facility
            for j in range(len(facility)):
        
        # check if facility j has enough spare capacity
                if demand[random_customer] <= capacity[j]:
                    new_allocation_cost = distances[random_customer][j]
                # check the new allocation is better than current allocation 
                    if new_allocation_cost < current_allocation_cost:
                 # if yes, reallocate customer to facility j
                        allocation[random_customer][random_facility] = 0
                        allocation[random_customer][j] = 1
                        capacity[random_facility] += demand[random_customer]
                        capacity[j] -= demand[random_customer]
                        number_reallocations += 1
                # start over with a new randomly selected customer
                        break
                    iteration += 1
    # check if minimum allocation cost is updated
        elif np.all(current_allocation_cost < min_allocation_cost):
            min_allocation_cost = current_allocation_cost

total_cost2 = 0
# Step 4: Print total cost
for i in range(len(demand)):
    for j in range(len(capacity)):
        total_cost2 += np.sum(allocation[i][j]*cost[i][j]*demand[i])
print("Total cost_FI: ", total_cost2)
print("Number of Reallocations (FI) =", number_reallocations)
## Mathematical Optimisation Model

# Step 1
cust = customer.index
fac = facility.index
capacity_om = facility.loc[facility.index[:], facility.columns[3]]

# Step 2: Optimisation model

model = pl.LpProblem('Optimization_Model', pl.LpMinimize)

# binary 
x = pl.LpVariable.dicts('x', (cust, fac), lowBound = 0,upBound = 1, cat = "Integer")
r = pl.LpVariable.dicts('r', (cust), lowBound = 0,upBound = 1, cat = "Integer")

# model
model += pl.lpSum(demand[i]*distances[i][j]*x[i][j] for i in cust for j in fac)

# constraints
for i in cust:
    model += pl.lpSum(x[i][j] for j in fac) == 1

for j in fac:
    model += pl.lpSum(demand[i]*x[i][j] for i in cust) <= capacity_om[j]
    
for i in cust:
    model += pl.lpSum(demand[i]*distances[i][j]*(allocation[i][j]-x[i][j]) for j in fac) >= r[i]*60
    model += 1- pl.lpSum(allocation[i][j]*x[i][j] for j in fac) == r[i]
                                                       
    
#print(model)

model.solve()

status = pl.LpStatus[model.status]

allocation_om = np.zeros((len(customer), len(facility)))

for var in model.variables():
    if var.varValue > 0.0:
        print(var.name, "=", var.varValue)
        
print('OOpened facilities:')
if (pl.LpStatus[model.status] == 'Optimal'):
    for i in cust:
        for j in fac:
            allocation_om[i][j] = x[i][j].varValue



