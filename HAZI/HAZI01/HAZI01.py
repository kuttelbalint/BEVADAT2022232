# %%
#Create a function that returns with a subsest of a list.
#The subset's starting and ending indexes should be set as input parameters (the list aswell).
#return type: list
#function name must be: subset
#input parameters: input_list,start_index,end_index

# %%
def subset(input_list, start_index, end_index):
    return input_list[start_index:end_index]

# %%
#Create a function that returns every nth element of a list.
#return type: list
#function name must be: every_nth
#input parameters: input_list,step_size

# %%
def every_nth(input_list, step_size):
    return input_list[::step_size]

# %%
#Create a function that can decide whether a list contains unique values or not
#return type: bool
#function name must be: unique
#input parameters: input_list

# %%
def unique(input_list):
    listIsunique = len(set(input_list)) == len(input_list)

    return listIsunique


# %%
#Create a function that can flatten a nested list ([[..],[..],..])
#return type: list
#fucntion name must be: flatten
#input parameters: input_list

# %%
def flatten(input_list):
    flat_list = []
    
    for sublist in input_list:
        if isinstance(sublist, list):
            flat_list.extend(flatten(sublist))
        else:
            flat_list.append(sublist)
    return flat_list

# %%
#Create a function that concatenates n lists
#return type: list
#function name must be: merge_lists
#input parameters: *args


# %%
def merge_lists(*args):
    merged_list = []
    for element in args:
        merged_list += element
    return merged_list

# %%
#Create a function that can reverse a list of tuples
#example [(1,2),...] => [(2,1),...]
#return type: list
#fucntion name must be: reverse_tuples
#input parameters: input_list

# %%
def reverse_tuples(input_list):
    return [tuple(reversed(t)) for t in input_list]


# %%
#Create a function that removes duplicates from a list
#return type: list
#fucntion name must be: remove_tuplicates
#input parameters: input_list

# %%
def remove_tuplicates(input_list):
    return list(set(input_list))

# %%
#Create a function that transposes a nested list (matrix)
#return type: list
#function name must be: transpose
#input parameters: input_list

# %%
def transpose(input_list):
    return [list(row) for row in zip(*input_list)]

# %%
#Create a function that can split a nested list into chunks
#chunk size is given by parameter
#return type: list
#function name must be: split_into_chunks
#input parameters: input_list,chunk_size

# %%
def split_into_chunks(input_list, chunk_size):
    return [input_list[i:i+chunk_size] for i in range(0, len(input_list), chunk_size)]


# %%
#Create a function that can merge n dictionaries
#return type: dictionary
#function name must be: merge_dicts
#input parameters: *dict

# %%
def merge_dicts(*dict):
    result_dict = {}
    for i in dict:
        result_dict.update(i)
    return result_dict

# %%
#Create a function that receives a list of integers and sort them by parity
#and returns with a dictionary like this: {"even":[...],"odd":[...]}
#return type: dict
#function name must be: by_parity
#input parameters: input_list

# %%
def by_parity(input_list):
    even = []
    odd = []
    for num in input_list:
        if num % 2 == 0:
            even.append(num)
        else:
            odd.append(num)
    return {"even": sorted(even), "odd": sorted(odd)}

# %%
#Create a function that receives a dictionary like this: {"some_key":[1,2,3,4],"another_key":[1,2,3,4],....}
#and return a dictionary like this : {"some_key":mean_of_values,"another_key":mean_of_values,....}
#in short calculates the mean of the values key wise
#return type: dict
#function name must be: mean_key_value
#input parameters: input_dict

# %%


# %%
#If all the functions are created convert this notebook into a .py file and push to your repo


