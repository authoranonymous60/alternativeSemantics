import itertools
import json

import subs as subs   # your original import

maxExamples = 50000


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def has_duplicate_elements(input_list):
    return len(input_list) != len(set(input_list))


def add_example(queryItems, seen, S1, S2, A, focus, logic, alternative, position):
    """
    Add a single example if it's not already seen.
    Returns the updated position.
    """
    key = (S1, S2, A, focus)
    if key not in seen:
        queryItems.append({
            'S1': S1,
            'S2': S2,
            'A': A,
            'focus': focus,
            'logic': logic,            
            'alternative': alternative,            
            'position': position            
        })
        seen.add(key)
        return position + 1

    return position


# ------------------------------------------------------------
# Build the space of combinations
# ------------------------------------------------------------
XList = [subs.X]
XAltList = [subs.X]
yList = [subs.y]
yAltList = [subs.y]

exampleList = XList + XAltList + yList + yAltList

comb = [p for p in itertools.product(*exampleList)]
comb = list(filter(lambda x: not has_duplicate_elements(x), comb))


# ------------------------------------------------------------
# Generate examples
# ------------------------------------------------------------
queryItems = []
seen = set()
position = 0

for c in comb:
    if len(queryItems) >= maxExamples:
        break

    X = c[0]
    XF = X.upper()
    XAlt = c[1]
    y = c[2]
    yF = y.upper()
    yAlt = c[3]



    ## Focus 1
    # Case 1: TRUE (A)
    S1 = f"Sam only gave {XF} {y}."
    S2 = f"Sam didn't give {XAlt} {y}."
    A = "A"
    focus = 1
    logic = "NEG"
    alternative = 1
    position = add_example(queryItems, seen, S1, S2, A, focus, logic, alternative, position)
    
    # Case 2: INDEPENDENT (B)
    S1 = f"Sam only gave {XF} {y}."
    S2 = f"Sam didn't give {X} {yAlt}."
    A = "B"
    focus = 1
    logic = "NEG"    
    alternative = "Y"    
    position = add_example(queryItems, seen, S1, S2, A, focus, logic, alternative, position)
    
    # Case 3: INDEPENDENT (B)
    S1 = f"Sam only gave {XF} {y}."
    S2 = f"Sam also gave {X} {yAlt}."
    A = "B"
    focus = 1
    logic = "POS"    
    alternative = "Y"        
    position = add_example(queryItems, seen, S1, S2, A, focus, logic, alternative, position)
    
    # Case 3: FALSE (C)
    S1 = f"Sam only gave {XF} {y}."
    S2 = f"Sam also gave {XAlt} {y}."
    A = "C"
    focus = 1
    logic = "POS"        
    alternative = 1        
    position = add_example(queryItems, seen, S1, S2, A, focus, logic, alternative, position)


    ## Focus 2
    # Case 1: TRUE (A)
    S1 = f"Sam only gave {X} {yF}."
    S2 = f"Sam didn't give {X} {yAlt}."
    A = "A"
    focus = 2
    logic = "NEG"        
    alternative = "Y"        
    position = add_example(queryItems, seen, S1, S2, A, focus, logic, alternative, position)

    # Case 2: INDEPENDENT (B)
    S1 = f"Sam only gave {X} {yF}."
    S2 = f"Sam didn't give {XAlt} {y}."
    A = "B"
    focus = 2
    logic = "NEG"            
    alternative = 1        
    position = add_example(queryItems, seen, S1, S2, A, focus, logic, alternative, position)
    
    # Case 3: INDEPENDENT (B)
    S1 = f"Sam only gave {X} {yF}."
    S2 = f"Sam also gave {XAlt} {y}."
    A = "B"
    focus = 2
    logic = "POS"            
    alternative = 1        
    position = add_example(queryItems, seen, S1, S2, A, focus, logic, alternative, position)

    # Case 4: FALSE (C)
    S1 = f"Sam only gave {X} {yF}."
    S2 = f"Sam also gave {X} {yAlt}."
    A = "C"
    focus = 2
    logic = "POS"                
    alternative = 2        
    position = add_example(queryItems, seen, S1, S2, A, focus, logic, alternative, position)
    
# ------------------------------------------------------------
# Save output
# ------------------------------------------------------------
queryOut = "all_examples.json"
with open(queryOut, 'w', encoding='utf-8') as f:
    json.dump(queryItems, f, ensure_ascii=False, indent=4)
