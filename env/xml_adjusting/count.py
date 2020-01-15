import os
import xml.etree.ElementTree as ET
from collections import defaultdict
from math import ceil

"""
This script provides functionality to either count the number of furniture and
their parts, or assign N people a fair amount of furniture to label by parts.
"""
def count_furn():
    """
    Outputs a file that counts the furnitures and their parts
    """
    furn_folders = [ d for d in os.scandir('env/models/assets/objects/complete') if d.is_dir()]
    furn_folders += [d for d in os.scandir('env/models/assets/objects/incomplete') if d.is_dir()]
    furn_count = defaultdict(list)
    total = 0
    total_parts = 0
    for folder in furn_folders:
        for f in os.scandir(folder.path):
            if f.is_file() and f.path.endswith('xml'):
                tree = ET.parse(f.path)
                wb = tree.getroot().find('worldbody')
                furn_count[len(wb)].append(f.name)
                total += 1
                total_parts += len(wb)

    with open('count.txt', 'w') as f:
        f.write(f'Total furniture: {total}\n')
        f.write(f'Total parts: {total_parts}\n')
        f.write('-' * 80 + '\n')
        for count in sorted(furn_count.keys()):
            f.write(f'{count}-part furnitures: {len(furn_count[count])}\n')
            for furn in furn_count[count]:
                f.write(furn + '\n')
            f.write('-' * 80 + '\n')
def assign(num_persons):
    """
    Total furniture: 60
    Total parts: 467
    Number of lablers: 6

    To make the labeling fair, we calculate the avg number of parts labeled for
    each group. 467/6 = 77.83 parts for each group.

    We can assign furniture greedily in descending order of parts such that
    the remainder parts can be taken by the smallest furniture.
    """
    furn_folders = [ d for d in os.scandir('env/models/assets/objects/complete') if d.is_dir()]
    furn_folders += [d for d in os.scandir('env/models/assets/objects/incomplete') if d.is_dir()]
    furn_count = defaultdict(list)
    total = 0
    total_parts = 0
    for folder in furn_folders:
        for f in os.scandir(folder.path):
            if f.is_file() and f.path.endswith('xml'):
                tree = ET.parse(f.path)
                wb = tree.getroot().find('worldbody')
                furn_count[len(wb)].append(f.name)
                total += 1
                total_parts += len(wb)

    persons = []
    remaining_budget = []
    descending_num_parts = sorted(furn_count.keys(), reverse=True)
    for i in range(num_persons):
        p = []
        budget = ceil(total_parts / num_persons)
        while budget >= 0:
            # find the biggest furniture to allot
            max_part = 0
            for part in descending_num_parts:
                if part <= budget and len(furn_count[part]) > 0:
                    max_part = part
                    break
            if max_part != 0:
                budget -= max_part
                p.append((furn_count[max_part].pop(), max_part))
            else: # if no furniture fit budget, we are done
                break
        print(f'Person {i} with budget {budget} remaining')
        remaining_budget.append(budget)
        persons.append(p)

    print('Furniture left over:', furn_count)

    with open('assignments.txt', 'w') as f:
        for i, p in enumerate(persons):
            rb = remaining_budget[i]
            f.write(f'Person {i}, remaining budget: {rb}\n')
            for furn_name, num_parts in p:
                f.write(f'{furn_name}, {num_parts} parts\n')
            f.write('-' * 80 + '\n')

        f.write('Furn left over\n')
        for k,v in furn_count.items():
            if len(v) > 0:
                for name in v:
                    f.write(f'{name}, {k} parts\n')

if __name__ == "__main__":
    assign(6)