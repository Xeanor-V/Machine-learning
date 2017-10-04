import random
def partition(data, fraction): 
    ldata = list(data) random.shuffle(ldata) breakPoint = int(len(ldata) * fraction) 
    return ldata[:breakPoint], ldata[breakPoint:]

monk1train, monk1val = partition(m.monk1, 0.6)