# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 12:05:59 2016
 
@authors: Sergio Alises Mendiola and Raul Gallego de la Sacristana Alises

"""
import codecs
def load_data_usa(path):
        
    f = codecs.open(path, "r", "utf-8")
    print(f)
    names = []
    count = 0
    for line in f:

        if count > 0: 
    		# remove double quotes
            row = line.split(";")
            row.pop(0)
            row.pop(0)
            row.pop(0)
            row.pop(0)
            if row != []:
                data = [float(el) for el in row]
                names.append(data)
        count += 1
    return names

