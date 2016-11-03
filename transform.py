import numpy as np
import pandas as pd
#import renders as rs
from IPython.display import display # Allows the use of display() for DataFrames


def updateDistinct(masterMetaDict, customerLevermMetaDict,partner,item) :

    key = partner+"-"+item
    if key in customerLevermMetaDict :
        customerLevermMetaDict[key] = customerLevermMetaDict[key] + 1
    else :
        customerLevermMetaDict[key] =  1

    if item in masterMetaDict:
        masterMetaDict[item] = masterMetaDict[item] + 1
    else:
        masterMetaDict[item] = 1

def dump(dictionary) :

    for key in sorted(dictionary):
        print "%s: %s" % (key, dictionary[key])



if __name__ == '__main__':

    masterProducts = dict()
    masterVersion = dict()
    masterOs = dict()

    distinctProducts = dict()
    distinctVersion  = dict()
    distinctOs       = dict()

    data = pd.read_csv("export_psirt.csv")

    import csv
    import sys

    f = open("export_psirt.csv", 'r')
    try:
        reader = csv.reader(f)
        for row in reader:
            #print row
            updateDistinct(masterProducts,distinctProducts,row[0],row[2])
            updateDistinct(masterVersion,distinctVersion,row[0], row[3])
            updateDistinct(masterOs,distinctOs,row[0], row[4])
    finally:
        f.close()

    print "************************** Master Products *******************************"
    dump(masterProducts)

    print "************************** Master Version *******************************"
    dump(masterVersion)

    print "************************** Master Os *******************************"
    dump(masterOs)

    print "************************** Distinct Products *******************************"
    dump(distinctProducts)

    print "************************** Distinct Products *******************************"
    dump(distinctVersion)

    print "************************** Distinct Products *******************************"
    dump(distinctOs)