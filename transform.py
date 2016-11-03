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


def updateMasterCompanies(masterCompanies,partner) :
    if partner in masterCompanies:
        masterCompanies[partner] = masterCompanies[partner] + 1
    else:
        masterCompanies[partner] = 1


def dump(partner,dictionary,f) :

    tmpDict = dict()

    for key in sorted(dictionary):
        #print "%s: %s" % (key, dictionary[key])
        partner_value = dictionary[key]

        tmparray = key.split("-")

        if tmparray[0] == partner :

            if partner in tmpDict:
                tmpDict[partner] = tmpDict[partner] + "," + tmparray[1]
            else:
                tmpDict[partner] =  tmparray[1]

    for key in tmpDict :
        f.write("%s: %s" % (key, tmpDict[key]))
        print " HERE you go" + "%s: %s" % (key, tmpDict[key])


if __name__ == '__main__':

    masterProducts = dict()
    masterVersion = dict()
    masterOs = dict()
    masterCompanies = dict()

    distinctProducts = dict()
    distinctVersion  = dict()
    distinctOs       = dict()

    data = pd.read_csv("../export_psirt.csv")

    import csv
    import sys

    f = open("../export_psirt.csv", 'r')
    try:
        reader = csv.reader(f)
        for row in reader:
            #print row
            updateDistinct(masterProducts,distinctProducts,row[0],row[2])
            updateDistinct(masterVersion,distinctVersion,row[0], row[3])
            updateDistinct(masterOs,distinctOs,row[0], row[4])
            updateMasterCompanies(masterCompanies,row[0])
    finally:
        f.close()

    print "************************** Master Companies *******************************"

    for partner in sorted(masterCompanies):

        print "%s: %s" % (partner, masterCompanies[partner])

        print "************************** Distinct Products *******************************"
        f = open("../product_vector.csv", 'w')
        dump(partner, distinctProducts, f)
        f.close()


if False :

        print "************************** Master Products *******************************"
        f = open("../product_master.csv", 'w')
        dump(partner,masterProducts,f)
        f.close()

        print "************************** Master Version *******************************"
        f = open("../version_master.csv", 'w')
        dump(partner,masterVersion,f)
        f.close()

        print "************************** Master Os *******************************"
        f = open("../os_master.csv", 'w')
        dump(partner,masterOs,f)
        f.close()


        print "************************** Distinct Products *******************************"
        f = open("../product_vector.csv", 'w')
        dump(partner,distinctProducts,f)
        f.close()

        print "************************** Distinct Products *******************************"
        f = open("../version_vector.csv", 'w')
        dump(partner,distinctVersion,f)
        f.close()

        print "************************** Distinct Products *******************************"
        f = open("../os_vector.csv", 'w')
        dump(partner,distinctOs,f)
        f.close()
