import numpy as np
import pandas as pd
#import renders as rs
from IPython.display import display # Allows the use of display() for DataFrames


def updateDistinct(masterMetaDict, customerLevermMetaDict,partner,item) :

    key = partner+":"+item
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


def dumpHeader(keyStr,masterDict,f) :

    f.write(keyStr + ",")
    for key in sorted(masterDict):
        if key == None or key == "" : continue
        f.write(key + ",")

    f.write("\n")


def dump(partner,masterDict,dictionary,f) :

    tmpDict = dict()


    tmpDict[partner] = ""

    for key in sorted(masterDict) :

        #print "%s: %s" % (key, dictionary[key])

        if partner+":"+key in dictionary :

            value = dictionary[partner+":"+key]
            if partner in tmpDict:
                tmpDict[partner] = str(tmpDict[partner]) + "," + str(value)
            else:
                tmpDict[partner] =  value

        else :
            tmpDict[partner] = str(tmpDict[partner]) + ","



    for key in tmpDict :
        f.write("%s: %s" % (key, tmpDict[key]))
        f.write("\n")
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
            updateDistinct(masterOs, distinctOs, row[0], row[3])
            updateDistinct(masterVersion,distinctVersion,row[0], row[4])
            updateMasterCompanies(masterCompanies,row[0])
    finally:
        f.close()

    print "************************** Master Companies *******************************"

    f = open("../product_vector.csv", 'w')
    dumpHeader("PARTNER",masterProducts, f)
    for partner in sorted(masterCompanies):

        print "%s: %s" % (partner, masterCompanies[partner])
        print "************************** Distinct Products *******************************"
        dump(partner, masterProducts,distinctProducts, f)

    f.close()

    f = open("../version_vector.csv", 'w')
    dumpHeader("PARTNER", masterVersion, f)
    for partner in sorted(masterCompanies):
        print "%s: %s" % (partner, masterCompanies[partner])
        print "************************** Distinct Products *******************************"
        dump(partner, masterVersion, distinctVersion, f)

    f.close()

    f = open("../os_vector.csv", 'w')
    dumpHeader("PARTNER", masterOs, f)
    for partner in sorted(masterCompanies):
        print "%s: %s" % (partner, masterCompanies[partner])
        print "************************** Distinct Products *******************************"
        dump(partner, masterOs, distinctVersion, f)

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
