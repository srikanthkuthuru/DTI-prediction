'''
1. This is a web scraping code
2. Collects the required kinase bioactivity data from drugKinet website. 
'''

import pandas as pd
a=pd.read_csv('Data/readinPy.csv')
from mechanize import Browser, HTTPError
import time


for i in range(0,636):
    
    br = Browser()
    kinase = a.siRNA[i]
    print i, kinase #Status Check
    br.open('http://www.drugkinet.ca/KinaseCompoundQuery.aspx')
    br.select_form(nr=0)
    br['ctl00$ContentPlaceHolder1$KinaseSpecTextbox'] = kinase
    item = br.find_control(name="ctl00$ContentPlaceHolder1$DataOutputDropdown", type="select").get("")
    item.selected = True
    
    try:
        br.submit()
    except HTTPError:
        print 'Server Error', i
        continue


    br.select_form(nr=0)
    
    try:
        br.form.find_control("ctl00$ContentPlaceHolder1$RegenTableButton")
    except:
        print 'not in database error', i
        continue
            
    response2 = br.submit(nr=0)
    fileobj = open('Kinases/%s.xls' %(kinase), 'w+')
    fileobj.write(response2.read())
    fileobj.close()
    #time.sleep(1)





    