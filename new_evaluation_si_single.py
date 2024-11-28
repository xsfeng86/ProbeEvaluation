import psutil
import os
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import time
from pprint import pprint
import spikeinterface
import spikeinterface as si  # import core only
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
import spikeinterface.sorters as ss
import spikeinterface.postprocessing as spost
import spikeinterface.qualitymetrics as sqm
import spikeinterface.comparison as sc
import spikeinterface.exporters as sexp
import spikeinterface.curation as scur
import spikeinterface.widgets as sw
import spikeinterface.full as si
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
from probeinterface import generate_linear_probe
from probeinterface.plotting import plot_probe
import probeinterface
import pandas as pd
import shutil
import pymongo as pm
import json
import requests
import traceback
from utils import *
## global parameters
ChannelmapPath= Path('/home/stairmed/ProbeEvaluation/ChannelMap.xlsx')
SSDPath = Path('/opt/tmp')
## impedance threshold
br_th = 2e6
sh_th = 5e4
INTAN_CHANNELNUM = 128



# Feishu IO parameters
#bitable info

bitable_id = "Urq0blfQqa87Cssd5a5c4fD7nPg"
# https://ex5xn5y3x9.feishu.cn/base/Urq0blfQqa87Cssd5a5c4fD7nPg?table=tblDfWSkwMVSe3t3&view=vewqMsv0f3
# https://ex5xn5y3x9.feishu.cn/base/DPdEb1dMza3Xuqs5A34cFkAQn0b?table=tbl5ehGeaXcVVrHi&view=vewq7mFMbz
probe_table_id =  "tblDfWSkwMVSe3t3"
probe_table_view = "vewqMsv0f3"
raw_table_id = "tbllXg1OECPD8xnu"
wafer_table_id = "tblzjioDoqA84Hlv"
wafer_table_view = "vewNE2vl64"
# app info
app_id="cli_a6748743f93f100c"
app_secret="MMszmg8J6f9XZBxcU2F1XuqMyjRwHXEC"


# cinnected to mongodb
tz_utc_8 = timezone(timedelta(hours=8))
# connect to mongoDB server
# myclient = pm.MongoClient('mongodb://stairmed:123456@10.100.10.140:27080')
myclient = pm.MongoClient("mongodb://UFLA:REMBap2fWGbAyWMB@10.100.10.131:27017/?authMechanism=DEFAULT&authSource=UFLA")

# print(myclient)
mydb = myclient['UFLA']
mycol = mydb['Recording']   


# In[450]:
def main():

   
    print("请输入需要分析的日期文件夹名称（例：20241014）")
    
    datefolder = input()

    DateFolder = Path(f"/mnt/nfs/TestData/{datefolder}")
    print("请输入电极名称")
    ele_id = input()
    rhdFiles = list(DateFolder.rglob('*'+ele_id+'*.rhd*'))    
    rhdfile = Path(rhdFiles[0])
    RecordingPath =rhdfile.parent
    print(RecordingPath)
    print("请输入第几片电极")
    j = int(input())-1
   
    single_electrode_analysis(RecordingPath,j)

               

                

if __name__ == '__main__':
    main()

