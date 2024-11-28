# !/usr/bin/env python3
# -*- coding: utf-8 -*-


"""Query the data from MongoDB and plot result figures 
    
    """

__author__ = 'xsfeng'

import sys

# Import the modules
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import pandas as pd
import pymongo as pm
import shutil
from datetime import datetime, timedelta, timezone

import seaborn as sns
# import plotly_express as px

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from pathlib import Path

import time

import traceback
import os

import requests
import json
from io import StringIO
import pandas as pd

# Feishu IO parameters
#bitable info
bitable_id = "Urq0blfQqa87Cssd5a5c4fD7nPg"

# https://ex5xn5y3x9.feishu.cn/base/DPdEb1dMza3Xuqs5A34cFkAQn0b?table=tbl5ehGeaXcVVrHi&view=vewq7mFMbz
probe_table_id =  "tblDfWSkwMVSe3t3"
probe_table_view = "vewq7mFMbz"
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

print(myclient)

# Channel map path
ChannelmapPath = Path(r"/home/stairmed/ProbeEvaluation/ChannelMap.xlsx")

ACTIVE_PERCENT= 0.2
# 单通道持续有信号占总记录时长的占比
ACTIVE_PERCENT_TIME  = 0.2
#        2. Initial functional channel 初始植入阻抗正常：50k~1M
INIT_IMP_MAXTH= 1e6
INIT_IMP_MINTH= 5e4
#        3. Functional channel：长期阻抗好的定义：50k~2M 

IMP_MAXTH= 2e6
IMP_MINTH= 5e4
DAY_LAST = 90
CHANNELNUM = 128
mydb = myclient['UFLA']
mycol = mydb['Recording']
Days2Update = 4

#API范例获取自建应用凭证tenant_access_token

def get_tenant_access_token(app_id,app_secret):

    url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
    payload = json.dumps({
        "app_id": app_id,
        "app_secret": app_secret
    })

    headers = {
    'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    json_Data=json.loads(response.text)
    # 打印返回体信息
    print(json_Data)
    tenant_access_token=json_Data['tenant_access_token']
    return tenant_access_token

AU = 'Bearer '+get_tenant_access_token(app_id,app_secret)

def upload_to_cloud(file_path,app_id):
  ## upload picture to cloud, return file token ## 仅限图片

  """upload files to clouds, return file token"""

  url = "https://open.feishu.cn/open-apis/drive/v1/medias/upload_all"
  from pathlib import Path
  f = Path(file_path)
  f_size = f.stat().st_size
  print(f_size)

  payload={'file_name': 'imp.jpg',
  'parent_type': 'bitable_image',
  'parent_node': app_id,
  'size': str(f_size)}
  files=[
    ('file',(file_path,open(file_path,'rb'),'application/json'))
  ]
  headers = {
    'Authorization': AU
  }

  response = requests.request("POST", url, headers=headers, data=payload, files=files)

  print(response.text)
  pic_token = response.json()['data']['file_token']
  print('pic_token')
  print(pic_token)
  return pic_token


def upload_to_cell(app_id,table_id,record_id,field_name,pic_token):
    url = f"https://open.feishu.cn/open-apis/bitable/v1/apps/{app_id}/tables/{table_id}/records/{record_id}"
    payload = json.dumps({
        "fields": {
            field_name: [
                {
                    "file_token": pic_token
                }
            ]
        }
    })


    headers = {
    'Content-Type': 'application/json',
    'Authorization': AU
    }

    response = requests.request("PUT", url, headers=headers, data=payload)
    print(response.text)
    
def upload_file(file_path,app_id,table_id,record_id,field_name):

    pic_token = upload_to_cloud(file_path=file_path,app_id=app_id)
    upload_to_cell(app_id,table_id,record_id,field_name,pic_token)

# find token
def get_probe_token(animal):
    url = f"https://open.feishu.cn/open-apis/bitable/v1/apps/{bitable_id}/tables/{probe_table_id}/records/search"
    payload = json.dumps({
    
    "field_names": [
        "电极编号","植入时间","Channel Map","FPC类型"
        
    ],
    "filter": {
            "conjunction": "and",
            "conditions": [
                {
                    "field_name": "电极编号",
                    "operator": "is",
                    "value": [
                        animal
                    ]
                }
            ]
        },
    
    "automatic_fields": False
    })


    headers = {
    'Content-Type': 'application/json',
    'Authorization': AU
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    print(response.text)
    dicts = response.json()['data']['items']
    token = dicts[0]["record_id"]
    return token

def get_meta_info(animal): 
    # 从飞书数据库中读出植入相关信息
    
    url = f"https://open.feishu.cn/open-apis/bitable/v1/apps/{bitable_id}/tables/{probe_table_id}/records/search"
    payload = json.dumps({
    
    "field_names": [
        "电极编号","动物类型","植入时间","FPC类型","Channel Map"
        
    ],
    "filter": {
            "conjunction": "and",
            "conditions": [
                {
                    "field_name": "电极编号",
                    "operator": "is",
                    "value": [
                        animal
                    ]
                }
            ]
        },
    
    "automatic_fields": False
    })


    headers = {
    'Content-Type': 'application/json',
    'Authorization': AU
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    print(response.text)
    dicts = response.json()['data']['items']
    # token = dicts[0]["record_id"]
    species = dicts[0]['fields']['动物类型']
    # dicts[0]['fields']['动物类型']['value'][0]
    implant_timestamp= dicts[0]['fields']['植入时间']
    tz_utc_8 = timezone(timedelta(hours=8))

    implant_time= datetime.fromtimestamp(implant_timestamp / 1000.0)

    implant_time.replace(tzinfo=tz_utc_8)
    print(implant_time)
    FPC_version = dicts[0]['fields']['FPC类型']['value'][0]
    Probe_version = dicts[0]['fields']['Channel Map']['value'][0]['text']
    return Probe_version,FPC_version,implant_time

def update_record(record_id, key, value,app_token,table_id):
    url = f"https://open.feishu.cn/open-apis/bitable/v1/apps/{app_token}/tables/{table_id}/records/{record_id}"
    payload = json.dumps({
        "fields": {
            key: value
        }
    })

    headers = {
    'Content-Type': 'application/json',
    'Authorization': AU
    }
    response = requests.request("PUT", url, headers=headers, data=payload)
    print(response.text)


#########
def get_channelmap(ChannelmapPath,CMname,RecordingSystem,FPC):
 

    ChannelMapDf = pd.read_excel(ChannelmapPath)

    column_name = [col for col in ChannelMapDf.columns if CMname in col]

    if column_name:
        # 按找到的列进行排序
        CMinfo = column_name[0]
        CMinfos = CMinfo.split('-')
        print(CMinfos)
        # probe info
        probe_dict = {

                        'Channel_number': int(CMinfos[1])*int(CMinfos[2])*int(CMinfos[3]),
                        'Lane_number': int(CMinfos[1])*int(CMinfos[2])
    }
    SHANK_NUM=int(CMinfos[1])
    LANE_NUM=int(CMinfos[1])*int(CMinfos[2])
    num_contact_per_column=int(CMinfos[3])
    CHANNEL_NUM = int(CMinfos[1])*int(CMinfos[2])*int(CMinfos[3])
        # 生成probe map

        ##连接device
        # 记录系统-FPC版本
    RF= RecordingSystem+'（'+FPC.split('-')[0]+'）'
    ## intan channel
    ch_map = ChannelMapDf.sort_values(by=column_name[0])[RF].to_numpy()[:probe_dict['Channel_number']]
    channelmap = ch_map.reshape((LANE_NUM,num_contact_per_column))
    
    # ch_map = ChannelMapDf.sort_values(by="焊盘")["CM1-2-1-64-65"].to_numpy()-1
    # pad_imp = np.array(imp)[ch_map].reshape((8,16)).T
    # hp = ChannelMapDf.sort_values(by="焊盘")["焊盘"].to_numpy().reshape((8,16)).T
    # print(hp)
    return channelmap, LANE_NUM, CHANNEL_NUM


def find_one_probe_recording(animal):
    """find all recording results for one electrode. return impedance magnitude, impedance phase, SNR, Amplitude, noise level"""
   
    Probe_version,FPC_version,implant_time = get_meta_info(animal)
    channelmap, LANE_NUM, CHANNEL_NUM = get_channelmap(ChannelmapPath,Probe_version,'Intan',FPC_version)
    myquery = {'Animal': animal}
    print(animal)

    feature  = {'_id': 0, 'Impedance_magnitude':1,'Recording_time':1,'With_spike_channel_number':1,
                'Functional_channel_number':1}
    mycol.find(myquery,feature).sort('Recording_time')
    multi_rec = []


    # layer_id = '01'
    myquery = {'Animal': animal}
    # print(layer_id)

    feature  = {'_id': 0, 'Impedance_magnitude':1,'Impedance_phase':1,'Recording_time':1,
                'Amplitude':1,'Noise_level':1,'Single_Unit_SNR':1,
                }


    for item in mycol.find(myquery,feature).sort('Recording_time'):
        print(item['Recording_time'])
        recording = {}
        for key, value in item.items():

            
            recording[key] = value

        multi_rec.append(recording)

    ITEM_NUM = len(multi_rec)
    time_list = []
    imp_A = []
    imp_P = []
    #     imp_A.append(imp_bf_release)
    #     imp_A.append(imp_af_release)
    amp = []
    noise = []
    
    snr = []
    

    for i in multi_rec:
        if np.array(i['Impedance_magnitude']).shape[0]==128:
            time_list.append(i['Recording_time'])
            imp_A.append(i['Impedance_magnitude'])
            imp_P.append(i['Impedance_phase'])
            amp.append(i['Amplitude'])
            noise.append(i['Noise_level'])
            snr.append(i['Single_Unit_SNR'])
    date_str = [t.strftime('%y%m%d')  for t in time_list]
    day_interval = np.array([(t-implant_time).days for t in time_list])
    
    imp_A = np.array(imp_A)
    imp_P = np.array(imp_P)
    amp = np.array(amp)
    noise = np.array(noise)
    print(str(imp_A.shape))
    imp_vld = np.where((imp_A>IMP_MAXTH)|(imp_A<IMP_MINTH),np.nan,imp_A)
    crossday_recording = {}
    crossday_recording['Impedance_magnitude'] = imp_A
    crossday_recording['Impedance_phase'] = imp_P
    crossday_recording['Amplitude'] = amp
    crossday_recording['SNR'] = snr
    crossday_recording['Noise_level'] = noise
    crossday_recording['Day_interval'] = day_interval
    crossday_recording['Date_string'] = date_str

    return channelmap, LANE_NUM, CHANNEL_NUM, crossday_recording


def calculate_valid_channel(crossday_recording,CHANNELNUM):

    ## init functional channel
    ini_imp = crossday_recording['Impedance_magnitude'][0,:]
    ini_vld_channel = np.where((ini_imp<INIT_IMP_MAXTH)&(ini_imp>INIT_IMP_MINTH))[0]
    print(ini_vld_channel)
    ini_func = {}
    
    ini_func['Impedance_magnitude'] = crossday_recording['Impedance_magnitude'][:,ini_vld_channel]
    ini_func['Impedance_phase'] = crossday_recording['Impedance_phase'][:,ini_vld_channel]
    ini_func['Amplitude'] = crossday_recording['Amplitude'][:,ini_vld_channel]
    ini_func['Noise_level'] = crossday_recording['Noise_level'][:,ini_vld_channel]
    ini_func['SNR'] = crossday_recording['SNR']
    ini_func['Day_interval'] = crossday_recording['Day_interval']
    active_ch_num = np.sum(~np.isnan(ini_func['Amplitude']),axis=1)
    functional_ch_num = np.sum(~np.isnan(ini_func['Impedance_magnitude']),axis=1)
    ini_func['Functional_channel_ratio'] = functional_ch_num/CHANNELNUM
    ini_func['Active_channel_ratio'] = active_ch_num/functional_ch_num

    ## functional channel for each recording
    each_func = {}
    imp_A = crossday_recording['Impedance_magnitude']
    each_func['Impedance_magnitude'] = np.where((imp_A>IMP_MAXTH)|(imp_A<IMP_MINTH),np.nan,imp_A) 
    each_func['Impedance_phase'] = np.where((imp_A>IMP_MAXTH)|(imp_A<IMP_MINTH),np.nan,crossday_recording['Impedance_phase'])
    each_func['Amplitude'] = np.where((imp_A>IMP_MAXTH)|(imp_A<IMP_MINTH),np.nan,crossday_recording['Amplitude'])
    each_func['Noise_level'] = np.where((imp_A>IMP_MAXTH)|(imp_A<IMP_MINTH),np.nan,crossday_recording['Noise_level'])
    each_func['SNR'] = crossday_recording['SNR']
    each_func['Day_interval'] = crossday_recording['Day_interval']
    active_ch_num = np.sum(~np.isnan(each_func['Amplitude']),axis=1)
    functional_ch_num = np.sum(~np.isnan(each_func['Impedance_magnitude']),axis=1)
    each_func['Functional_channel_ratio'] = functional_ch_num/CHANNELNUM
   
    each_func['Active_channel_ratio'] = active_ch_num/functional_ch_num
   

    return ini_func,each_func




def boxplot_for_array(ax,func,Ylabel,Title,val_min,val_max, islog=True):
    if isinstance(func[Title],list):
        imp = func[Title]
        ax.boxplot(func[Title],showmeans=False,tick_labels=func['Day_interval'])
        ax.plot(np.arange(len(imp))+1,[np.mean(func[Title][i]) for i in range(len(imp))])
    else:
        imp = func[Title]
        cleaned_data = [imp[i,~np.isnan(imp[i])] for i in range(imp.shape[0])]
        ax.boxplot(cleaned_data,tick_labels=func['Day_interval'],meanline=True)
        ax.plot(np.arange(imp.shape[0])+1,np.nanmean(imp,axis=1))
    
    if islog:
        ax.semilogy()
    ax.set_ylim([val_min,val_max])
    ax.set_xlabel('Implant time (days)')
    ax.set_ylabel(Ylabel)
    ax.set_title(f'{Title} box plot')
# plt.savefig(file_path)
# plt.close()

def lineplot(ax,func):
    func_ratio = func['Functional_channel_ratio']
    act_ratio = func['Active_channel_ratio']
    ax.plot(func['Day_interval'],act_ratio)
    ax.plot(func['Day_interval'],func_ratio)
    ax.set_ylim([0,1])
    ax.set_xlabel('Implant time (days)')
    ax.legend(['Active channel ratio','Functional channel ratio'])
    ax.set_title(f'Functional and active channel ratio')
    ax.grid()


def get_color(feature,val_min=40,val_max=200,is_log=0):
    # color map

    cmap = mpl.colormaps['autumn']
    #normalize to [0,1]
    if is_log: 
        log_feature = np.log(feature)            
        norm_feature = (log_feature - np.log(val_min))/(np.log(val_max) - np.log(val_min))
        norm = mpl.colors.LogNorm(vmin=val_min, vmax=val_max)
    else:
        norm_feature = (feature - val_min)/(val_max - val_min)
        #color bar
        norm = mpl.colors.Normalize(vmin=val_min, vmax=val_max)

    norm_feature = np.where(norm_feature>1,1,norm_feature)
    norm_feature = np.where(norm_feature<0,0,norm_feature)
    # norm_feature[norm_feature>1] = 1
    # norm_feature[norm_feature<0] = 0

    #create color map
    colorseq = cmap(norm_feature)
    return colorseq, cmap, norm

#### Plot#################
def draw_amp_imp(ax,func,val_min=40,val_max=200):
   
    amp = func['Amplitude']
    imp_A = func['Impedance_magnitude']
    day_interval = func['Day_interval']
    ## find active channels
    # days*channels
    # print(type(amp[-1,1]))
    # print((amp[-1,1]))
    active_percent = np.sum(~np.isnan(amp),axis = 0)/len(day_interval)
    # print(active_percent)

    imp_A_act = imp_A[:,active_percent>ACTIVE_PERCENT_TIME]
    amp_act = amp[:,active_percent>ACTIVE_PERCENT_TIME]
    # print(amp_act.shape)
    # plt.plot(imp_A_act)
    # plt.savefig(animal+'.jpg')
    # plt.close()

    # for every channel
  
    for i in range(imp_A_act.shape[1]):
        co, cmap, norm= get_color(amp_act[:,i],val_min=val_min,val_max=val_max,is_log=0)
       
        ax.scatter(day_interval,imp_A_act[:,i],c=co,alpha=0.4)
    ax.semilogy()
    ax.set_title('impedance-amplitude scatter plot')
    ax.set_xlabel('Implant time (days)')
    ax.set_ylabel('Impedance magnitude (Ohms)')
    
    
    ax.set_ylim((IMP_MINTH,IMP_MAXTH))
    
    
   

    clb = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),ax=ax,
                orientation='vertical',) 
   
    clb.ax.set_title('Amplitude (\u03BCV)')
    
def plot_result(axes,func,i):
    if i==0:
        boxplot_for_array(axes[i,0],func,'Impedance magnitude (Ohms)','Impedance_magnitude',1e4,1e7)
    else:
        boxplot_for_array(axes[i,0],func,'Impedance magnitude (Ohms)','Impedance_magnitude',IMP_MINTH,IMP_MAXTH)
    boxplot_for_array(axes[i,1],func,'Impedance phase (degree)','Impedance_phase',-90,0,False)
    boxplot_for_array(axes[i,2],func,'Amplitude (\u03BCV)','Amplitude',0,500,False)
    boxplot_for_array(axes[i,3],func,'SNR','SNR',0,50,False)
    boxplot_for_array(axes[i,4],func,'Noise level (\u03BCV)','Noise_level',0,50,False)
    try:
        draw_amp_imp(axes[i,5],func)
    except:
        pass
    lineplot(axes[i,6],func)
    
#############heat map plot##############
def heatmap_plot(channelmap, LANE_NUM, CHANNEL_NUM, crossday_recording):
    imp_A = crossday_recording['Impedance_magnitude']
    noise = crossday_recording['Noise_level']
    amp = crossday_recording['Amplitude']
    day_intervel = crossday_recording['Day_interval']

    days_num = len(day_intervel)
    fig, axes = plt.subplots(LANE_NUM,3,figsize=(days_num*2+20,CHANNEL_NUM*0.5))
                    
    for i in range(LANE_NUM):
        
    
        ax_I = axes[i,0]
    
            
        # impedance
        val_min=1e5
        val_max = 5e6
        lane=imp_A.T[channelmap[i,:].astype(int),:]
        norm = mpl.colors.LogNorm(vmin=val_min, vmax=val_max)
        sns.heatmap(lane, vmin= 0, vmax=1, norm = norm,cmap='RdYlBu_r', 
                    annot=True,xticklabels=day_intervel,ax=ax_I)
        ax_I.set_title('Impedance Magnitude (Ohms)',fontweight = 'bold', fontsize=20)
    
        intan_ch = [str(ch).zfill(3) for ch in channelmap[i]]
        

        ax_A = axes[i,1]
        ax_N = axes[i,2]
        #signal amp
        lane=amp.T[channelmap[i,:].astype(int),:]

        sns.heatmap(lane, vmin= 20, vmax=100, cmap='YlGn', xticklabels=day_intervel, 
                    annot=True,fmt = '.1f',ax=ax_A)
        ax_A.set_title('Spike Amplitude (\u03BCV)',fontweight = 'bold', fontsize=20)
        
    

    # noise
        
        #noise level
        lane=noise.T[channelmap[i,:].astype(int),:]

        sns.heatmap(lane, vmin= 3, vmax=10, cmap='Oranges', xticklabels=day_intervel, 
                    annot=True,fmt = '.1f',ax=ax_N)
        ax_N.set_title('Noise Level (\u03BCV)',fontweight = 'bold', fontsize=20)
    
    
        
        # 添加第二排刻度
        for axx in [ax_I,ax_A,ax_N]:
            axx.set_yticks(ticks=np.linspace(0.5,(CHANNEL_NUM/LANE_NUM)-0.5,int(CHANNEL_NUM/LANE_NUM)),
                            labels=intan_ch)
            
            axx.set_ylabel(f"Shank{i}, Intan Channel")
            ax2 = axx.twinx()
            ax2.set_yticks(ticks=np.linspace(0.5,(CHANNEL_NUM/LANE_NUM)-0.5,int(CHANNEL_NUM/LANE_NUM)),
                            labels=np.arange((i+1)*(CHANNEL_NUM/LANE_NUM),i*(CHANNEL_NUM/LANE_NUM),-1).astype(int),rotation=0)
            ax3 = axx.twiny()
            ax3.set_xticks(ticks=np.linspace(0.5,days_num-0.5,days_num),
                            labels=crossday_recording['Date_string'],rotation=40)
            ax3.xaxis.set_ticks_position('bottom')
            ax2.yaxis.set_ticks_position('right')
            ax2.set_ylabel(f"Shank{i}, Probe Channel",labelpad=7)
            axx.set_xlabel('Days after implantation',fontsize=20,labelpad=40)
            ax2.set_ylim([0,(CHANNEL_NUM/LANE_NUM)])
            ax3.set_xlim([0,len(day_intervel)])



    
    plt.tight_layout()
    plt.savefig('heatmap.jpg')
    plt.close()


## 找出所有没有最新测试时间的电极编号
def find_new_probe():

  url = f"https://open.feishu.cn/open-apis/bitable/v1/apps/{bitable_id}/tables/{probe_table_id}/records/search"
  payload = json.dumps({
    
    "field_names": [
      "电极编号"
    ],
    
    "filter": {
      "conjunction": "and",
      "conditions": [
        
        { "field_name": "电极编号",
          "operator": "isNotEmpty",
          "value":[]}
      ,{ "field_name": "最新测试时间",
          "operator": "isEmpty",
          "value":[]}
          ,{ "field_name": "测试状态",
          "operator": "is",
          "value":["测试中"]}
      ]},
    
    "automatic_fields": False
  }

  )

  headers = {
    'Content-Type': 'application/json',
    'Authorization': AU
  }

  response = requests.request("POST", url, headers=headers, data=payload)
  dicts = response.json()['data']['items']
  print(dicts)



  Probe_list= list(set([dicts[i]['fields']['电极编号'][0]['text'] for i in range(len(dicts))]))
  print(Probe_list)
  return Probe_list

def get_time_interval(test_timestamp):

  tz_utc_8 = timezone(timedelta(hours=8))

  test_time= datetime.fromtimestamp(test_timestamp / 1000.0)

  test_time.replace(tzinfo=tz_utc_8)

  return (datetime.now()-test_time).days
## 找出最新测试时间超过3天的电极编号
def find_probe_need_update():
  url = f"https://open.feishu.cn/open-apis/bitable/v1/apps/{bitable_id}/tables/{probe_table_id}/records/search"
  payload = json.dumps({
    
    "field_names": [
      "电极编号","最新测试时间"
    ],
    
    "filter": {
      "conjunction": "and",
      "conditions": [
        
        { "field_name": "电极编号",
          "operator": "isNotEmpty",
          "value":[]}
          ,{ "field_name": "最新测试时间",
          "operator": "isNotEmpty",
          "value":[]}
          ,{ "field_name": "测试状态",
          "operator": "is",
          "value":["测试中"]}
      
      ]},
    
    "automatic_fields": False
  }

  )

  headers = {
    'Content-Type': 'application/json',
    'Authorization': AU
  }

  response = requests.request("POST", url, headers=headers, data=payload)
  dicts = response.json()['data']['items']
  print(dicts)

  test_timestamp= dicts[0]['fields']['最新测试时间']
  tz_utc_8 = timezone(timedelta(hours=8))

  test_time= datetime.fromtimestamp(test_timestamp / 1000.0)

  test_time.replace(tzinfo=tz_utc_8)

  Probe_list= np.array(list([d['fields']['电极编号'][0]['text'] for d in dicts]))
  day_interval = np.array([get_time_interval(d['fields']['最新测试时间']) for d in dicts])
  test_timestamp= np.array([d['fields']['最新测试时间'] for d in dicts])
  ind = np.where(day_interval>Days2Update)[0]
  # print(Probe_list[ind])
  return Probe_list[ind].tolist(),test_timestamp[ind].tolist()

##

def find_single_probe(Electrode_id):
  url = f"https://open.feishu.cn/open-apis/bitable/v1/apps/{bitable_id}/tables/{probe_table_id}/records/search"
  payload = json.dumps({
    
    "field_names": [
      "电极编号","最新测试时间"
    ],
    
    "filter": {
      "conjunction": "and",
      "conditions": [
        
        { "field_name": "电极编号",
          "operator": "is",
          "value":[Electrode_id]}
         
        
      
      ]},
    
    "automatic_fields": False
  }

  )

  headers = {
    'Content-Type': 'application/json',
    'Authorization': AU
  }

  response = requests.request("POST", url, headers=headers, data=payload)
  dicts = response.json()['data']['items']
  print(dicts)

  test_timestamp= dicts[0]['fields']['最新测试时间']
  tz_utc_8 = timezone(timedelta(hours=8))

  test_time= datetime.fromtimestamp(test_timestamp / 1000.0)

  test_time.replace(tzinfo=tz_utc_8)

  Probe_list= np.array(list([d['fields']['电极编号'][0]['text'] for d in dicts]))
  day_interval = np.array([get_time_interval(d['fields']['最新测试时间']) for d in dicts])
  test_timestamp= np.array([d['fields']['最新测试时间'] for d in dicts])
  ind = np.where(day_interval>Days2Update)[0]
  # print(Probe_list[ind])
  return Probe_list[ind].tolist(),test_timestamp[ind].tolist()


def update_single_plot(animal):
    
    tz_utc_8 = timezone(timedelta(hours=8))
    channelmap, LANE_NUM, CHANNEL_NUM, crossday_recording= find_one_probe_recording(animal)
    
            
    recording_time = datetime.strptime(crossday_recording['Date_string'][-1], '%y%m%d')
    recording_time.replace(tzinfo=tz_utc_8)

    record_time_ts= int(recording_time.timestamp()*1000)
    ini_func,each_func = calculate_valid_channel(crossday_recording,CHANNEL_NUM)
    
    days = len(crossday_recording['Day_interval'])
    fig, axes = plt.subplots(2,7,figsize=(30+days*2,10))
    fig.suptitle(animal)
    
    plot_result(axes,ini_func,0)
    plot_result(axes,each_func,1)
    fig.savefig('result.jpg')
    plt.tight_layout()
    plt.close()
    token = get_probe_token(animal)
    # print(token)
    upload_file('result.jpg',bitable_id,probe_table_id,token,'箱线折线图')
    heatmap_plot(channelmap, LANE_NUM, CHANNEL_NUM, crossday_recording)
    upload_file('heatmap.jpg',bitable_id,probe_table_id,token,'热图')
    update_record(token, "最新测试时间", record_time_ts,bitable_id,probe_table_id)

# def update_plot(Probe_list,test_timestamp,is_update_all=False):

# ## 对于没有最新日期的，test_timestamp输入[]
#     tz_utc_8 = timezone(timedelta(hours=8))  
#     for animal in Probe_list:
        


# ## find probe
# Probe_list2,test_timestamp2 = find_probe_need_update()
# Probe_list = find_new_probe()
print(f"请输入电极编号：")
Electrode_id = input()
Probe_list = [Electrode_id]
update_single_plot(Electrode_id)
# update_plot(Probe_list,[])
# update_plot(Probe_list2,test_timestamp2,True)
