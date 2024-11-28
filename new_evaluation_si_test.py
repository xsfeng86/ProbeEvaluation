

# In[429]:
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

   
    ## find rhd folders



# In[430]:


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
    # print(json_Data)
    tenant_access_token=json_Data['tenant_access_token']
    return tenant_access_token

 ## 飞书权限
AU = 'Bearer '+get_tenant_access_token(app_id,app_secret)

def upload_to_cloud(file_path,bitable_id):
  ## upload picture to cloud, return file token ## 仅限图片

  """upload files to clouds, return file token"""
  AU = 'Bearer '+get_tenant_access_token(app_id,app_secret)  
  url = "https://open.feishu.cn/open-apis/drive/v1/medias/upload_all"
  from pathlib import Path
  f = Path(file_path)
  f_size = f.stat().st_size
  print(f_size)

  payload={'file_name': 'imp.jpg',
  'parent_type': 'bitable_image',
  'parent_node': bitable_id,
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


def upload_to_cell(bitable_id,table_id,record_id,field_name,pic_token):

    AU = 'Bearer '+get_tenant_access_token(app_id,app_secret)
    url = f"https://open.feishu.cn/open-apis/bitable/v1/apps/{bitable_id}/tables/{table_id}/records/{record_id}"
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

def update_record(record_id, key, value,bitable_id,table_id):

    AU = 'Bearer '+get_tenant_access_token(app_id,app_secret)
    url = f"https://open.feishu.cn/open-apis/bitable/v1/apps/{bitable_id}/tables/{table_id}/records/{record_id}"
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
    # put_text(response.text) 
    
    
def upload_file(file_path,bitable_id,table_id,record_id,field_name):

    pic_token = upload_to_cloud(file_path=file_path,bitable_id=bitable_id)
    upload_to_cell(bitable_id,table_id,record_id,field_name,pic_token)

# update record


# find token
def get_probe_token(animal):

    AU = 'Bearer '+get_tenant_access_token(app_id,app_secret)
    url = f"https://open.feishu.cn/open-apis/bitable/v1/apps/{bitable_id}/tables/{probe_table_id}/records/search"
    payload = json.dumps({
    
    "field_names": [
        "电极编号",
        
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


# In[433]:


def update_record(record_id, key, value,app_token,table_id):

    AU = 'Bearer '+get_tenant_access_token(app_id,app_secret)
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
    # put_text(response.text)    


# In[434]:


def get_meta_info(animal): 
    # 从飞书数据库中读出植入相关信息
    
    AU = 'Bearer '+get_tenant_access_token(app_id,app_secret)
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
    return species, Probe_version,FPC_version


# In[435]:

def add_record2feishu(record_time,recordsystem,electrode_id):

    AU = 'Bearer '+get_tenant_access_token(app_id,app_secret)
    record_time_ts= int(record_time.timestamp()*1000)
    url = f"https://open.feishu.cn/open-apis/bitable/v1/apps/{bitable_id}/tables/{raw_table_id}/records/search"
    payload = json.dumps({
    
    "field_names": [
        "电极编号"
        
    ],
    "filter": {
            "conjunction": "and",
            "conditions": [
                {
                    "field_name": "电极编号",
                    "operator": "is",
                    "value": [
                        electrode_id
                    ]
                },
                {
                    "field_name": "测试时间",
                    "operator": "is",
                    "value": [
                        "ExactDate",
                        record_time_ts
                    ]
                },
            
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

    if len(dicts)==0:
    # 如果没有该记录，则创建一个
        url = f"https://open.feishu.cn/open-apis/bitable/v1/apps/{bitable_id}/tables/{raw_table_id}/records"
        payload = json.dumps({
            "fields": {
                "测试时间": record_time_ts,
                "测试系统": recordsystem,
                "电极编号": electrode_id
            }
        })


        headers = {
        'Content-Type': 'application/json',
        'Authorization': AU
        }

        response = requests.request("POST", url, headers=headers, data=payload)
        print(response.text)
        response_data = response.json()

        # token = response_data['data']
    
        token = response_data['data']['record']['record_id']
        print(token)
        return token
    else:
        token = dicts[0]["record_id"]
        print(f"{record_time} {electrode_id}的测试结果已存在！")
        return token

from datetime import datetime, timedelta, timezone

## 查询是否成功导入数据


def query_result(electrode_id, record_time):
    
    AU = 'Bearer '+get_tenant_access_token(app_id,app_secret)
    record_time_ts= int(record_time.timestamp()*1000)
    url = f"https://open.feishu.cn/open-apis/bitable/v1/apps/{bitable_id}/tables/{raw_table_id}/records/search"
    headers = {
        'Authorization': AU,
        'Content-Type': 'application/json'
    }
    payload = {
        "filter": {
            "conjunction": "and",
            "conditions": [
                {
                    "field_name": "电极编号",
                    "operator": "is",
                    "value": [electrode_id]
                },
                {
                    "field_name": "测试时间",
                    "operator": "is",
                    "value": [
                        "ExactDate",
                        record_time_ts
                    ]
                },
            ]
        }
    }

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code != 200:
        print(f"Error: {response.status_code}, {response.text}")
        # return False

    records = response.json().get('data', {}).get('items', [])

    if not records:
        # return False
        print('False')

    record = records[0]
    fields = record.get('fields', {})

    field_value = fields.get('是否成功导入数据库')

    print(field_value)
    return field_value

# def query_result(animal,record_time):
#     url = f"https://open.feishu.cn/open-apis/bitable/v1/apps/{bitable_id}/tables/{raw_table_id}/records/search"
#     record_time_ts= int(record_time.timestamp()*1000)
#     payload = json.dumps({

#     "field_names": [
#         "是否成功导入数据库",
        
#     ],
#     "filter": {
#             "conjunction": "and",
#             "conditions": [
#                 {
#                     "field_name": "电极编号",
#                     "operator": "is",
#                     "value": [
#                         animal
#                     ]
#                 },
#                 {
#                     "field_name": "测试时间",
#                     "operator": "is",
#                     "value": [
#                         "ExactDate",
#                         record_time_ts
#                     ]
#                 },
            
                
#             ]
#         },

#     "automatic_fields": False
#     })


#     headers = {
#     'Content-Type': 'application/json',
#     'Authorization': AU
#     }

#     response = requests.request("POST", url, headers=headers, data=payload)
#     print(response.text)
#     dicts = response.json()['data']['items']
#     # token = dicts[0]["record_id"]
#     Is_in_dataset = dicts[0]['fields']['是否成功导入数据库']
   

#     return Is_in_dataset
# In[435]:

########## 删除进程##################
        
def signal_handler(sig, frame):
    
    current_pid = os.getpid()
    # 遍历所有进程 
    for proc in psutil.process_iter(['pid', 'name', 'ppid']):
        # try:
        # 检查进程的父进程ID是否与当前PID相同
        if proc.info['ppid'] == current_pid:
            print(f'Terminating process: {proc.info["pid"]} ({proc.info["name"]})')
            proc.terminate()  # 尝试终止子进程
            
            # 获取子进程的 Process 对象
            process = psutil.Process(proc.info['pid'])
            
            # 非阻塞等待，最多等待3秒
            for _ in range(3):
                if not process.is_running():  # 检查子进程是否已结束
                    print("子进程已终止，资源已释放。")
                    break
                time.sleep(1)  # 每秒检查一次
            
            # 如果子进程仍在运行，使用 kill 强制终止
            if process.is_running():
                print(f"终止所有进程 {proc.info['pid']}.")
                process.kill()  # 强制结束子进程
            
            print("子进程处理完毕。")

        # except (psutil.NoSuchProcess, psutil.AccessDenied):
            # continue


# # Recording和Impedance相关

# In[436]:


# def get_meta_info(Electrode_id):
#     species = 'mice'
#     Probe_version = 'CM2'
#     FPC_version = '外插FPC-短'
#     return  species, Probe_version,FPC_version


# In[437]:


lab = 'stairmed'


# In[438]:


def get_recording(RecordingPath):

    rhdFiles = list(RecordingPath.rglob('*.rhd*'))
    if len(rhdFiles):
        print(rhdFiles)
        # recording = spikeinterface.extractors.read_binary(rhdFile, num_channels=128, 
        #                                        time_axis=0, file_offset=0, is_filtered=None)
        rhdFiles[0].stem
        list_of_recordings = []
        for file in rhdFiles:
            list_of_recordings.append(se.read_intan(file, stream_id='0',use_names_as_ids=True,all_annotations=True,ignore_integrity_checks = True))
        recording = si.concatenate_recordings(list_of_recordings)
        recording = spre.unsigned_to_signed(recording=recording)
        print(recording.annotate)
        RecordingSystem = 'Intan'
    else:
        binFiles = list(RecordingPath.rglob('*.bin*'))
        if len(binFiles):
            list_of_recordings = []
            for file in binFiles:
                

                recording_single = spikeinterface.extractors.read_binary(file, num_channels=128, 
                                                                sampling_frequency=30000, dtype=np.int16,
                                                time_axis=0, file_offset=0, is_filtered=None)
                list_of_recordings.append(recording_single)
                recording = si.concatenate_recordings(list_of_recordings)
                recording.set_channel_gains(1)
                recording.set_channel_offsets(0)

            print(recording.annotate)
            RecordingSystem = 'Stairplex'
        else:
            print("没有找到.rhd 或.bin文件，请检查")
    
        return recording,RecordingSystem



# In[439]:


def slice_recording(recording,e):
    ## channel id 换成单电极通道的形式，格式3位str
    channel_ids = recording.channel_ids
    INTAN_CHANNELNUM = 128
    all_channel_ids = [str(i) for i in range(INTAN_CHANNELNUM)]
    all_channel_ids = range(INTAN_CHANNELNUM)
    
    channel_number_ids = [int(channel_ids[i][2:] )for i in range(len(channel_ids))]
    channel_port = [channel_ids[i][0] for i in range(len(channel_ids))]

    port_name = list(set(channel_port))
    port_name.sort()
    channel_port.index(port_name[0])

    ## 对于每个port，channel start的index
    ch_start = [channel_port.index(port_name[j]) for j in range(len(port_name))]
    j = 1
    ch_start.append(-1)

    ## 分析第j片电极的数据

    channel_number_ids_part = channel_number_ids[ch_start[e]:ch_start[e+1]]
    channel_ids_part = channel_ids[ch_start[e]:ch_start[e+1]]
    recording_part = recording.channel_slice(channel_ids=channel_ids_part,renamed_channel_ids=channel_number_ids_part)
    WO_record_channel_ids = list(set(all_channel_ids)-set(channel_number_ids_part) )

    return recording_part, WO_record_channel_ids


# In[440]:
def generate_probe(ChannelmapPath,CMname,RecordingSystem,FPC,WO_record_channel_ids):


    # ch_map = pd.read_csv('/opt/ProbeEvaluation/app/FNEv1.4-128-4-1StairMed_Long_FPC_Intan.csv',header=None).to_numpy()
    # ch_map = ch_map.T.flatten()
    # 找到包含CMname的列
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
        
        # 生成probe map
        probe1 = probeinterface.generate_multi_shank(num_shank=int(CMinfos[1]), num_columns=int(CMinfos[2]), 
                                                        num_contact_per_column=int(CMinfos[3]),
                                                        shank_pitch=[1000,0],ypitch=-float(CMinfos[4]),
                                                contact_shapes='circle', contact_shape_params={'radius': 25})
        ##连接device
        # 记录系统-FPC版本
        RF= RecordingSystem+'（'+FPC.split('-')[0]+'）'
        # channel map,若为少于64通道去除无效通道
        ch_map = ChannelMapDf.sort_values(by=column_name[0])[RF].to_numpy()[:probe_dict['Channel_number']]
        
        probe1.set_device_channel_indices(ch_map)
        
        prb_df = probe1.to_dataframe(complete=True)
        #由于直接排列会改变通道编号，所以重新赋值
        prb_df['device_channel_indices']  = ch_map
        ##去掉recording里面没有的通道
        prb_df_vld = prb_df[prb_df.device_channel_indices.isin([int(c) for c in WO_record_channel_ids]) == False]
        vld_ch_map = prb_df_vld['device_channel_indices'].to_numpy()
        vld_ch_map = vld_ch_map.argsort().argsort()
        probe2 = probe1.from_dataframe(prb_df_vld)
        probe2.set_device_channel_indices(vld_ch_map)
    else:
        print(f"没有{CMname}对应的Channelmap！")

    return probe2,probe_dict

# In[461]:

def preprocessing(recording_part,bad_channel_ids,pre_path):
    duration = 600
    freq_min = 300
    freq_max = 6000
    recording_notch = spre.notch_filter(recording_part, freq=50, q=30, margin_ms=5.0)
    signal_dict = {'Band-pass_frequency_min': freq_min, 'Band-pass_frequency_max': freq_max,
                            'Sorting_method': 'mountainsort5'}
    recording_f = spre.bandpass_filter(recording=recording_notch, freq_min=freq_min, freq_max=freq_max,filter_order=20)
    
    noise_level = [np.nan] * INTAN_CHANNELNUM
    noise_levels_r = si.get_noise_levels(recording_f).tolist()
    
    noise_level = [np.nan] * INTAN_CHANNELNUM
    for i in range(recording_f.get_num_channels()):
        noise_level[recording_f.channel_ids[i]] = noise_levels_r[i]
    signal_dict['Noise_level']=noise_level
    ##
    fs = recording_part.get_sampling_frequency()
    signal_dict['Sampling_frequency'] = fs
    # take a maximum duration of 15min
    actual_duration = recording_f.get_total_duration()
    print(f"total duration: {actual_duration}")
    if actual_duration > duration:
       recording_f = recording_f.frame_slice(start_frame=0 * fs, end_frame=duration * fs)
       signal_dict['Duration'] = duration
    else:
       signal_dict['Duration'] = actual_duration
    
    recording_rm = recording_f.remove_channels(remove_channel_ids=bad_channel_ids)
    recording_cmr = spre.common_reference(recording=recording_rm, operator="median")
# if path already exists, delete it
    if pre_path.is_dir():
        shutil.rmtree(pre_path, ignore_errors=False, onerror=None)
      
    recording_cmr.save(folder=pre_path,format= "binary",progress_bar=True,overwrite=True,
                       n_jobs=-1,chunk_duration='1s',mp_context="spawn")
    recording_cmr = si.load_extractor(pre_path)
    return recording_cmr,signal_dict


# In[442]:


def slice_imp(IMPD,j):
        IMPD_magnitude = IMPD['Impedance Magnitude at 1000 Hz (ohms)'].to_numpy()[INTAN_CHANNELNUM*j:INTAN_CHANNELNUM*(j+1)]
        IMPD_phase = IMPD['Impedance Phase at 1000 Hz (degrees)'].to_numpy()[INTAN_CHANNELNUM*j:INTAN_CHANNELNUM*(j+1)]

        broken_ch_idx = np.where(IMPD_magnitude > br_th)[0]
        print(broken_ch_idx)
        short_ch_idx = np.where(IMPD_magnitude < sh_th)[0]
        functional_ch_idx = np.setdiff1d(np.arange(128), broken_ch_idx)
        functional_ch_idx = np.setdiff1d(functional_ch_idx, short_ch_idx)
        func_imp_mean = np.mean(IMPD_magnitude[functional_ch_idx])
        functional_ch_num = functional_ch_idx.size
        Functional = np.where(((IMPD_magnitude <= br_th) & (IMPD_magnitude >= sh_th)), True, False)
        IMPD_dict = {'Impedance_magnitude': IMPD_magnitude.tolist(),
                            'Impedance_phase': IMPD_phase.tolist(),
                            'Is_functional': Functional.tolist(),
                            'Broken_channels': broken_ch_idx.tolist(),
                            'Short-circuited_channels': short_ch_idx.tolist(),
                            'Set_broken_thresholds': br_th,
                            'Set_short-circuited_thresholds': sh_th,
                            'Functional_channel_number': functional_ch_num,
                            'Mean_impedance_magnitude_of_functional_channels': func_imp_mean

                            }
        return IMPD_dict


# In[ ]:





# In[443]:


## 焊盘图

def plot_pad(ChannelmapPath,imp,RecordingSystem,FPC,pad_path):
    ## 焊盘目前全部按照128通道出图
    ChannelMapDf = pd.read_excel(ChannelmapPath)
    ## 记录系统通道编号按照焊盘顺序排序
    RF= RecordingSystem+'（'+FPC.split('-')[0]+'）'
    ch_map = ChannelMapDf.sort_values(by="焊盘")[RF].to_numpy()
    pad_imp = np.array(imp)[ch_map].reshape((8,16)).T
    # hp = ChannelMapDf.sort_values(by="焊盘")["焊盘"].to_numpy().reshape((8,16)).T
    # print(hp)
   
    plt.figure(figsize=(10,15))
    norm = mpl.colors.LogNorm(vmin=1e5, vmax=3e6)

    sns.heatmap(pad_imp,norm=norm, cmap='RdYlBu_r',
                annot=True)
    
    plt.title(f'Pad map')
    plt.savefig(pad_path)

                            
                          




# In[459]:


def plot_subplots(pic_path, signal_dict):
    # 每行小图个数
    num_images = signal_dict['Single_Unit_number']
    images_per_row = 5
    fs = signal_dict['Sampling_frequency']
    sample_point = fs/1000
    t = (np.arange(3*sample_point)-sample_point)/sample_point
    # 计算行数
    num_rows = (num_images + images_per_row - 1) // images_per_row
    # 设置图形大小
    figsize = (images_per_row * 3, num_rows * 3)  # 每个小图3x3单位

    # 创建图形和子图
    fig, axes = plt.subplots(num_rows, images_per_row, figsize=figsize)

    # 如果只有一行，则将 axes 转换为一维数组
    if num_rows == 1:
        axes = np.array([axes])  # 转换为 2D 数组形式

    # 填充子图
    for i in range(num_images):
        ax = axes[i // images_per_row, i % images_per_row]
        ax.plot(t,signal_dict['Template'][i])  
        fig_title = 'Cluster' + str(i) + ' Channel' + str(signal_dict['Single_Unit_extreme_channel'][i])
        ax.set_title(fig_title)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Amplitude (\u03BCV)')
  

    # 关闭多余的子图
    for j in range(num_images, num_rows * images_per_row):
        axes[j // images_per_row, j % images_per_row].axis('off')

    plt.tight_layout()
    plt.savefig(pic_path)

    plt.close()

# 示例调用
 # 输入小图个数


# In[445]:


def save(index_dict,dict_tmp, mycol,update = True):
        """save the data to mongoDB
        """
        
        dict_to_save = {}
        for key, value in dict_tmp.items():
            
            if isinstance(value, (np.int64, np.int32,np.int16,np.float32)):
                dict_to_save[key] = value.item()
                print(key)
            else:
                dict_to_save[key] = value
        #print(dict_to_save)
        if update:
            index = index_dict
            
            mycol.update_one(index,{'$set':dict_to_save},upsert=True)
        else:
            mycol.insert_one(dict_to_save)


# In[446]:


def get_valid_unit_id(analyzer,sparsity):
    # calculate template metrics
    # tm = spost.compute_template_metrics(analyzer,load_if_exists=False)
    analyzer.compute(["random_spikes", "waveforms", "templates", "template_metrics","noise_levels","spike_amplitudes",'spike_locations'])
    # calculate template metrics
    ext = analyzer.get_extension('template_metrics')

    tm = ext.get_data()

    # calculate spike metrics
    metrics = si.compute_quality_metrics(analyzer, metric_names=["snr", "isi_violation","amplitude_cutoff","presence_ratio"])
    print(metrics)

    metrics = pd.concat([metrics, tm],axis =1)
    metrics['snr_PV'] = (1-metrics['peak_trough_ratio'])*metrics['snr']/2.0
    snr_thresh = 6
    isi_viol_thresh = 0.1
    presence_ratio_thresh = 0.1
    half_width = 0.0015
    
    # keep_mask = (metrics["snr"] > snr_thresh) & (metrics["isi_violations_ratio"] < 0.1)&(metrics["half_width"] < 0.0015)&(metrics['presence_ratio']>0.1)
    # print(keep_mask)
    our_query = f"snr > {snr_thresh} & isi_violations_ratio < {isi_viol_thresh}   & half_width <{half_width}"
    keep_units = metrics.query(our_query)
    keep_unit_ids = keep_units.index.tolist()
        ##大于1/4通道的spike视为噪声
    spread_vld_unit_id = [id for id,value in sparsity.unit_id_to_channel_indices.items() if len(value)<INTAN_CHANNELNUM/4]
    vld_unit_id = list(set(keep_unit_ids)&set(spread_vld_unit_id))
    print(vld_unit_id)
    return vld_unit_id,metrics


# In[447]:


def get_sorting_dict(recording_cmr,analyzer,sparsity,metrics,vld_unit_id):

    ## SNR

    signal_dict = {}
    signal_dict['SNR_threshold_SU'] = 6
    SU_SNR = metrics.loc[vld_unit_id]['snr'].tolist()
    SU_NUM = len(vld_unit_id)
    ## get unit info
    unit_ids = list(sparsity.unit_ids) 
    ## extreme id 为intan的值，但是不是str (dict)
    extreme_ch_ids = si.get_template_extremum_channel(analyzer,peak_sign="both")
    ## extreme ind 为i总长度为valid channel个数的角标(方便template和amplitude处理) (dict)
    extreme_ch_ind = {key: list(recording_cmr.channel_ids).index(value) for key, value in extreme_ch_ids.items()}
    print(extreme_ch_ind)
    ## extreme amplitude (dict)
    extreme_amp = si.get_template_extremum_amplitude(analyzer,peak_sign="both")

    ## amp按照valid channel个数排序的矩阵，需要转化为intan
    amp = si.get_template_amplitudes(analyzer,peak_sign="both")

    ## templates 按照valid channel个数排序的矩阵，需要转化为intan，unit按照unit ind 需要转化为序号对应
    ext = analyzer.get_extension('templates')
    all_templates = ext.get_data()

    ## channel_indices为总长度为valid channel个数的角标，channel_ids为intan值，但不是str
    spk_channels = sparsity.unit_id_to_channel_ids
    spk_channels_ind = sparsity.unit_id_to_channel_indices




    #channel为intan命名，string格式
    ## SU对应最大波形的channel
    ch_SU = [str(value) for key, value in extreme_ch_ids.items() if key in vld_unit_id ]
    print(ch_SU)

    # SU最大channel的template
    templates = []
    for key in vld_unit_id:

        extreme_ch_ids[key]
        ch = extreme_ch_ind[key]
        unit_ind = unit_ids.index(key)
        templates.append(all_templates[unit_ind][:,ch].tolist())

    # SU最大channel对应的幅值
    SU_amplitude= [value for key, value in extreme_amp.items() if key in vld_unit_id ]

    # SU spread channel数量
    SU_spread = [len(value) for key, value in spk_channels.items() if key in vld_unit_id ]
    # SU spread的channel名称
    SU_spread_ch = []
    for i in vld_unit_id:
        spk_ch = spk_channels[i]
        spk_ch = [str(ch) for ch in spk_ch]
        SU_spread_ch.append(spk_ch)

    ## SU spread channel 对应的amplitude
    SU_spread_ch_amp = []
    for  key in vld_unit_id:
        
        amp_u = amp[key]
        amp_u = amp_u[amp_u!=0]
        SU_spread_ch_amp.append(amp_u.tolist())

    ### channel amplitude
    amplitude = [[] for i in range(INTAN_CHANNELNUM)]
    all_amp = []
    for  key in vld_unit_id:
        
        amp_u = amp[key]
        all_amp.append(amp_u)
    all_amp = np.array(all_amp)

    ## 找到每个通道最大amplitude
    amp_max = np.max(all_amp,axis=0)

    amplitude = np.zeros((128,))
    amplitude
    for i in range(amp_max.shape[0]):
        id = recording_cmr.channel_ids[i]
        amplitude[id] = amp_max[i]
    with_spike = np.where(amplitude == 0, False, True)
    with_spike_ch_num = sum(with_spike)

    tmp_dict = {'Template': templates, 'Single_Unit_amplitude': SU_amplitude, 'Single_Unit_SNR': SU_SNR,
                        'Single_Unit_spread': SU_spread, 'Single_Unit_number': SU_NUM,
                        'Single_Unit_spread_channel': SU_spread_ch, 
                        'Single_Unit_extreme_channel':ch_SU,
                        'Single_Unit_channel_amplitude': SU_spread_ch_amp,   
                        'Amplitude': amplitude.tolist(),
                        'With_spike_channel_ids': with_spike.tolist(), 'With_spike_channel_number': with_spike_ch_num}
    signal_dict.update(tmp_dict)
    return signal_dict


# In[448]:
## 和之前记录比较得出初始有效通道平均阻抗和阻抗下降比例
def find_other_record(animal,this_time):
    """find all recording results for one electrode. return impedance magnitude, impedance phase, SNR, Amplitude, noise level"""
   
   
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
    # day_interval = np.array([(t-implant_time).days for t in time_list])
    
    imp_A = np.array(imp_A)
    imp_P = np.array(imp_P)
    amp = np.array(amp)
    noise = np.array(noise)
    print(str(imp_A.shape))
    
    crossday_recording = {}
    crossday_recording['Impedance_magnitude'] = imp_A
   
 
    crossday_recording['Recording_time'] = time_list

    ind = crossday_recording['Recording_time'].index(this_time)
    ## init functional channel
    ini_imp = crossday_recording['Impedance_magnitude'][0,:]
    ini_vld_channel = np.where((ini_imp<br_th)&(ini_imp>sh_th))[0]
    print(ini_vld_channel)
    ini_func = {}

    ini_func['Impedance_magnitude'] = crossday_recording['Impedance_magnitude'][:,ini_vld_channel]
    ini_func_mean_imp = np.nanmean(ini_func['Impedance_magnitude'])
    if ind:
        
        numerator = ini_func['Impedance_magnitude'][ind,:]
        denominator = ini_func['Impedance_magnitude'][ind-1,:]

    # 使用 np.where() 来处理除法
        decrease_rate_for_each_ch = np.where(np.isnan(denominator), np.nan, numerator / denominator)
    ##阻抗明显下降（30%）的电极个数
        decrease_count = float(np.sum(decrease_rate_for_each_ch < 0.7))
    else:
        decrease_count = float(np.nan)
    
    return decrease_count,  ini_func_mean_imp


## 向飞书在体测试原始数据表格中上传结果
def update_feishu_raw_table(dict_tmp,decrease_count,ini_func_mean_imp,RecordingPath,token):
    imp = dict_tmp['Impedance_magnitude']
    print(imp)
    mask = dict_tmp['With_spike_channel_ids']
    imp_with_spk = [value for value, m in zip(imp, mask) if m]
    imp_with_spk_mean = np.mean(imp_with_spk)
    feishu_dict = {
                    "Active Channel数量":int(dict_tmp['With_spike_channel_number']),
                "Active Channel阻抗均值":imp_with_spk_mean,
                "Functional Channel数量":dict_tmp['Functional_channel_number'],
                "Functional Channel阻抗均值（Ohms）":float(dict_tmp['Mean_impedance_magnitude_of_functional_channels']),
                "Active Channel幅值均值（微伏）":float(np.mean(dict_tmp['Single_Unit_amplitude'])),
                
                "Initial Functional Channel阻抗均值（Ohms）": float(ini_func_mean_imp),
                "是否成功导入数据库":"是",
                "原始文件地址":str(RecordingPath)}
    if ~np.isnan(decrease_count):

        feishu_dict['阻抗下降30%通道个数'] = decrease_count
        
    for key,value in feishu_dict.items():
        update_record(token, key, value,bitable_id,raw_table_id)


#####报错则返回unit为0的sorting结果#############
def set_unit_zero():
    signal_dict = {}
    signal_dict['SNR_threshold_SU'] = 6
    tmp_dict = {'Template': [], 'Single_Unit_amplitude': [], 'Single_Unit_SNR': [],
                        'Single_Unit_spread': [], 'Single_Unit_number': 0,
                        'Single_Unit_spread_channel': [], 
                        'Single_Unit_extreme_channel':[],
                        'Single_Unit_channel_amplitude': [],   
                        'Amplitude': [np.nan for i in range(INTAN_CHANNELNUM)],
                        'With_spike_channel_ids': [False for i in(range(INTAN_CHANNELNUM))],
                        'With_spike_channel_number': 0}
    signal_dict.update(tmp_dict)
    return signal_dict
# # Main




# In[460]:

### 读取单个recording文件每个电极的meta数据，包括.bin文件和RHD文件
def get_recording_meta(RecordingPath,RecordingSystem):
    
    if RecordingSystem =='Intan':
        rhdFiles = list(RecordingPath.rglob('*.rhd*'))
        
        print(rhdFiles)
        # recording = spikeinterface.extractors.read_binary(rhdFile, num_channels=128, 
        #                                        time_axis=0, file_offset=0, is_filtered=None)
    
    elif RecordingSystem =='Stairplex':
         rhdFiles = list(RecordingPath.rglob('*.bin*'))
    
    Electrodes,RecordingTime = rhdFiles[0].stem.split('_',1)
   
    # translate time into 
    tz_utc_8 = timezone(timedelta(hours=8))
    recording_time = datetime.strptime(RecordingTime, '%y%m%d_%H%M%S')
    recording_time.replace(tzinfo=tz_utc_8)
    
    ##Impedance
     # impedance file
    DateFolder = RecordingPath.parent
    ImpPath = DateFolder / ('imp'+Electrodes+'.csv')
    # 阻抗信息：
    # IMPD = pd.read_csv(ImpPath,encoding='gbk')
    IMPD = pd.read_csv(ImpPath,encoding='UTF-8')
    
    ## 从文件名判断有几个电极同时测试
    if Electrodes.count('&'):
        ElectrodesList = Electrodes.split('&')
    else:
        ElectrodesList = [Electrodes]
    return ElectrodesList,IMPD,recording_time

def sorting_curation(recording_cmr,RecordingPath,Electrode,signal_dict,token):
    sorting = ss.run_sorter('mountainsort5', recording_cmr, 
                                folder='/home/stairmed/ProbeEvaluation/results_MS5',detect_sign=0,
                                        remove_existing_folder=True,
                                        docker_image=False,singularity_image=False,
                                        n_jobs=24, chunk_duration="1s",#200ms",
                                        max_threads_per_process=16,mp_context="spawn"
                                    )
    sorting_path = RecordingPath/f"{Electrode}_sorting_result.json"
    sorting.dump(sorting_path)

    sorting = si.load_extractor(sorting_path)
    ### if sorted unit number is NOT zero
    if sorting.get_num_units():
        
        #
        #curation
        sparsity = si.estimate_sparsity(recording=recording_cmr,
                                    sorting=sorting,method='snr',peak_sign='both',
                                    threshold=4.5,noise_levels=si.get_noise_levels(recording_cmr),n_jobs=1)
        
        analyzer = si.create_sorting_analyzer(sorting=sorting, recording=recording_cmr, format="memory",sparsity=sparsity,
                                            n_jobs=24,total_memory='64G', chunk_duration="500ms",max_threads_per_process=12,
                                            mp_context="spawn")
        try:
            # valid unit id
            vld_unit_id,metrics = get_valid_unit_id(analyzer,sparsity)
            # sorting dict
            sorting_dict = get_sorting_dict(recording_cmr,analyzer,sparsity,metrics,vld_unit_id)
            pic_path = 'SU.jpg'
            s_dict = sorting_dict | signal_dict
            plot_subplots(pic_path,s_dict ) 
            upload_file(pic_path,bitable_id,raw_table_id,token,"Unit templates")
            
        except Exception as e:
                ##若报错则sorting结果置0，出错写到rhd文件夹里面（一般如果curation后为0会报错）
                sorting_dict = set_unit_zero()
                traceback.format_exc()
                

                with open(RecordingPath / 'dairy.txt', 'a') as f:
                    f.write(traceback.format_exc() + '\r\n')
                signal_handler(None, None)   
                
        
    else:   # ms5没有结果, try ms4
        with open(RecordingPath / 'dairy.txt', 'a') as f:
                    f.write("MS5未sorting出unit，改为ms4尝试" + '\r\n')
        sorting = ss.run_sorter('mountainsort4', recording_cmr, 
                        detect_sign=0,remove_existing_folder=True,
                        filter=False,
                                docker_image=False,singularity_image=False,
                                num_workers=24
                    
                            )
        if sorting.get_num_units() == 0:
            ###如果ms跑完仍为0，则认为unit个数为0
            sorting_dict = set_unit_zero()
        else:
            #curation
            sparsity = si.estimate_sparsity(recording=recording_cmr,
                                        sorting=sorting,method='snr',peak_sign='both',
                                        threshold=4.5,noise_levels=si.get_noise_levels(recording_cmr),n_jobs=1)
            
            analyzer = si.create_sorting_analyzer(sorting=sorting, recording=recording_cmr, format="memory",sparsity=sparsity,
                                                n_jobs=24,total_memory='64G', chunk_duration="500ms",max_threads_per_process=12,
                                                mp_context="spawn")
            try:
                # valid unit id
                vld_unit_id,metrics = get_valid_unit_id(analyzer,sparsity)
                # sorting dict
                sorting_dict = get_sorting_dict(recording_cmr,analyzer,sparsity,metrics,vld_unit_id)
                pic_path = 'SU.jpg'
                s_dict = sorting_dict | signal_dict
                plot_subplots(pic_path,s_dict ) 
                upload_file(pic_path,bitable_id,raw_table_id,token,"Unit templates")
                
            except Exception as e:
                    ##若报错则sorting结果置0，出错写到rhd文件夹里面（一般如果curation后为0会报错）
                    sorting_dict = set_unit_zero()
                    traceback.format_exc()
                    

                    with open(RecordingPath / 'dairy.txt', 'a') as f:
                        f.write(traceback.format_exc() + '\r\n')
                    signal_handler(None, None)

## 对于单个文件夹中的rhd文件：
def single_recording_file_analysis(RecordingPath):
    #在程序所在位置建立临时存储recording文件的位置，文件夹命名和rhd文件名相同
    # pre_path = SSDPath/(rhdFiles[0].stem)
    ## 节省空间改成相同的文件路径，不支持同时运行若干程序
    pre_path = SSDPath
    DateFolder = RecordingPath.parent
    
    ## 读取recording
    recording, RecordingSystem = get_recording(RecordingPath)
    
    ElectrodesList,IMPD,recording_time = get_recording_meta(RecordingPath,RecordingSystem)
    recording_date = recording_time.replace(hour=0,minute=0,second=0)

    
    for j in range(len(ElectrodesList)):
        Has_result = query_result(ElectrodesList[j],recording_time)
        if Has_result == '是':
            continue
        # get meta data
        species, Probe_version,FPC_version = get_meta_info(ElectrodesList[j])
        probe = Probe_version
        index_dict = {'Animal': ElectrodesList[j], 'Lab': lab, 
                            'Implant_batch':1,
                            'Recording_date':recording_date,
                            'Layer_id': ElectrodesList[j]}
        
        info_dict = {
                            'Animal_species': species,
                            'Recording_time': recording_time,
                            
                            'FPC_version': FPC_version,
                            'Probe_version': Probe_version,
                            'Recording_system': RecordingSystem
                            }
        ## 飞书新增记录，返回该记录token
        token = add_record2feishu(recording_time,RecordingSystem,ElectrodesList[j])
        
        IMPD_dict = slice_imp(IMPD,j)
        ## 上传焊盘图片
        pad_path =str(RecordingPath/'pad.jpg')
        plot_pad(ChannelmapPath,IMPD_dict['Impedance_magnitude'],RecordingSystem,FPC_version,pad_path)
    
        upload_file(pad_path,bitable_id,raw_table_id,token,"焊盘图")
        update_record(token, "是否成功导入数据库", "否",bitable_id,raw_table_id)
        if RecordingSystem == 'Intan':
            # 得到单片电极的recording，和关通道的channel id
            recording_part, WO_record_channel_ids = slice_recording(recording,j)
            #对于intan外的其他记录：
        else:
            recording_part, WO_record_channel_ids = recording, []
        
        # 考虑到关通道后的probe

        probe,probe_dict = generate_probe(ChannelmapPath,Probe_version,RecordingSystem,FPC_version,WO_record_channel_ids)
        ## channel id仍为recording中读取的id（有间隔）和device id(无间隔）不同
        recording_part =recording_part.set_probe(probe)
        bad_channel_ids = list(set(IMPD_dict['Broken_channels'])&set(recording_part.channel_ids))
        recording_cmr,signal_dict = preprocessing(recording_part,bad_channel_ids,pre_path)
        # sorting and curation
        sorting_dict = sorting_curation(recording_cmr,RecordingPath,ElectrodesList[j],signal_dict,token)
        #update mongodb
        dict_tmp = index_dict | info_dict | IMPD_dict | probe_dict | signal_dict |sorting_dict
        # delete preprocessing SSD
        save(index_dict,dict_tmp, mycol,update = True)
        shutil.rmtree(pre_path, ignore_errors=False, onerror=None)
        #和monogodb数据库中比较
        decrease_ratio,ini_func_mean_imp = find_other_record(index_dict['Animal'], dict_tmp['Recording_time'])
        
        # 上传飞书原始表格
        update_feishu_raw_table(dict_tmp,decrease_ratio,ini_func_mean_imp,RecordingPath,token)
        del recording_cmr


   
   
        
    


# In[450]:
def main():

    global_job_kwargs = dict(n_jobs=16,total_memory='64G', chunk_duration="500ms",max_threads_per_process=12,mp_context="spawn")
    # si.set_global_job_kwargs(**global_job_kwargs)
    try:
 

        print("请输入需要分析的日期文件夹名称（例：20241014）")
        
        datefolder = input()

        DateFolder = Path(f"/mnt/nfs/TestData/{datefolder}")
        
        # RecordingPath= Path(r"/mnt/nfs/TestData/20241013/TEST04&TEST06_241013_192159/")
        # ImpPath = Path(r"/mnt/nfs/TestData/20241013/impTEST04&TEST06.csv")
    
        folder_path = Path(DateFolder)

        # 列出所有下一级文件夹
        subfolders = [f for f in folder_path.iterdir() if f.is_dir()]
        RecordingSystemList = ['Intan','Stairplex']
        # 将所有文件夹的记录上传到原始数据表里
        for subfolder in subfolders:
            #对于不同的记录系统
            for RecordingSystem in RecordingSystemList:
                try:
                    ElectrodesList,_,recording_time = get_recording_meta(subfolder,RecordingSystem)
                    
                    for j in range(len(ElectrodesList)):
                        _ = add_record2feishu(recording_time,RecordingSystem,ElectrodesList[j])
                except:
                    pass
        print("开始进行数据分析...")
        for subfolder in subfolders:
            ##目前不需要用pad.jpg
            if 0:#(subfolder/'pad.jpg').exists():
                continue
            else:
                with open(DateFolder / 'dairy.txt',"a") as f:
                    f.write('Begin to analyse' + subfolder.stem + ': ')
                    localtime = time.asctime(time.localtime(time.time()))
                    f.write(localtime + '\r\n')
                try:
                    single_recording_file_analysis(subfolder)

                except Exception as e:
                    traceback.format_exc()
                    

                    with open(DateFolder / 'dairy.txt', 'a') as f:
                        f.write(traceback.format_exc() + '\r\n')
                signal_handler(None, None)
                print("所有子进程处理完毕，主程序继续运行。") 

    except KeyboardInterrupt:
        print("捕获到 SIGINT，正在终止子进程...")
        signal_handler(None, None)  #
   
        exit(0)
                

if __name__ == '__main__':
    main()

