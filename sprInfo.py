import struct
import sys
import base64
import numpy as np
import os
from pathlib import Path
# 判断系统是大端序还是小端序，0=小端序，1=大端序
byte_order = 0
if sys.byteorder == "big":
    byte_order = 1

_type_int8 = 1
_type_uint8 = _type_int8
_type_int16 = 2
_type_uint16 = _type_int16
_type_int32 = 4
_type_uint32 = _type_int32
_type_int64 = 8
_type_uint64 = _type_int64

_type_float = _type_int32
_type_double = _type_int64

class SprInfo:
    def __init__(self, file_name):
        self.fileName = file_name
        self.file = open(self.fileName, 'rb')
        self.headerSize = 0
        self.sampleDepth = 1024
        self.oneChannelCount = 128
        self._version = 0
        self._headFlag = ""
        self._portId = 0
        self._portChannelCount = 0
        self._startTime = 0
        self._sampleRate = 0
        self._sampleDepth = 0
        self._digitalReferenceType = 0
        self._digitalReferenceChannels = []
        self._k = 0
        self._b = 0
        self._notchP = []
        self._hfpPCount = 0
        self._hfpP = []
        self._lfpPCount = 0
        self._lfpP = []
        self._notchHz = 0
        self._highPassFilterCutoff = 0
        self._highPassFilterOrder = 0
        self._highPassFilterType = 0
        self._lowPassFilterCutoff = 0
        self._lowPassFilterOrder = 0
        self._lowPassFilterType = 0
        self._desText = ""
        self._desCount = 0
        self._channelMap = []
        self._dzMap = []
        self._packageNum = 0
        self._magicNumber = 0
        self._t = 0
        self._chunkSize = 0
        self._digitalIn1 = []
        self._digitalIn2 = []
        self._analogIn1 = []
        self._analogIn2 = []
        self._digitalOut1 = []
        self._digitalOut2 = []
        self._analogOut1 = []
        self._analogOut2 = []
        self._head64 = []
        self._rawData = []
        self._sequenceId = 0
        self._totalNum = 0

    def get_filename(self):
        return self.fileName

    def close_file(self):
        self.file.close()

    def get_header_size(self):
        return self.headerSize

    def get_version(self):
        return self._version

    def get_head_flag(self):
        return self._headFlag

    def get_sequence_id(self):
        return self._sequenceId

    def get_total_num(self):
        return self._totalNum

    def get_package_count(self):
        return self._packageNum

    def get_sample_rate(self):
        return self._sampleRate
    
    def read_header(self):
        try:
            if byte_order == 1:
                print(" byte_order = 1 ")
            else:
                byte = self.file.read(_type_uint8)
                self._version = int.from_bytes(byte, byteorder='little', signed=False)
                self.headerSize += _type_uint8
                # 标记
                byte = self.file.read(4)
                self._headFlag = byte.decode('utf-8')
                self.headerSize += 4
                # port口 ID
                byte = self.file.read(_type_uint8)
                self._portId = int.from_bytes(byte, byteorder='little', signed=False)
                self.headerSize += _type_uint8
                # 单port口通道数量
                byte = self.file.read(_type_uint16)
                self._portChannelCount = int.from_bytes(byte, byteorder='little', signed=False)
                self.headerSize += _type_uint16
                # 开始采样时间
                byte = self.file.read(_type_uint64)
                self._startTime = int.from_bytes(byte, byteorder='little', signed=False)
                self.headerSize += _type_uint64
                # 采样率
                byte = self.file.read(_type_double)
                self._sampleRate = struct.unpack('<d', byte)[0]
                self.headerSize += _type_double
                # 采样深度
                byte = self.file.read(_type_uint16)
                self._sampleDepth = int.from_bytes(byte, byteorder='little', signed=False)
                self.headerSize += _type_uint16
                # 数字滤波算法种类（中值 / 均值）
                byte = self.file.read(_type_uint8)
                self._digitalReferenceType = int.from_bytes(byte, byteorder='little', signed=False)
                self.headerSize += _type_uint8
                # 数字滤波使用的通道
                byte = self.file.read(self._portChannelCount)
                self._digitalReferenceChannels = np.frombuffer(byte, dtype=np.uint8)
                self.headerSize += self._portChannelCount
                # RAW k
                byte = self.file.read(_type_float)
                self._k = struct.unpack('<f', byte)[0]
                self.headerSize += _type_float
                # RAW b
                byte = self.file.read(_type_int32)
                self._b = struct.unpack('<i', byte)[0]
                self.headerSize += _type_int32
                # Notch
                byte = self.file.read(_type_double*6)
                self._notchP = np.frombuffer(byte, dtype=np.float64)
                self.headerSize += (_type_double*6)
                # HFP
                byte = self.file.read(1)
                self._hfpPCount = int.from_bytes(byte, byteorder='little', signed=False)
                self.headerSize += 1
                byte = self.file.read(_type_double * self._hfpPCount)
                self._hfpP = np.frombuffer(byte, dtype=np.float64)
                self.headerSize += (_type_double * self._hfpPCount)
                # LFP
                byte = self.file.read(1)
                self._lfpPCount = int.from_bytes(byte, byteorder='little', signed=False)
                self.headerSize += 1
                byte = self.file.read(_type_double * self._lfpPCount)
                self._lfpP = np.frombuffer(byte, dtype=np.float64)
                self.headerSize += (_type_double * self._lfpPCount)
                # NotchHz
                byte = self.file.read(_type_double)
                self._notchHz = struct.unpack('<d', byte)[0]
                self.headerSize += _type_double
                # HighPass
                byte = self.file.read(_type_int32)
                self._highPassFilterCutoff = int.from_bytes(byte, byteorder='little', signed=False)
                self.headerSize += _type_int32
                byte = self.file.read(_type_int32)
                self._highPassFilterOrder = int.from_bytes(byte, byteorder='little', signed=False)
                self.headerSize += _type_int32
                byte = self.file.read(_type_int32)
                self._highPassFilterType = int.from_bytes(byte, byteorder='little', signed=False)
                self.headerSize += _type_int32
                # LowPass
                byte = self.file.read(_type_int32)
                self._lowPassFilterCutoff = int.from_bytes(byte, byteorder='little', signed=False)
                self.headerSize += _type_int32
                byte = self.file.read(_type_int32)
                self._lowPassFilterOrder = int.from_bytes(byte, byteorder='little', signed=False)
                self.headerSize += _type_int32
                byte = self.file.read(_type_int32)
                self._lowPassFilterType = int.from_bytes(byte, byteorder='little', signed=False)
                self.headerSize += _type_int32
                # describe
                byte = self.file.read(_type_int16)
                self._desCount = int.from_bytes(byte, byteorder='little', signed=False)
                self.headerSize += _type_int16
                byte = self.file.read(self._desCount)
                self._desText = byte.decode('utf-8')
                self.headerSize += self._desCount
                # channel map
                byte = self.file.read(_type_uint16 * self._portChannelCount)
                self._channelMap = np.frombuffer(byte, dtype=np.uint16)
                self.headerSize += _type_uint16 * self._portChannelCount
                # dz(Ω)
                byte = self.file.read(_type_double * self._portChannelCount)
                self._dzMap = np.frombuffer(byte, dtype=np.float64)
                self.headerSize += _type_double * self._portChannelCount

        except IOError as e:
            print(f"Error reading file {self.fileName}: {e}")

    def calPackage_Num(self):
        try:
            size = os.path.getsize(self.fileName)
            onepackage_size = 16 + self._sampleDepth * 4 + self._sampleDepth * 2 * 4 + 64 + self._sampleDepth * self._portChannelCount * 2
            if onepackage_size > 0:
                self._packageNum = (size - self.headerSize) / onepackage_size
        except IOError as e:
            print(f"Error reading file {self.fileName}: {e}")

    def read_onepackage(self, offset):
        try:
            file_size = os.path.getsize(self.fileName)
            with open(self.fileName, 'rb') as f:
                fp = np.memmap(self.fileName, mode='r', dtype=np.uint8)
                # 计算本次读取的字节数（避免超出文件末尾）
                bytes_to_read = min(4, file_size - offset)
                byte_data = fp[offset:offset + bytes_to_read]
                self._magicNumber = struct.unpack('<I', byte_data.tobytes())[0]
                offset += bytes_to_read

                bytes_to_read = min(8, file_size - offset)
                byte_data= fp[offset:offset + bytes_to_read]
                self._t = struct.unpack('<q', byte_data.tobytes())[0]
                offset += bytes_to_read

                bytes_to_read = min(4, file_size - offset)
                byte_data = fp[offset:offset + bytes_to_read]
                self._chunkSize = struct.unpack('<I', byte_data.tobytes())[0]
                offset += bytes_to_read

                bytes_to_read = min(self.sampleDepth, file_size - offset)
                byte_data = fp[offset:offset + bytes_to_read]
                self._digitalIn1 = byte_data.view(dtype=np.int8)
                offset += bytes_to_read

                bytes_to_read = min(self.sampleDepth, file_size - offset)
                byte_data = fp[offset:offset + bytes_to_read]
                self._digitalIn2 = byte_data.view(dtype=np.int8)
                offset += bytes_to_read

                bytes_to_read = min(self.sampleDepth, file_size - offset)
                byte_data = fp[offset:offset + bytes_to_read]
                self._digitalOut1 = byte_data.view(dtype=np.int8)
                offset += bytes_to_read

                bytes_to_read = min(self.sampleDepth, file_size - offset)
                byte_data = fp[offset:offset + bytes_to_read]
                self._digitalOut2 = byte_data.view(dtype=np.int8)
                offset += bytes_to_read

                bytes_to_read = min(self.sampleDepth*_type_int16, file_size - offset)
                byte_data = fp[offset:offset + bytes_to_read]
                self._analogIn1 = byte_data.view(dtype=np.int16)
                offset += bytes_to_read

                bytes_to_read = min(self.sampleDepth*_type_int16, file_size - offset)
                byte_data = fp[offset:offset + bytes_to_read]
                self._analogIn2 = byte_data.view(dtype=np.int16)
                offset += bytes_to_read

                bytes_to_read = min(self.sampleDepth*_type_int16, file_size - offset)
                byte_data = fp[offset:offset + bytes_to_read]
                self._analogOut1 = byte_data.view(dtype=np.int16)
                offset += bytes_to_read

                bytes_to_read = min(self.sampleDepth*_type_int16, file_size - offset)
                byte_data = fp[offset:offset + bytes_to_read]
                self._analogOut2 = byte_data.view(dtype=np.int16)
                offset += bytes_to_read

                byte_data = fp[offset + 12:offset + 16]
                self._sequenceId = struct.unpack('<I', byte_data.tobytes())[0]

                bytes_to_read = min(64, file_size - offset)
                byte_data = fp[offset:offset + bytes_to_read]
                self._head64 = byte_data.view(dtype=np.int8)
                offset += bytes_to_read

                bytes_to_read = min(self.sampleDepth*self._portChannelCount*2, file_size - offset)
                byte_data = fp[offset:offset + bytes_to_read]
                self._rawData = byte_data.view(dtype=np.int8)
                offset += bytes_to_read

                self._totalNum = offset

        except IOError as e:
            print(f"Error reading file {self.fileName}: {e}")


    def save2bin(self,RawName = 'raw.bin'):
        # 打开一个二进制文件并写入 byte_data
        binfile = Path(self.fileName).parent/RawName 
        self.binfile = binfile
        with open(binfile, 'wb') as f:
            
            self.read_header()
            self.calPackage_Num()
            total = self.get_package_count()
            print(f"total = {total} ")
            

            # 使用 np.memmap 映射文件到内存
            # mem_map_obj = np.memmap(file, dtype=np.int16, mode='r', shape=matrix_size,order='F')
            ver = self.get_version()
            print( f"version = {ver} ")
            count = self.get_header_size()
            index = 0
            while index < total:
                self.read_onepackage(count)
                restored_data =( self._rawData.view(dtype=np.uint16).astype(np.float64)-32768.).astype(np.int16)
                restored_data.tofile(f)
                # mem_map_obj
                # a.append(my_spr._rawData)
                # sig = np.array(my_spr._rawData,dtype=np.float64).reshape((1024*2,128))-32768.
                # mem_map_obj
                count = self.get_total_num()
                sid = self.get_sequence_id()
                # print(f"sequence id = {sid}")
                index += 1

                self.close_file()

    def del_bin_file(self):
        
        try:
            self.binfile.unlink()  # 删除文件
            print(f"'{self.binfile}' has been deleted successfully.")
        except FileNotFoundError:
            print(f"The file '{self.binfile}' does not exist.")
        except Exception as e:
            print(f"Error occurred while deleting the file: {e}")