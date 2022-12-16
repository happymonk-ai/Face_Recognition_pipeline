# gstreamer
import sys
from io import BytesIO
import os
from dotenv import load_dotenv
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GObject, GLib

#multi treading 
import asyncio
import nats
import os
import json
import numpy as np 
from PIL import Image
import cv2
import glob
from nanoid import generate
from multiprocessing import Process, Queue
import torch
import torchvision.transforms as T
from general import (check_requirements_pipeline)
import logging 
import threading
import gc
import fnmatch
import subprocess as sp
import time
import ipfsApi

# Detection
from track import run
from track import lmdb_known
from track import lmdb_unknown

path1 = "./Nats_output"

if os.path.exists(path1) is False:
    os.mkdir(path1)
    
# Multi-threading
TOLERANCE = 0.62
MODEL = 'cnn'
count_person =0
known_whitelist_faces = []
known_whitelist_id = []
known_blacklist_faces = []
known_blacklist_id = []
face_did_encoding_store = dict()
track_type = []
dict_frame = {}
frame = []
count_frame ={}
count = 0
processes = []
devicesUnique = []
activity_list = []
detect_count = []
person_count = []
vehicle_count = []
avg_Batchcount_person =[]
avg_Batchcount_vehicel = []
activity_list= []
geo_locations = []
track_person = []
track_vehicle = []
batch_person_id = []
timestamp = []
person_cid = []

iterator = 1

#ipfs
# api = ipfsApi.Client('216.48.181.154', 5001)


# gstreamer
# Initializes Gstreamer, it's variables, paths
Gst.init(sys.argv)
image_arr = None
device_types = ['', 'h.264', 'h.264', 'h.264', 'h.265', 'h.265', 'h.265']
load_dotenv()

       
async def json_publish(primary):    
    nc = await nats.connect(servers=["nats://216.48.181.154:5222"] , reconnect_time_wait= 50 ,allow_reconnect=True, connect_timeout=20, max_reconnect_attempts=60)
    js = nc.jetstream()
    JSONEncoder = json.dumps(primary)
    json_encoded = JSONEncoder.encode()
    Subject = "sample.activity_json"
    Stream_name = "Testing_activity"
    await js.add_stream(name= Stream_name, subjects=[Subject])
    ack = await js.publish(Subject, json_encoded)
    print(f'Ack: stream={ack.stream}, sequence={ack.seq}')
    print("Activity is getting published")

async def Video_creating(file_id, device_data):
    device_id = device_data[0]
    device_urn = device_data[1]
    print(device_id)
    print(device_urn)
    queue1 = Queue()
    queue2 = Queue()
    queue3 = Queue()
    queue4 = Queue()
    queue5 = Queue()
    queue6 = Queue()
    queue7 = Queue()
    global avg_Batchcount_person, avg_Batchcount_vehicel,track_person,track_vehicle,detect_count
    file_id_str = str(file_id)
    video_name1 = path1 + '/' + str(device_id) +'/Nats_video'+str(device_id)+'-'+file_id_str+'.mp4'
    print(video_name1)
    det = Process(target= run(video_name1, queue1, queue2, queue3, queue4, queue5, queue6, queue7))
    det.start()
    track_type = queue1.get()
    track_person = queue2.get()
    detect_count = queue3.get()
    batch_person_id = queue4.get()
    avg_Batchcount_person = queue5.get()
    timestamp = queue6.get()
    file_path = queue7.get()
    #ipfs
    # print("################################")
    # file_path = save_dir
    # print(file_path)
    for path, dirs, files in os.walk(os.path.abspath(file_path)):
        for filename in fnmatch.filter(files, "*.mp4"):
            src_file = os.path.join(file_path, filename)
            print(src_file)
            # res = api.add(src_file)
            # print(res)
            # res_cid = res[0]['Hash']
            # print(res_cid)
            command = 'ipfs --api=/ip4/216.48.181.154/tcp/5001 add {file_path} -Q'.format(file_path=src_file)
            output = sp.getoutput(command)
            person_cid = output
            # video_cid.append(res_cid)
    # print("################################")
    metapeople ={
                    "type":str(track_type),
                    "track":str(track_person),
                    "id":batch_person_id,
                    "activity":str("Null") ,
                    "detect_time":str(timestamp),
                    "cid": str(person_cid)
                    }
    
    metaVehicle = {
                    "type":str("Null"),
                    "track":str("Null"),
    }
    metaObj = {
                "people":metapeople,
                "vehicle":metaVehicle
            }
    
    metaBatch = {
        "Detect": str(detect_count),
        "Count": {"people_count":str(avg_Batchcount_person),
                    "vehicle_count":str("Null")} ,
                "Object":metaObj
    }
    
    primary = { "deviceid":str(device_urn),
                "batchid":str(file_id), 
                "timestamp":str("Null"), 
                "metaData": metaBatch}
    print(primary)
    Process(target= await json_publish(primary=primary)).start()
    # track_type.clear()
    # track_person.clear()
    # detect_count.clear()
    # batch_person_id.clear()
    # avg_Batchcount_person.clear()
    # timestamp.clear()
    # person_cid.clear()
    await asyncio.sleep(1)

async def gst_stream(device_id, urn, location, device_type):

    def format_location_callback(mux, file_id, data):
        device_data = []
        global iterator
        if(file_id == iterator):
            device_data.append(data)
            device_data.append(urn)
            asyncio.run(Video_creating((file_id-1), device_data))
            iterator += 1
    try:
        # rtspsrc location='rtsp://happymonk:admin123@streams.ckdr.co.in:3554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif' protocols="tcp" ! rtph264depay ! h264parse ! splitmuxsink location=file-%03d.mp4 max-size-time=60000000000
        # pipeline = Gst.parse_launch('filesrc location={location} name={device_id} ! decodebin name=decode-{device_id} ! videoconvert name=convert-{device_id} ! videoscale name=scale-{device_id} ! video/x-raw, format=GRAY8, width = 1080, height = 1080 ! appsink name=sink-{device_id}'.format(location=location, device_id=device_id))
        video_name = path1 + '/' + str(device_id)
        print(video_name)
        if not os.path.exists(video_name):
            os.makedirs(video_name, exist_ok=True)
        video_name = path1 + '/' + str(device_id) + '/Nats_video'+str(device_id)
        print(video_name)
        if(device_type == "h.264"):
            pipeline = Gst.parse_launch('rtspsrc location={location} protocols="tcp" name={device_id} ! rtph264depay name=depay-{device_id} ! h264parse name=parse-{device_id} ! splitmuxsink location={path}-%01d.mp4 max-size-time=10000000000 max-files=10 name=sink-{device_id}'.format(location=location, path=video_name, device_id = device_id))
        elif(device_type == "h.265"):
            pipeline = Gst.parse_launch('rtspsrc location={location} protocols="tcp" name={device_id} ! rtph265depay name=depay-{device_id} ! h265parse name=parse-{device_id} ! splitmuxsink location={path}-%01d.mp4 max-size-time=10000000000 max-files=10 name=sink-{device_id}'.format(location=location, path=video_name, device_id = device_id))

        sink = pipeline.get_by_name('sink-{device_id}'.format(device_id=device_id))

        if not pipeline:
            print("Not all elements could be created.")

        sink.connect("format-location", format_location_callback, device_id)
        
        # Start playing
        ret = pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            print("Unable to set the pipeline to the playing state.")

    except TypeError as e:
        print(TypeError," gstreamer streaming error >> ", e)

def on_message(bus: Gst.Bus, message: Gst.Message, loop: GLib.MainLoop):
    mtype = message.type
    """
        Gstreamer Message Types and how to parse
        https://lazka.github.io/pgi-docs/Gst-1.0/flags.html#Gst.MessageType
    """

    if mtype == Gst.MessageType.EOS:
        print("End of stream")
        loop.quit()

    elif mtype == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        print(("Error received from element %s: %s" % (
            message.src.get_name(), err)))
        print(("Debugging information: %s" % debug))
        loop.quit()

    elif mtype == Gst.MessageType.STATE_CHANGED:
        if isinstance(message.src, Gst.Pipeline):
            old_state, new_state, pending_state = message.parse_state_changed()
            print(("Pipeline state changed from %s to %s." %
            (old_state.value_nick, new_state.value_nick)))

    elif mtype == Gst.MessageType.ELEMENT:
        print(message.src)
    return True

async def main():
    
    await lmdb_known()
    await lmdb_unknown()
    
    pipeline = Gst.parse_launch('fakesrc ! queue ! fakesink')

    # Init GObject loop to handle Gstreamer Bus Events
    loop = GLib.MainLoop()

    bus = pipeline.get_bus()
    # allow bus to emit messages to main thread
    bus.add_signal_watch()

    # Add handler to specific signal
    bus.connect("message", on_message, loop)

    # Start pipeline
    pipeline.set_state(Gst.State.PLAYING)

    i = 3
    device_urn = 'uuid:eaadf637-a191-4ae7-8156-07433934718b'
    stream_url = os.getenv('RTSP_URL_{id}'.format(id=i))
    Process(target= await gst_stream(device_id=i, urn=device_urn, location=stream_url, device_type=device_types[i]))
    time.sleep(5)
    i = 6
    device_urn = 'uuid:3266ee49-9fc4-d257-0e8d-17a0469a9fc4'
    stream_url = os.getenv('RTSP_URL_{id}'.format(id=i))
    Process(target= await gst_stream(device_id=i, urn=device_urn, location=stream_url, device_type=device_types[i]))
    
    try:
        loop.run()
    except Exception:
        traceback.print_exc()
        loop.quit()

    # Stop Pipeline
    pipeline.set_state(Gst.State.NULL)
    del pipeline

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    try :
        loop.run_until_complete(main())
        loop.run_forever()
    except RuntimeError as e:
        print("error ", e)
        print(torch.cuda.memory_summary(device=None, abbreviated=False), "cuda")
        
"""
Json Object For a Batch Video 

JsonObjectBatch= {ID , TimeStamp , {Data} } 
Data = {
    "person" : [ Device Id , [Re-Id] , [Frame TimeStamp] , [Lat , Lon], [Person_count] ,[Activity] ]
}  


"""

"""
metapeople ={
                    "type":{" 00: known whitelist, 01: known blacklist, 10: unknown first time, 11: unknown repeat"},
                    "track":{" 0: tracking OFF, 1: tracking ON"},
                    "id":"face_id",
                    "activity":{"Null"}  
                    }
    
    metaVehicel = {
                    "type":{"Null"},
                    "track":{"Null"},
                    "id":"Null",
                    "activity":"Null"
    }
    metaObj = {
                 "people":metapeople,
                 "vehicle":"Null"
               }
    
    metaBatch = {
        "Detect": "0: detection NO, 1: detection YES",
        "Count": {"people_count":str(avg_Batchcount),
                  "vehicle_count":"Null" ,
        "Object":metaObj
        
    }
    
    primary = { "deviceid":str(Device_id),
                "batchid":str(BatchId), 
                "timestamp":str(frame_timestamp), 
                "geo":str(Geo_location),
                "metaData": metaBatch}
    print(primary)
    
"""
