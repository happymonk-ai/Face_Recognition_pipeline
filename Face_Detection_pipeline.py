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

#Detection
from track import run


# face_detection
# import lmdb

path = "./Nats_output"

if os.path.exists(path) is False:
    os.mkdir(path)
    
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

queue1 = Queue()
queue2 = Queue()
queue3 = Queue()
queue4 = Queue()
queue5 = Queue()


# gstreamer
# Initializes Gstreamer, it's variables, paths
Gst.init(sys.argv)
image_arr = None
device_types = ['', 'h.264', 'h.264', 'h.264', 'h.265', 'h.265', 'h.264']
load_dotenv()

       
async def json_publish(primary):    
    nc = await nats.connect(servers=["nats://216.48.181.154:5222"] , reconnect_time_wait= 50 ,allow_reconnect=True, connect_timeout=20, max_reconnect_attempts=60)
    js = nc.jetstream()
    JSONEncoder = json.dumps(primary)
    json_encoded = JSONEncoder.encode()
    Subject = "model.activity_v1"
    Stream_name = "Testing_json"
    await js.add_stream(name= Stream_name, subjects=[Subject])
    ack = await js.publish(Subject, json_encoded)
    print(f'Ack: stream={ack.stream}, sequence={ack.seq}')
    print("Activity is getting published")

async def Video_creating(path, device_id):
    global avg_Batchcount_person, avg_Batchcount_vehicel,track_person,track_vehicle,detect_count
    image_folder = path
    video_name = path+'/Nats_video'+str(device_id)+'.mp4'
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpeg")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_name, fourcc , 1, (width,height))
    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))
    video.release()
    det = Process(target= run(source=video_name, queue1=queue1,queue2=queue2,queue3=queue3,queue4=queue4,queue5=queue5))
    det.start()
    avg_Batchcount_person = queue1.get()
    avg_Batchcount_vehicel = queue2.get()
    detect_count= queue3.get()
    track_person = queue4.get()
    track_vehicle = queue5.get()
    await asyncio.sleep(1)
    

async def batch_save(device_id,time_stamp):
    global avg_Batchcount_person ,avg_Batchcount_vehicel
    BatchId = generate(size= 8)
    count_batch = 0
    for msg in dict_frame[device_id]:
        try :
            im = np.ndarray(
                    (512,
                    512),
                    buffer=np.array(msg),
                    dtype=np.uint8) 
            im = Image.fromarray(im)
            device_path = os.path.join(path,str(device_id))
            if os.path.exists(device_path) is False:
                os.mkdir(device_path)
            im.save(device_path+"/"+str(count_batch)+".jpeg")
            await asyncio.sleep(1)
        except TypeError as e:
            print(TypeError," gstreamer error 64 >> ",e,"Device Id",device_id)
        count_batch += 1
    Process(target=await Video_creating(path=device_path, device_id=device_id)).start()
    await asyncio.sleep(1)
    if count_batch >= 10 :
        pattern = device_path+"/**/*.jpeg"
        for item in glob.iglob(pattern, recursive=True):
            os.remove(item) 
    metapeople ={
                    "type":str(track_type),
                    "track":str(track_person),
                    "id":batch_person_id,
                    "activity":{"Null"}  
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
    
    primary = { "deviceid":str(device_id),
                "batchid":str(BatchId), 
                "timestamp":str(time_stamp), 
                "metaData": metaBatch}
    print(primary)
    Process(target= await json_publish(primary=primary)).start()
    dict_frame[device_id].clear()
    count_frame[device_id] = 0 
    avg_Batchcount_person = []
    avg_Batchcount_vehicel =[]
    gc.collect()
    torch.cuda.empty_cache()

                
async def stream_thread(device_id , frame_byte,timestamp) :
    if len(dict_frame) == 0 :
        dict_frame[device_id] = list(frame_byte)
        count_frame[device_id] = 1 
    else:
        if device_id in list(dict_frame.keys()):
            dict_frame[device_id].append(list(frame_byte))
            count_frame[device_id] += 1
            if count_frame[device_id] % 10 == 0 :
                Process(target = await batch_save(device_id=device_id,time_stamp=timestamp)).start()
                await asyncio.sleep(1)
        else:
            dict_frame[device_id] = list(frame_byte)
            count_frame[device_id] = 1
    print(count_frame, "count frame ", threading.get_ident(),"Threading Id" ,device_id ,"Device id")
    await asyncio.sleep(1)


async def gst_data(device_id ,frame_byte, timestamp):
    global count 
    sem = asyncio.Semaphore(1)
    await sem.acquire()
    try:
        if device_id not in devicesUnique:
            t = Process(target= await stream_thread(device_id=device_id ,frame_byte=frame_byte, timestamp=timestamp))
            t.start()
            processes.append(t)
            devicesUnique.append(device_id)
        else:
            ind = devicesUnique.index(device_id)
            t = processes[ind]
            Process(name = t.name, target= await stream_thread(device_id=device_id ,frame_byte=frame_byte, timestamp=timestamp))
    
    except TypeError as e:
        print(TypeError," gstreamer error 121 >> ", e)
        
    finally:
        print("done with work ")
        sem.release()

    logging.basicConfig(filename="log_20.txt", level=logging.DEBUG)
    logging.debug("Debug logging test...")
    logging.info("Program is working as expected")
    logging.warning("Warning, the program may not function properly")
    logging.error("The program encountered an error")
    logging.critical("The program crashed")

async def gst_stream(device_id, location, device_type):
    def gst_to_opencv(sample):
        buf = sample.get_buffer()
        caps = sample.get_caps()
            
        arr = np.ndarray(
            (caps.get_structure(0).get_value('height'),
            caps.get_structure(0).get_value('width')),
            buffer=buf.extract_dup(0, buf.get_size()),
            dtype=np.uint8)
        return arr

    def new_buffer(sink, data):
        global image_arr
        sample = sink.emit("pull-sample")
        buffer = sample.get_buffer()
        timestamp = buffer.pts
        arr = gst_to_opencv(sample)
        resized = cv2.resize(arr, (512, 512))
        asyncio.run(gst_data(device_id=data ,frame_byte=resized, timestamp=timestamp))     
        return Gst.FlowReturn.OK

    try:
        # pipeline = Gst.parse_launch('filesrc location={location} name={device_id} ! decodebin name=decode-{device_id} ! videoconvert name=convert-{device_id} ! videoscale name=scale-{device_id} ! video/x-raw, format=GRAY8, width = 1024, height = 1024 ! appsink name=sink-{device_id}'.format(location=location, device_id=device_id))
        if(device_type == "h.264"):
            pipeline = Gst.parse_launch('rtspsrc location={location} name={device_id} ! queue max-size-buffers=2 ! rtph264depay name=depay-{device_id} ! h264parse name=parse-{device_id} ! decodebin name=decode-{device_id} ! videoconvert name=convert-{device_id} ! videoscale name=scale-{device_id} ! video/x-raw, format=GRAY8, width = 512, height = 512 ! appsink name=sink-{device_id}'.format(location=location, device_id=device_id))
        elif(device_type == "h.265"):
            pipeline = Gst.parse_launch('rtspsrc location={location} name={device_id} ! queue max-size-buffers=2 ! rtph265depay name=depay-{device_id} ! h265parse name=parse-{device_id} ! decodebin name=decode-{device_id} ! videoconvert name=convert-{device_id} ! videoscale name=scale-{device_id} ! video/x-raw, format=GRAY8, width = 512, height = 512 ! appsink name=sink-{device_id}'.format(location=location, device_id=device_id))

        sink = pipeline.get_by_name('sink-{device_id}'.format(device_id=device_id))

        if not pipeline:
            print("Not all elements could be created.")
        
        sink.set_property("emit-signals", True)
        sink.connect("new-sample", new_buffer, device_id)
        
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
    return True

async def main():
    
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

    for i in range(1, 7):
        stream_url = os.getenv('RTSP_URL_{id}'.format(id=i))
        t = Process(target= await gst_stream(device_id=i ,location=stream_url, device_type=device_types[i]))
        t.start()
    
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