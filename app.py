from fastapi import FastAPI
from pydantic import computed_field, BaseModel, Field
from typing import Literal, Annotated
import pickle
import pandas as pd
from fastapi.responses import JSONResponse
import math 

app= FastAPI()

with open("model_rfr.pkl","rb") as f:
    model= pickle.load(f)

brand_names=['Apple', 'Google', 'Honor',
       'Huawei', 'Infinix', 'Iqoo',
       'Itel', 'Motorola', 'Nokia',
       'Oneplus', 'Oppo', 'Poco',
       'Realme', 'Samsung', 'Tecno',
       'Vivo', 'Xiaomi']

brand_names1=['brand_name_apple', 'brand_name_google', 'brand_name_honor',
       'brand_name_huawei', 'brand_name_infinix', 'brand_name_iqoo',
       'brand_name_itel', 'brand_name_motorola', 'brand_name_nokia',
       'brand_name_oneplus', 'brand_name_oppo', 'brand_name_poco',
       'brand_name_realme', 'brand_name_samsung', 'brand_name_tecno',
       'brand_name_vivo', 'brand_name_xiaomi']

processor_brands=['Bionic', 'Dimensity',
       'Exynos', 'Helio',
       'Snapdragon', 'Tiger',
       'Unisoc']

processor_brands1=['processor_brand_bionic', 'processor_brand_dimensity',
       'processor_brand_exynos', 'processor_brand_helio',
       'processor_brand_snapdragon', 'processor_brand_tiger',
       'processor_brand_unisoc']

os_names=['Ios', 'Other']

os_names1=['os_ios', 'os_other']

class UserInput(BaseModel):
    brand_name: Annotated[str, Field(...,description="Select your brand")]
    os: Annotated[str,Field(...,description="Select Operating System")]
    processor: Annotated[str,Field(...,description="Select Processor")]
    processor_speed: Annotated[float, Field(...,gt=0,description="what's your processor speed in GHz")]
    battery_capacity: Annotated[int,Field(...,gt=0,description="Enter your battery capacity in mah")]
    num_cores: Annotated[int,Field(...,description="Enter number of cores")]
    num_front_cameras: Annotated[int, Field(...,gt=0,lt=5, description="Enter number of front cameras")]
    primary_camera_front: Annotated[int, Field(...,gt=0,description="Enter mega pixel")]
    has_5g: Annotated[str,Field(...,description="does your phone has 5g?")]
    has_nfc :Annotated[str,Field(...,description="does you phone has nfc ?")]
    has_ir_blaster: Annotated[str,Field(...,description="does your phone has IR Blaster ?")]
    fast_charging_available: Annotated[str,Field(...,description="does your phone support fast charging ?")]
    ram_capacity: Annotated[int,Field(...,gt=0,lt=50,description="Enter your phone Ram")]
    internal_memory: Annotated[int , Field(...,gt=0,lt=1024, description="What is your phone storage")] 
    screen_size: Annotated[float,Field(...,gt=0,description="Enter screen size in inch")]
    refresh_rate :Annotated[int,Field(...,gt=0,description="Enter your phone refresh_rate")]
    num_rear_cameras:Annotated[int,Field(...,gt=0,description="Enter number of rear cameras ")]
    primary_camera_rear :Annotated[int,Field(...,gt=0,description="Enter rear camera megapixel")]
    extended_memory_available :Annotated[str,Field(...,description="does your phone has extended_memory feature ")]
    resolution_width :Annotated[int,Field(...,gt=0,description="Enter your screen resolution width")]
    resolution_height :Annotated[int,Field(...,gt=0,description="Enter your screen resolution height")]


@app.post("/predict")
def predict(data:UserInput):
    brands=[]
    for brand in brand_names:
        if(brand== data.brand_name):
            brands.append(1)
        else:
            brands.append(0)
    
    processors=[]
    for proc in processor_brands :
        if(proc== data.processor):
            processors.append(1)
        else:
            processors.append(0)
    operating_sys=[]
    for operating in os_names:
        if(operating==data.os):
            operating_sys.append(1)
        else:
            operating_sys.append(0)
    complete = brands + operating_sys + processors
    complete1= brand_names1 + os_names1 + processor_brands1
    df1= pd.DataFrame([complete],columns=complete1)
    input = {
    "processor_speed": data.processor_speed,
    "battery_capacity": data.battery_capacity,
    "num_cores": data.num_cores,
    "num_front_cameras": data.num_front_cameras,
    "primary_camera_front": data.primary_camera_front,
    "has_5g": 1 if data.has_5g.lower() == "yes" else 0,
    "has_nfc": 1 if data.has_nfc.lower() == "yes" else 0,
    "has_ir_blaster": 1 if data.has_ir_blaster.lower() == "yes" else 0,
    "fast_charging_available": 1 if data.fast_charging_available.lower() == "yes" else 0,
    "ram_capacity": data.ram_capacity,
    "internal_memory": data.internal_memory,
    "screen_size": int(data.screen_size),
    "refresh_rate": data.refresh_rate,
    "num_rear_cameras": data.num_rear_cameras,
    "primary_camera_rear": data.primary_camera_rear,
    "extended_memory_available": 1 if data.extended_memory_available.lower() == "yes" else 0,
    "resolution_width": data.resolution_width,
    "resolution_height": data.resolution_height
    }

    df2= pd.DataFrame([input])

    final_df= pd.concat([df1,df2],axis=1)
    prediction = int(math.exp(model.predict(final_df)[0]))
    return JSONResponse(status_code=200, content={'predicted_value':prediction})









