#Expedia Divvy Challenge...
#Script file to load functions needed to execute on rows of dataframes
#Author: Ligaj Pradhan

import math

def getDistanceInKmFromLatitudeAndLongitude(x):
    #Mean Radius of the Earth...
    R = 6371.0 
    lat1 = math.radians(abs(x['latitude_x']))
    lon1 = math.radians(abs(x['longitude_x']))
    lat2 = math.radians(abs(x['latitude_y']))
    lon2 = math.radians(abs(x['longitude_y']))
    # We are computing the angle between then two stations...
    del_lon = lon2 - lon1 
    del_lat = lat2 - lat1
    # ‘haversine’ formula to calculate the great-circle distance between two points as described in 
    # https://www.movable-type.co.uk/scripts/latlong.html
    a = math.sin(del_lat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(del_lon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    # Finally we get distance in kilometers...
    distance = R * c
    return distance

###This function gets the ration of the total short and long trips
def getShortByLongRatio(x):
    long_count = x['LONG']
    short_count = x['SHORT']
    ratioSL = short_count/long_count
    return ratioSL



    
