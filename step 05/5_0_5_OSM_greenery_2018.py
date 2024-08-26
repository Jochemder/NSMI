# ---------------------------------------- File 5_0_1_OSM_greenery_2018 ---------------------------------------- #
# This piece of code imports all modules and libraries that are required for obtaining the greenery surface per neighborhood for 2018 via Open Street Map
import os
import time
import overpy
import geopandas as gpd
from geopandas import clip
import pandas as pd
from osgeo import ogr
from shapely.geometry import shape
import re
from shapely.geometry import Point
from shapely.geometry import Polygon
from shapely.geometry import MultiPolygon
import pyproj
from shapely.ops import transform
from shapely.ops import unary_union
import uuid
import numpy as np
import notebook
import overpass
from multiprocessing import  Pool
from functools import partial
import pandas as pd
import numpy as np
import logging

print('Modules imported')

# The user can choose from the following endpoints. The first one is the default endpoint which enables going back in time.
# In case the first overpass endpoint API does not react, another endpoint can be used. The user can choose from the following endpoints:
endpoint_overpassAPI = 'https://overpass.kumi.systems/api/interpreter'
# endpoint_overpassAPI = 'https://lz4.overpass-api.de/api/interpreter'
# endpoint_overpassAPI = 'https://z.overpass-api.de/api/interpreter'
# endpoint_overpassAPI = 'https://maps.mail.ru/osm/tools/overpass/api/interpreter'

#__ Functions for multithreading ______________________________________________________________________________________________________________________

def parallelize(data, func, num_of_processes=45):
    data_split = np.array_split(data, num_of_processes)
    pool = Pool(num_of_processes)
    data = pd.concat(pool.map(func, data_split))
    pool.close()
    pool.join()
    return data

def run_on_subset(func, data_subset):
    return data_subset.apply(func, axis=1)

def parallelize_on_rows(data, func, num_of_processes=45):
    return parallelize(data, partial(run_on_subset, func), num_of_processes)

#___ Funtions for obtaining greenery areas from the OSM server, clipping it to neighborhoods, and obtaining the total area_______________________________________________________________________________________________________


# Create a map if this does not yet exist.
def checkDir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

# Transform the boundingbox to 'EPSG:4326' coördinate-system. 
# This is demanded for using the Overpass API.
def extent(poly):
    RD = pyproj.CRS('EPSG:28992')
    Latlng = pyproj.CRS('EPSG:4326')
    project = pyproj.Transformer.from_crs(RD, Latlng, always_xy=True).transform
    poly = transform(project, poly)

    xmin, ymin, xmax, ymax = poly.bounds

    BB_Ext = f"{ymin},{xmin},{ymax},{xmax}"

    return BB_Ext


def explode_hstore(df, column):
    try:
        # split the tags column out as a new series, and break up each k=>v pair
        s = df[column].str.split(', ').apply(pd.Series, 1).stack()
        
        # extract the keys and values into columns
        extracted = s.str.extract(r'"(?P<key>[\w-]+)"=>"(?P<val>[\w-]+)"', re.IGNORECASE)
        
        # toss the unnecessary multi index that is generated in the process
        extracted.index = extracted.index.droplevel(1)
        
        # pivot the table make the rows in keys become columns
        pivoted = extracted.pivot_table(values='val', index=extracted.index, columns='key', aggfunc='first')
        
        # join with the original table and return
        return df.join(pivoted)
    except:
        return df
 
 
# This piece of code initializes a list tags with a single string 'landuse'. It sets tagvalue to np.nan (which stands for 'Not a Number', a special floating-point value in numpy).
# It enters a while loop that continues until stop is set to True.nside the loop, it iterates over each tag in the tags list. For each tag, it tries to access the value of that tag in the JSON object's 'properties'.
# If it succeeds, it sets tagvalue to that value and stop to True, which will end the loop. 
# If it fails to find the tag (which would raise a KeyError), it does nothing and tries the next tag. If it has tried all tags and none of them were found, it sets stop to True to end the loop.
# Finally, it returns tagvalue, which will be the value of the 'landuse' property if it was found, or np.nan if it wasn't.
def getType(json):
    tags = ['landuse']

    tagvalue = np.nan
    stop = False
    while stop == False:
        for tag in tags:
            try:
                tagvalue = json['properties'][tag]
                stop = True
            except:
                stop = False
        stop = True
    
    return tagvalue


# This piece of code takes the BB_Ext, which is the bounding box for the area of interest. It initializes the Overpass API with a specified timeout and endpoint.
# It constructs a multi-line string request that represents the Overpass QL (Query Language) to be sent to the Overpass API. 
# This query searches for all ways (linear features and area boundaries in OSM) with certain "landuse" tags within the specified bounding box.
# It sends the request to the Overpass API using the api.get() method, specifying that the response should be in XML format.
# The response from the API (the OSM data in XML format) is stored in the result variable.
# After this, the file location defined by loc_xml is opened in write mode, and the result is written to the file. IT also returns the file location.
def downloadXML(BB_Ext, loc_xml):
    api = overpass.API(timeout=120, endpoint=endpoint_overpassAPI, date="2018-12-31T23:59:59Z")
    request = f"""
        (
        way["landuse"="farm"]({BB_Ext});
        way["landuse"="farmland"]({BB_Ext});
        way["landuse"="farmyard"]({BB_Ext});
        way["landuse"="flowerbed"]({BB_Ext});
        way["landuse"="landfill"]({BB_Ext});
        way["landuse"="meadow"]({BB_Ext});
        way["landuse"="village_green"]({BB_Ext});
        way["landuse"="recreation_ground"]({BB_Ext});
        way["landuse"="greenfield"]({BB_Ext});
        way["landuse"="grass"]({BB_Ext});
        way["landuse"="forest"]({BB_Ext});
        way["landuse"="vineyard"]({BB_Ext});
        way["landuse"="orchard"]({BB_Ext});
        way["landuse"="aquaculture"]({BB_Ext});
        way["landuse"="allotments"]({BB_Ext});
        way["landuse"="cemetery"]({BB_Ext});
        way["landuse"="quarry"]({BB_Ext});
        );
        (._;>;);
        out ids geom;
        """
    result = None  # Initialize result to ensure it's defined even if the API call fails
    try:
        result = api.get(request, responseformat="xml")
    except Exception as e:
        logging.error(f"Failed to download XML data: {e}")
        # Handle the error appropriately. For example, you might want to:
        # - Return a default value
        # - Raise a custom exception
        # - Return None or a specific error code/message
        return None  # Or handle the error as appropriate for your application

    if result:
        with open(loc_xml, "w", encoding="utf-8") as f:
            f.write(result)
        return loc_xml
    else:
        # Handle the case where result is None (e.g., due to a failed API call)
        logging.error("No result to write to XML file.")
        return None  # Or handle as appropriate

# This piece of code takes the OSM data and extracts the multipolygons from it. It initializes an empty list data_list. It then iterates over each feature in the OSM data.
# For each feature, it exports the feature to a JSON object, extracts the coordinates of the feature, and converts them to a Shapely geometry object.
# It extracts the OSM way ID and the "other_tags" property from the JSON object, and calls the getType() function to extract the "landuse" tag from the JSON object.
# It appends a list containing the OSM way ID, the "other_tags" property, the Shapely geometry object, and the "landuse" tag to the data_list.
# After iterating over all features, it constructs a GeoDataFrame from the data_list, specifying the column names and the coordinate reference system (CRS) as "EPSG:4326".
# It then converts the GeoDataFrame to the "EPSG:28992" CRS and returns it.
def getPolyFeatures(data):
    data_list = []
    layer = data.GetLayer('multipolygons')
    features = [x for x in layer]

    for feature in features:
        json = feature.ExportToJson(as_object=True)
        coords = json['geometry']
        shapely_geo = shape(coords)
        osm_id = json['properties']['osm_way_id']
        other_tags = json['properties']['other_tags']
        greentype = getType(json)
        data_list.append([osm_id, other_tags, shapely_geo, greentype])

    gdf = gpd.GeoDataFrame(data_list,columns = ['osm_id','other_tags','geometry', 'type'], crs = "EPSG:4326")
    gdf = gdf.to_crs("EPSG:28992")
    return gdf

# This piece of code opens the OSM data from the file location defined by loc_xml using the OGR driver for OSM data. It calls the getPolyFeatures() function to extract the multipolygons from the OSM data.
# It dissolves the multipolygons to create a single polygon for each area of greenery, and then explodes the dissolved polygons to create a separate row for each polygon.
# It then calculates the area of each polygon and adds it as a new column to the GeoDataFrame. It resets the index of the GeoDataFrame and defines a list cols_keep containing the column names to keep.
# If the 'out_type' option in the input dictionary is 'gpkg', it saves the GeoDataFrame to a GeoPackage file in a temporary directory and returns the output path.
# If the 'out_type' option is 'gdf', it returns the GeoDataFrame directly.
def xml2gdf(loc_xml, input):
    driver = ogr.GetDriverByName('OSM')
    data = driver.Open(loc_xml)
    
    # Check if the data object is None
    if data is None:
        raise FileNotFoundError(f"Failed to open the file at {loc_xml}. Ensure the file exists and is accessible.")
    
    gdf_poly = getPolyFeatures(data)

    gdf = pd.concat([gdf_poly])
    
    # Remove invalid geometries before dissolve operation
    gdf = gdf[gdf['geometry'].is_valid]

    if input['dissolve']:
        gdf_knowntype = gdf[gdf['type'].notnull()]
        gdf_dissolved = gdf.dissolve().explode()
        gdf = gpd.sjoin(gdf_dissolved, gdf_knowntype, how="left")

    gdf['area'] = gdf['geometry'].area

    gdf = gdf.reset_index()

    cols_keep = ['geometry', 'type', 'area']
    cols_drop = set(gdf.columns.values.tolist()) - set(cols_keep)
    gdf = gdf.drop(cols_drop, axis=1)

    if input['out_type'] == 'gpkg':
        rel_path = 'OSM_greenery.gpkg'
        loc_OSM = os.path.join(temp_dir, rel_path)
        gdf.to_file(loc_OSM, layer='OSM_greenery', driver="GPKG", OVERWRITE="YES")
        return input['out_path']
    
    elif input['out_type'] == 'gdf':
        return gdf

# This Python function, downloadOSM(), takes 'input' as an argument, which contains various parameters including a directory path, an ID, a geometry type, and a polygon.
# It ensures the directory specified in input['temp_dir'] exists, then constructs a file path loc_xml in that directory with a filename based on the ID.
# If the geometry type is 'Polygon', it extracts the bounding box of the polygon and downloads OpenStreetMap data for that area, saving it to loc_xml.
# It then converts the downloaded data to a GeoPandas GeoDataFrame and clips it to the area of the neighborhood polygon.
# Finally, it returns the clipped GeoDataFrame. 
def downloadOSM(input):
    checkDir(input['temp_dir'])

    id = input['neighborhoodcode']
    rel_path = f'OSM_{id}.xml'
    loc_xml = os.path.join(input['temp_dir'], rel_path) 
    poly = input['polygon']
    
    if not os.path.exists(loc_xml):
        if input['geomtype'] == 'Polygon':
            BB_Ext = extent(poly)
            downloadXML(BB_Ext, loc_xml)

    gdf = xml2gdf(loc_xml, input)
        
    # Clean the polygon geometry
    poly = poly.buffer(0)

    # Clip the greenery areas based on the neighborhood's geometry
    gdf = clip(gdf, poly)
    return gdf

# This piece of code is used to calculate the total area of greenery for each neighborhood in the dataset which contains the geometries and information of all CBS 2018 neighborhoods.
# The sum_clipped_area() function takes a row from a GeoDataFrame, downloads OpenStreetMap data for the area defined by the 'geometry' field of the row, and calculates the total area of greenery in that area.
# The 'geometry' and 'neighborhoodcode' fields are extracted from the row and passed to the downloadOSM() function along with other parameters defined in the input dictionary.
# The downloadOSM() function downloads OpenStreetMap data for the specified area, processes the data, and returns it as a GeoPandas GeoDataFrame.
# The total area of greenery is calculated by summing the 'area' field of the returned GeoDataFrame, and this value is stored in the 'area_greenery' field of the row.
# The script then reads a GeoPackage file containing neighborhood data into a GeoDataFrame and applies the sum_clipped_area() function to each row of the GeoDataFrame in parallel.
# The resulting GeoDataFrame, which now includes the total area of greenery for each neighborhood, is saved to a new GeoPackage file. Duplicate areas are dealth with. 

def sum_clipped_area(row, temp_dir, out_path): 
    input = {
        'geomtype' : 'Polygon',                                     # The geometry type of the input is a polygon.
        'polygon' : row['geometry'],                                
        'print' : True,                                             # The geometry type of the input is a polygon.
        'temp_dir' : temp_dir,                                      # Location in which the temporary xml-files are saved.
        'dissolve' : False,                                         # dissolve is false, so geometries are not dissolved.
        'out_type' : 'gdf',                                         # The output is a geodataframe
        'out_path' : out_path,
        'neighborhoodcode' : row['neighborhoodcode'],               # "neighborhoodcode" which is the neighborhood code of the neighborhood for which the data is gathered.
        }
    gdf = downloadOSM(input)
    gdf['area'] = gdf['geometry'].area
    merged_geometry = unary_union(gdf['geometry'])
    row['area_greenery'] = merged_geometry.area
    print(merged_geometry.area)
    print(row['neighborhoodcode'])
    return row

if __name__ == '__main__':
    start_time = time.time()

    notebook_path = os.path.abspath("")
    script_dir = os.path.dirname(notebook_path)
    script_dir = notebook_path
    rel_path = 'intermediate/TEMP_greenery_2018'
    temp_dir = os.path.join(script_dir, rel_path)

    rel_path = 'greenery.gpkg'
    out_path = os.path.join(temp_dir, rel_path)

    source_neighborhoods = 'output/4_merge_SN_2018.gpkg'
    source_neighborhoods_read = gpd.read_file(source_neighborhoods)
    func = partial(sum_clipped_area, temp_dir=temp_dir, out_path=out_path)
    
    source_neighborhoods_read = parallelize_on_rows(source_neighborhoods_read, func)

    source_neighborhoods_read.to_file("output/5_0_1_OSM_greenery_2018.gpkg", driver="GPKG")
    
    # Show if the code has been executed successfully
    print(source_neighborhoods_read.sample(10))