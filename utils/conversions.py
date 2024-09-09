# import sys
# sys.path.append("../")
# import Dimensions

def pixels_to_meters(pixels,ref_pixels,ref_meters):
    return ((ref_meters * pixels)/ref_pixels)

def meters_to_pixels(meters,ref_pixels,ref_meters):
    return ((ref_pixels * meters)/ref_meters)