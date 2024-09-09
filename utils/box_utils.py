def center_of_bbox(bbox):
    x1,y1,x2,y2 = bbox
    x = int((x1+x2)/2)
    y = int((y1+y2)/2)
    return (x,y)

def get_distance(p1,p2):
    x1,y1 = p1
    x2,y2 = p2
    return ((x1-x2)**2 + (y1-y2)**2)**0.5

def get_foot_bbox(bbox):
    x1,y1,x2,y2 = bbox
    return (int((x1+x2)/2),y2)

def get_height_bbox(bbox):
    x1,y1,x2,y2 = bbox
    return y2-y1

def xy_distance(p1,p2):
    x1,y1 = p1
    x2,y2 = p2
    return x1-x2,y1-y2

def get_closest_marker(foot_coods,court_markers,markers_to_check):
    min_distance = float('inf')
    closest_marker = markers_to_check[0]
    for i in markers_to_check:
        xy = court_markers[2*i],court_markers[2*i+1]
        distance = abs(foot_coods[1]-xy[1])
        if distance < min_distance:
            min_distance = distance
            closest_marker = i

    return closest_marker