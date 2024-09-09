from ultralytics import YOLO
import cv2
import pickle
import pandas as pd
class ballTracker:
    def __init__(self,model_path):
        self.model = YOLO(model_path)

    def interpolate_ball(self,ball_detections):
        ball_detections = [x.get(1,[]) for x in ball_detections] #list of lists

        df_ball_detections = pd.DataFrame(ball_detections,columns=["x1","y1","x2","y2"])

        df_ball_detections = df_ball_detections.interpolate()
        df_ball_detections = df_ball_detections.bfill() #to handle edges cases when no ball is detected in the first frame

        ball_detections = [{1:x} for x in df_ball_detections.to_numpy().tolist()]
        return ball_detections
    
    def get_frames_ball_hit(self,ball_pos):
        frames_to_check =25

        ball_pos = [x.get(1,[]) for x in ball_pos]
        df_ball_pos = pd.DataFrame(ball_pos,columns=["x1","y1","x2","y2"])
        df_ball_pos = df_ball_pos.interpolate()
        df_ball_pos = df_ball_pos.bfill()

        df_ball_pos["y_mid"] = (df_ball_pos["y1"] + df_ball_pos["y2"])/2    
        df_ball_pos["y_mid_rolling_mean"] = df_ball_pos["y_mid"].rolling(window=5,min_periods=1,center=False).mean()
        df_ball_pos["y_delta"] = df_ball_pos["y_mid_rolling_mean"].diff()
        df_ball_pos['ball_hit'] = 0
        for i in range(1,len(df_ball_pos)-int(1.2*frames_to_check)):    
            negative_change = df_ball_pos['y_delta'].iloc[i] > 0 and df_ball_pos['y_delta'].iloc[i+1] < 0 
            positive_change = df_ball_pos['y_delta'].iloc[i] < 0 and df_ball_pos['y_delta'].iloc[i+1] > 0

            if negative_change or positive_change:
                persistance = 0
                for frame in range(i,i+int(1.2*frames_to_check)):
                    negative_change_next = df_ball_pos['y_delta'].iloc[i] > 0 and df_ball_pos['y_delta'].iloc[frame+1] < 0 
                    positive_change_next = df_ball_pos['y_delta'].iloc[i] < 0 and df_ball_pos['y_delta'].iloc[frame+1] > 0

                    if negative_change and negative_change_next:
                        persistance +=1
                    elif positive_change and positive_change_next:
                        persistance +=1
            
                if persistance > frames_to_check-1:
                    df_ball_pos['ball_hit'].iloc[i] = 1
                    
        frames_ball_hit = df_ball_pos[df_ball_pos['ball_hit'] == 1].index.tolist()
        return frames_ball_hit
    
    def track_multiple_frames(self,frames,use_stubs=False,stub_path=None):
        balls = []

        if use_stubs == True:
            with open(stub_path,"rb") as f:
                balls = pickle.load(f)
            return balls
        
        for frame in frames:
            ball = self.track_frame(frame)
            balls.append(ball)

        if stub_path is not None:
            with open(stub_path,"wb") as f:
                pickle.dump(balls,f)
        return balls
    

    def track_frame(self, frame):
        output = self.model.predict(frame,conf=0.15)[0]  #[0] for models first output

        balls = {}
        for box in output.boxes:
            coods = box.xyxy.tolist()[0]
            balls[1] = coods

        return balls
    
    def draw_boxes_on_vid(self,frames,balls):
        output_frames = []
        for frame,ball in zip(frames,balls):
            for id,coods in ball.items():
                x1,y1,x2,y2 = coods
                
                cv2.putText(frame,f"Ball ID: {id}",(int(x1),int(y1-10)),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
                cv2.rectangle(frame,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),2)

            output_frames.append(frame)
        return output_frames


