import torch
import torchvision.transforms as transforms
import torchvision.models as models
import cv2

class courtLineDetector:
    def __init__(self,model_path):
        self.model = models.resnet50(weights = None)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 2*14)
        self.model.load_state_dict(torch.load(model_path))
        self.transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),
        ])

    def predict_keypoints(self,frame):
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        frame_tensor = self.transforms(frame)
        frame_tensor = frame_tensor.unsqueeze(0) #we are only passing one frame of the video as the keypoints dont move

        with torch.no_grad():
            output = self.model(frame_tensor)

        keypoints = output.squeeze().numpy()   

        #we will have to rezise the keypoints to the original frame size
        og_h,og_w = frame.shape[:2]
        keypoints[::2] = keypoints[::2]*og_w/224.0    #all x coods
        keypoints[1::2] = keypoints[1::2]*og_h/224.0  #all y coods

        return keypoints
    
    def draw_keypoints_on_img(self,frame,keypoints):
        for i in range(0,len(keypoints),2):         # ,2 means it increments by 2 
            x,y = int(keypoints[i]),int(keypoints[i+1])

            cv2.putText(frame,f"{i//2}",(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),2)
            cv2.circle(frame,(x,y),5,(255,0,0),-1)
        return frame
    
    def draw_keypoints_on_vid(self,frames,keypoints):
        output_frames =[]
        for frame in frames:
            frame = self.draw_keypoints_on_img(frame,keypoints)
            output_frames.append(frame)
        return output_frames
                
