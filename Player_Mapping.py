import cv2, numpy as np
from collections import OrderedDict
from ultralytics import YOLO

class CentroidTracker:
    def __init__(self,max_disappeared=30):
        self.next_object_id=0
        self.objects=OrderedDict()
        self.disappeared=OrderedDict()
        self.max_disappeared=max_disappeared

    def register(self,centroid):
        self.objects[self.next_object_id]=centroid
        self.disappeared[self.next_object_id]=0
        self.next_object_id+=1

    def deregister(self,object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self,rects):
        if not rects:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id]+=1
                if self.disappeared[object_id]>self.max_disappeared:
                    self.deregister(object_id)
            return self.objects
        input_centroids=np.zeros((len(rects),2),dtype="int")
        for i,(x1,y1,x2,y2) in enumerate(rects):
            cX,cY=int((x1+x2)/2),int((y1+y2)/2)
            input_centroids[i]=(cX,cY)
        if not self.objects:
            for c in input_centroids:self.register(c)
        else:
            object_ids=list(self.objects.keys())
            object_centroids=list(self.objects.values())
            D=np.linalg.norm(np.array(object_centroids)[:,None]-input_centroids,axis=2)
            rows=D.min(1).argsort()
            cols=D.argmin(1)[rows]
            used_rows,used_cols=set(),set()
            for row,col in zip(rows,cols):
                if row in used_rows or col in used_cols:continue
                object_id=object_ids[row]
                self.objects[object_id]=input_centroids[col]
                self.disappeared[object_id]=0
                used_rows.add(row)
                used_cols.add(col)
            for row in set(range(D.shape[0]))-used_rows:
                object_id=object_ids[row]
                self.disappeared[object_id]+=1
                if self.disappeared[object_id]>self.max_disappeared:
                    self.deregister(object_id)
            for col in set(range(D.shape[1]))-used_cols:
                self.register(input_centroids[col])
        return self.objects

video_path="C:\\Users\\jayan\\Downloads\\15sec_input_720p.mp4"
model_path="C:\\Users\\jayan\\Downloads\\best.pt"
output_path="tracked_output.mp4"

model=YOLO(model_path)
tracker=CentroidTracker()
cap=cv2.VideoCapture(video_path)
fps=cap.get(cv2.CAP_PROP_FPS)
W,H=int(cap.get(3)),int(cap.get(4))
out=cv2.VideoWriter(output_path,cv2.VideoWriter_fourcc(*'mp4v'),fps,(W,H))

while True:
    ret,frame=cap.read()
    if not ret:break
    results=model(frame,verbose=False)[0]
    rects=[tuple(map(int,box)) for box,conf in zip(results.boxes.xyxy,results.boxes.conf) if conf>0.4]
    objects=tracker.update(rects)
    for object_id,(cX,cY) in objects.items():
        for x1,y1,x2,y2 in rects:
            if abs(cX-(x1+x2)//2)<15 and abs(cY-(y1+y2)//2)<15:
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.putText(frame,f"ID {object_id}",(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
    out.write(frame)

cap.release()
out.release()
print(f"✅ Saved to {output_path}")
