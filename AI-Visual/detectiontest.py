import torch
import torch.nn as nn
import threading
from picamera2 import Picamera2
import cv2
import numpy as np
from collections import deque
import time
import csv
from itertools import groupby


#Parametrage pour tests
message_sequences = []
current_sequence = []
last_sample_time = 0
SAMPLE_INTERVAL = 0.1
SEQUENCE_LENGTH = 40
DISTANCE_LABELS = [str(x/1.0) for x in range(5,101)] # 5.0, 5.1, 5.2...... 100.0
label_index = 0
OUTPUT_CSV = "data_sample.csv"
recording_enabled = False

# Paramètres de calibration HSV
lower_bound = np.array([0, 0, 0])
upper_bound = np.array([180,255,255])
hsv_tolerance = np.array([20, 30, 30])
roi_size = 50
calibrated = False

display_enabled = False

led_states_buffer =deque(maxlen=2000)
symbol_decoded = []
frame_timestamps = deque(maxlen=60)

PREAMBLE_PATTERN=[(1,0),(0,1)]*4
SYMBOLS = {
    0: (0,0),
    1: (0,1),
    2: (1,0),
    3: (1,1),
}

#AI Params

aistart=False
endme=False
BUFFER_LEN = 2000
CAPTURE_LEN = 2000
symbol_buffer =deque(maxlen=BUFFER_LEN)
capturing = False
capture_buffer=[]
MODEL_PATH= "gru_bit_model.pt"

class GRUDecoder(nn.Module):
    def __init__(self,input_size=2,hidden_size=256,num_layers=2,output_size=16):
        super(GRUDecoder, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc= nn.Linear(hidden_size,output_size)
    def forward(self,x):
        _, h_n= self.gru(x)
        return self.fc(h_n[-1])

#Chargement du modele
model = GRUDecoder()
model.load_state_dict(torch.load(MODEL_PATH,map_location=torch.device('cpu')))
model.eval()

class LedTracker:
    def __init__(self,max_dist=50):
        self.prev_left = None
        self.prev_right = None
        self.max_dist = max_dist

    def assign_labels(self,leds):
        labeled=[]

        if len(leds) ==0:
            return labeled
        
        #Step 1 : Classification left or right

        leds = sorted(leds, key=lambda b:b[0])

        #step 2 : init

        if self.prev_left is None and len(leds) >=1:
            self.prev_left = self.center(leds[0])
            labeled.append(("g",leds[0]))
        if self.prev_right is None and len(leds) >1:
            self.prev_right = self.center(leds[1])
            labeled.append(("g",leds[1]))
        else:
            pass

        #step 3 : if both are on

        if len(leds)==2:
            c1=self.center(leds[0])
            c2=self.center(leds[1])
            if c1[0]<c2[0]:
                self.prev_left,self.prev_right = c1,c2
                labeled = [("g",leds[0]),("d",leds[1])]
            else:
                self.prev_left,self.prev_right = c2,c1
                labeled = [("g",leds[1]),("d",leds[0])]

        #step 4 : one led visible
        elif len(leds)==1:
            c=self.center(leds[0])
            if self.prev_left and self.distance(c, self.prev_left)<self.max_dist:
                self.prev_left=c
                labeled = [("g",leds[0])]
            elif self.prev_right and self.distance(c, self.prev_right)<self.max_dist:
                self.prev_right=c
                labeled = [("d",leds[0])]
            else:
                #based on position
                if c[0] < (self.prev_left[0]+self.prev_right[0])/2:
                    self.prev_left = c
                    labeled = [("g",leds[0])]
                else:
                    self.prev_right = c
                    labeled = [("d",leds[0])]
        return labeled
    
    @staticmethod
    def center(box):
        x,y,w,h=box
        return (x+w//2,y+h//2)
    
    @staticmethod
    def distance(a,b):
        return np.hypot(a[0]-b[0],a[1]-b[1])



# Black Area
class BlackRegionDetector:
    def __init__(self, min_area=400, smoothing=5):
        self.min_area = min_area
        self.rect_history = deque(maxlen=smoothing)

    def preprocess(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        return cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    def detect_largest_black_rectangle(self, image):
        mask = self.preprocess(image)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, mask
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < self.min_area:
            return None, mask
        x, y, w, h = cv2.boundingRect(largest)
        self.rect_history.append((x, y, w, h))
        xs, ys, ws, hs = zip(*self.rect_history)
        return (int(np.mean(xs)), int(np.mean(ys)), int(np.mean(ws)), int(np.mean(hs))), mask


def compress_symbol_stream(symbols, target_len=16):

    compressed= [key for key, _ in groupby(symbols)]

    if len(compressed) < target_len:
        compressed += [(0,0)*(target_len - len(compressed))]
    elif len(compressed) > target_len:
            compressed = compressed[:target_len]
    return compressed

def dynamic_frames_per_symbols():
    if len(frame_timestamps)<2:
        return 60
    durations = [t2-t1 for t1,t2 in zip(frame_timestamps, list(frame_timestamps)[1:])]
    avg_duration = sum(durations) / len(durations)
    fps=1.0/avg_duration if avg_duration >0 else 15
    return max(1,int(fps*0.2))

def run_gru_on_capture(symbols):
    print('decodace message')

    inputs=torch.tensor(symbols,dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        print(inputs)
        outputs = model(inputs)
        preds = (torch.sigmoid(outputs) > 0.5).int().squeeze().tolist()
        print(preds)
    if isinstance(preds[0],list):
        preds=[p[0] for p in preds]

    bytes_out= []
    for i in range(0, len(preds)-7,8):
        byte=0
        for b in range(8):
            byte |= preds[i+b] << (7-b)
        bytes_out.append(byte)
    try:
        msg=''.join(chr(b) for b in bytes_out if 32<= b <=126)
        print('Message reconstruit,', msg)
    except Exception as e:
        print('ca marche pas ptn',e )

def clean_and_merge_blocks(raw_blocks,min_length=20):
    filtered = [b for b in raw_blocks if b['length'] >= min_length]

    merged=[]
    for block in filtered:
        if not merged:
            merged.append(block)
        else:
            last=merged[-1]
            if last['state']==block['state']:
                last['length']+=block['length']
            else:
                merged.append(block)
    return merged


def try_decode_from_bufferAI(symbol):
    global capturing,capture_buffer,endme
    
    if not capturing:
        if symbol!=(0,0):
            capturing=True
            capture_buffer=[symbol]
            print('debut de capture 1 detecte')
    else:
        capture_buffer.append(symbol)

def ai_watcher():
    global endme, aistart
    while True:
        if aistart and endme:
            finalize_capture()
            endme=False
        time.sleep(0.05)


def finalize_capture():
    global capturing, capture_buffer
    print('fin de capture',capture_buffer)

    blocks =[]
    current_state = capture_buffer[0]
    count=1
    for s in capture_buffer:
        if s==current_state:
            count+=1
        else:
            blocks.append({'state':current_state,'length' : count})
            current_state=s
            count=1
    blocks.append({'state':current_state,'length' : count})

    blocks = clean_and_merge_blocks(blocks)

    if len(blocks) < 2:
        print("pas assez de blocks")
        capturing= False
        return

    threshold = blocks[0]['length']/8.0

    message_buffer = []
    for b in blocks[1:]:
        state = list(map(int,b['state']))
        l = b['length']
        if l< 5:
            continue
        repeat =1

        for n in range(1,8):
            if l<n*threshold:
                repeat=n
                break
        message_buffer.extend([state]*repeat)

        if len(message_buffer) >= 8 :
            break
    
    while len(message_buffer) < 8:
        message_buffer.append([0,0])
    
    print('message_buffer',message_buffer)
    run_gru_on_capture(message_buffer)

    capturing = False
    capture_buffer=[]
            

def try_decode_from_buffer():
    global led_states_buffer, symbol_decoded

    frames_per_symbol = dynamic_frames_per_symbols()
    total_needed = 8*frames_per_symbol
    if len(led_states_buffer) < total_needed:
        return None

    #Find preamble

    best_score = float('inf')
    best_start = None
    for i in range (len(led_states_buffer)-total_needed):
        score=0
        for j in range(8):
            symbol_frames = list(led_states_buffer)[i+j*frames_per_symbol: i+(j+1)*frames_per_symbol]
            expected = PREAMBLE_PATTERN[j]
            score+= sum(1 for f in symbol_frames if f!=expected)
        if score < best_score:
            best_score=score
            best_start= i
    if best_start is None or best_score > frames_per_symbol * 8 * 0.3:
        return None #pas trouve
    
    #decodage
    data_start = best_start + 8 *frames_per_symbol
    new_symbols = []
    i=data_start
    while i + frames_per_symbol <=len(led_states_buffer):
        group =list(led_states_buffer)[i:i + frames_per_symbol]
        errors= {k: sum(1 for g in group if g!= v) for k,v in SYMBOLS.items()}
        sym = min(errors, key=errors.get)
        new_symbols.append(sym)
        i+= frames_per_symbol
    
    #Conversion in bytes
    bytes_out= []
    for i in range(0,len(new_symbols)-3,4):
        byte = (new_symbols[i]<<6) | (new_symbols[i+1]<<4) | (new_symbols[i+2]<<2) | (new_symbols[i+3])
        bytes_out.append(byte)
    
    try:
        decoded =''.join(chr(b) for b in bytes_out)
        print("Message decode =",decoded)
        symbol_decoded=new_symbols
    except:
        print("donnees non valides")

def detect_leds_positions(mask):
    contours,_ =cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    boxes=[]
    for c in contours:
        hull=cv2.convexHull(c)
        area=cv2.contourArea(hull)
        if area>3:
            boxes.append(cv2.boundingRect(hull))
    return boxes

def detect_leds_hsv(image, lower, upper, roi_coords=None, max_leds=2, merge_dist_thresh=40):
    roi = image[roi_coords[1]:roi_coords[1]+roi_coords[3], roi_coords[0]:roi_coords[0]+roi_coords[2]] if roi_coords else image
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 2]
    boxes =[]
    cy_list = []

    for c in contours:
        hull=cv2.convexHull(c)
        area=cv2.contourArea(hull)
        if area>3:
            x_,y_,w_,h_=cv2.boundingRect(hull)
            cy=y_+h_//2
            boxes.append((x_,y_,w_,h_,cy))
            cy_list.append(cy)
    if not boxes:
        return image.copy(),mask
    cy_mean=np.mean(cy_list)
    y_tol=3
    filtered = [(x,y,w,h) for (x,y,w,h,cy) in boxes if abs(cy - cy_mean) < y_tol]

    merged = []
    for b in filtered:
        x, y, w, h = b
        cx1, cy1 = x + w // 2, y + h // 2
        for i, (mx, my, mw, mh) in enumerate(merged):
            cx2, cy2 = mx + mw // 2, my + mh // 2
            if np.hypot(cx1 - cx2, cy1 - cy2) < merge_dist_thresh:
                nx, ny = min(x, mx), min(y, my)
                nw = max(x + w, mx + mw) - nx
                nh = max(y + h, my + mh) - ny
                merged[i] = (nx, ny, nw, nh)
                break
        else:
            merged.append(b)
    merged = sorted(merged, key=lambda b: b[1])[:max_leds]
    out = image.copy()
    for (x, y, w, h) in merged:
        if roi_coords: x, y = x + roi_coords[0], y + roi_coords[1]
        cv2.rectangle(out, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.circle(out, (x + w // 2, y + h // 2), 3, (0, 255, 0), -1)
    full_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    if roi_coords:
        full_mask[roi_coords[1]:roi_coords[1]+roi_coords[3], roi_coords[0]:roi_coords[0]+roi_coords[2]] = mask
    else:
        full_mask = mask
    return out, full_mask

def mouse_callback(event, x, y, flags, param):
    global lower_bound, upper_bound, calibrated
    if event == cv2.EVENT_LBUTTONDOWN:
        frame = param["frame"]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        print(f"[🖱️] Clic à : ({x},{y})")

        # Zone autour du clic
        search_radius = 10
        x1 = max(0, x - search_radius)
        x2 = min(hsv.shape[1], x + search_radius)
        y1 = max(0, y - search_radius)
        y2 = min(hsv.shape[0], y + search_radius)

        zone = hsv[y1:y2, x1:x2]

        # Moyenne HSV des pixels assez saturés et pas trop sombres
        mask_valid = (zone[:, :, 1] > 50) & (zone[:, :, 2] > 50)
        valid_pixels = zone[mask_valid]

        if valid_pixels.shape[0] < 10:
            print("❌ Trop peu de pixels valides autour du clic")
            return

        mean_hsv = np.mean(valid_pixels, axis=0).astype(np.uint8)

        # Tolérance personnalisable
        tolerance = np.array([20, 60, 60])
        lower_bound = np.clip(mean_hsv - tolerance, 0, 255)
        upper_bound = np.clip(mean_hsv + tolerance, 0, 255)

        calibrated = True
        print(f"[✅] HSV calibré : {mean_hsv} ± {tolerance}")


def main():
    global recording_enabled
    global display_enabled
    global frame_timestamps
    global aistart,endme
    picam2 = Picamera2()
    picam2.preview_configuration.main.size = (160, 120)
    picam2.preview_configuration.main.format = "RGB888"
    picam2.configure("preview")
    picam2.start()

    tracker = LedTracker()
    detector = BlackRegionDetector()

    param = {"frame": None}
    cv2.namedWindow("Flux")
    cv2.setMouseCallback("Flux", mouse_callback, param)

    #last_time=time.time()

    threading.Thread(target=ai_watcher,daemon=True).start()

    while True:
        frame = picam2.capture_array()

        #now=time.time()
        #fps=1.0/(now-last_time)
        #print(f"FPS:{fps:.2f}")
        #last_time = now 
        param["frame"] = frame.copy()
        
        roi_rect, _ = detector.detect_largest_black_rectangle(frame)
        if calibrated and roi_rect:
            annotated, mask = detect_leds_hsv(frame, lower_bound, upper_bound, roi_coords=roi_rect)
            led_boxes=detect_leds_positions(mask)
            labeled_leds = tracker.assign_labels(led_boxes)

            left=1 if any(l=="g" for l,_ in labeled_leds) else 0
            right=1 if any(l== "d" for l,_ in labeled_leds) else 0

            now=time.time()
            symbol=(left,right)
            led_states_buffer.append((left,right))
            frame_timestamps.append(now)

            if aistart:
                try_decode_from_bufferAI(symbol)

            #sample and record only if recording is enabled
            if recording_enabled:
                global current_sequence
                current_sequence.append((left,right,now))

            for label, (x,y,w,h) in labeled_leds:
                if roi_rect: x,y =x+roi_rect[0],y+roi_rect[1]
                cv2.putText(annotated,label,(x+w//2 - 5, y+h+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255),1)
        else:
            annotated, mask = frame.copy(), np.zeros(frame.shape[:2], dtype=np.uint8)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key==ord('i'):
            aistart = not aistart
            print('Ai state = ', aistart)
        elif key==ord('u'):
            endme= not endme
            print('endme ? = ', endme)
        elif key == ord('c'):
            display_enabled= not display_enabled
        elif key==ord('p'):
            recording_enabled=True
            print("Record start")
        elif key==ord('l'):
            recording_enabled=False
            print("Record stop")
        elif key==ord('k'):
            global label_index,current_label,message_sequences
            if current_sequence:
                if label_index<len(DISTANCE_LABELS):
                    current_label = DISTANCE_LABELS[label_index]
                    for led1,led2, ts in current_sequence:
                        message_sequences.append([led1,led2,ts,current_label])
                    print(f"Sample registered for label {current_label}")
                    label_index +=1
                else:
                    print("tous les labels sont utilises")
                current_sequence = []
            else:
                print("Nothing to register")
        if display_enabled:
            cv2.imshow("Flux", annotated)
            cv2.imshow("Masque", mask)
    
    #Saving the csv
    if message_sequences:
        with open(OUTPUT_CSV, mode='w', newline='') as f:
            writer = csv.writer(f)
            header =["LED1","LED2","Timestamp","Label"]
            writer.writerow(header)
            writer.writerows(message_sequences)
        print('CSV exported')
    else:
        print('no record')
    cv2.destroyAllWindows()
    picam2.close()
    
if __name__ == "__main__":
    main()




