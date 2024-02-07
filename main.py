from matplotlib import pyplot as plt
import cv2

window_name = 'face detector'

try:
    img = cv2.imread('download.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    model = cv2.CascadeClassifier('model.xml') 
    face = model.detectMultiScale(img)

    if len(face) != 0 and  face is not None:
        
        x = face[0][0]
        y = face[0][1]  
        a = face[0][2]
        b = face[0][3]
        
        img = cv2.rectangle(img,(x,y),(x+a,y+b),(255,0,0),3)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        img = cv2.putText(img,'your face', (x,y-10),font,1,(255,0,0),2,cv2.LINE_AA)
        plt.imshow(img)
        plt.show()
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print ('face not detected')
        
except FileNotFoundError:
    print('Could not load image file')
    
except cv2.error:
    print('Could not load cascade classifier model')
    
except:
    print('Unexpected error')
    
