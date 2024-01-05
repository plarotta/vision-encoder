import pickle 
import matplotlib.pyplot as plt
import numpy as np
import cv2


# vidcap = cv2.VideoCapture('simple_demo.mov')
# success,image = vidcap.read()
# count = 0
# while success:
#   cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file      
#   success,image = vidcap.read()
#   print('Read a new frame: ', success)
#   count += 1

with open("config1", "rb") as fp:   # Unpickling
   config1 = pickle.load(fp)

with open("config2", "rb") as fp:   # Unpickling
   config2 = pickle.load(fp)

with open("config3", "rb") as fp:   # Unpickling
   config3 = pickle.load(fp)

average = np.array([(config1 + config2 + config3) / 3 for config1, config2,config3 in zip(config1, config2,config3)])
std = np.std(average)

x = np.linspace(0,len(config1),len(config1))
plt.plot(config1,alpha=0.5,marker='o')
plt.plot(config2,alpha=0.5,marker='x')
plt.plot(config3,alpha=0.5,marker='*')
plt.plot(average,'k')
plt.fill_between(x,0.9*average,1.1*average,alpha=0.5,color='r')

plt.title('Simulation State Estimation Error',weight='bold',size='large')
plt.xlabel('Step',weight='bold')
plt.ylabel('RMSE (Radians)',weight='bold')
plt.legend(['Configuration 1','Configuration 2', 'Configuration 3', 'Average','10 percent error'])
plt.show()