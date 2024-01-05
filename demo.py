import pybullet as p
import pybullet_data
# import cv2
import numpy as np
import os
import pandas as pd
import time
import predict
from skimage import io
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import pickle



#define joint ranges
min, max= -np.pi, np.pi
joint_ranges = [(min, max), (min, max), (min, max),(min, max),(min, max),(min, max)]


#Top Camera Setup
width, height = 640, 480
top_camera = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0, 0, 0],distance=2.5,yaw=0, pitch=-90, roll=0, upAxisIndex=2)
top_projection = p.computeProjectionMatrixFOV(fov=60, aspect=float(width) / height, nearVal=0.1, farVal=100.0)

# Side Camera Setup
side_camera_position = [2, 0, .5]  
side_camera_target = [0, 0, .5]      # Point the camera is looking at
side_camera_up_vector = [0, 0, 1]   # Which direction is 'up' for the camera
side_camera = p.computeViewMatrix(cameraEyePosition=side_camera_position, cameraTargetPosition=side_camera_target, cameraUpVector=side_camera_up_vector)
side_projection = p.computeProjectionMatrixFOV(fov=60, aspect=float(width) / height, nearVal=0.1, farVal=100.0) # Define the side camera projection matrix

#Front Camera Setup
front_camera_position = [0, 2, .5] 
front_camera_target = [0, 0, .5]      # Point the camera is looking at
front_camera_up_vector = [0, 0, 1]   # Which direction is 'up' for the camera
front_camera = p.computeViewMatrix(cameraEyePosition=front_camera_position, cameraTargetPosition=front_camera_target, cameraUpVector=front_camera_up_vector)
front_projection = p.computeProjectionMatrixFOV(fov=60, aspect=float(width) / height, nearVal=0.1, farVal=100.0) # Define the side camera projection matrix

#corner 1 Camera Setup
corner1_camera_position = [1.5, 1.5, 1]  
corner1_camera_target = [0, 0, .5]      # Point the camera is looking at
corner1_camera_up_vector = [0, 0, 1]   # Which direction is 'up' for the camera
corner1_camera = p.computeViewMatrix(cameraEyePosition=corner1_camera_position, cameraTargetPosition=corner1_camera_target, cameraUpVector=corner1_camera_up_vector)
corner1_projection = p.computeProjectionMatrixFOV(fov=60, aspect=float(width) / height, nearVal=0.1, farVal=100.0)

#corner 2 Camera Setup
corner2_camera_position = [1.5, -1.5, -1]  
corner2_camera_target = [0, 0, .5]      # Point the camera is looking at
corner2_camera_up_vector = [0, 0, 1]   # Which direction is 'up' for the camera
corner2_camera = p.computeViewMatrix(cameraEyePosition=corner2_camera_position, cameraTargetPosition=corner2_camera_target, cameraUpVector=corner2_camera_up_vector)
corner2_projection = p.computeProjectionMatrixFOV(fov=60, aspect=float(width) / height, nearVal=0.1, farVal=100.0)

def get_depth_image(depth_buffer, near, far, width, height):
    depth = far * near / (far - (far - near) * depth_buffer)
    depth_image = np.array(depth * 255, dtype=np.uint8).reshape(height, width)
    return depth_image

def Extract(lst):
    return [item[0] for item in lst]

#connect to a pybulllet sim
if p.isConnected():
    p.disconnect()
p.connect(p.GUI)

p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME,1)
p.setGravity(0, 0, -9.8)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
# plane_id= p.loadURDF("plane.urdf")
arm_id = p.loadURDF("assets/ur5.urdf", [0, 0, 0.4],p.getQuaternionFromEuler([0, 0, 0]))


# #create a directory for images
# images_dir = "UR5_images"
# os.makedirs(images_dir, exist_ok=True)

# Initialize DataFrame
columns = ['ImageID', 'Joint1', 'Joint2', 'Joint3', 'Joint4', 'Joint5', 'Joint6']
positions_df = pd.DataFrame(columns=columns)

temp_data = []

#Hard Coded Joint Angle Target
# target = [0.6437651036289567, -1.2335062711276403, 2.4557816193483397, -2.046707833712239, 1.3771630697058157, -2.332648026683035]

target = [-1.1174159144895100, 0.3294032173512840, -0.9370810367864940, -1.5443014802068600, 0.16574323589562200, 0.11006210159184000]
pred = [-1.1174159144895100, 0.3294032173512840, -0.9624, -1.4749,  0.1441,  0.1512]

pred = [0.15365865900091800, -0.935218921231974, 0.9210, -0.7042,  0.4381, -0.2624]
target = [0.15365865900091800,	-0.935218921231974,	0.934998068423925,	-0.7126856942116650,	0.43345713942753700,	-0.24694073741689900]

pred = [0.17446497555935100, 1.249861461789380, 0.7186,  0.6317,  0.1003, -0.6722]
target = [0.17446497555935100,	1.249861461789380,	0.7204762230355180,	0.6053791215357700,	0.2138024441078490,	-0.6830162646581530]

#Cartesian End Effector Locations
target_loc = [-0.2588303111633394, 0.028918766719470745, 1.2708511959218198]
pred_loc = [-0.2836612767602417, 0.07228159010284901, 1.2571201014199511]

yaw = -30.0
pitch = -45.0
dist = 3.0

p.resetDebugVisualizerCamera( cameraDistance=dist, cameraYaw=yaw, cameraPitch=pitch, cameraTargetPosition=target_loc)
time.sleep(3)

joint_positions = p.calculateInverseKinematics(arm_id,
                                10,
                                target_loc)

#Add text and graphics
p.addUserDebugText('Target',target_loc,[0,0,0],0.5)
p.addUserDebugLine([0,0,0],target_loc,[0,1,0],10)
p.addUserDebugLine(target_loc,pred_loc,[1,0,0],10)
p.addUserDebugPoints([target_loc,pred_loc],[[0,1,0],[1,0,0]],10)

for i in range(1):
    # Set random pose
    for joint_index, (min_val, max_val) in enumerate(joint_ranges):
        p.setJointMotorControl2(bodyUniqueId=arm_id,
                                jointIndex=joint_index,
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=pred[joint_index])
    # Allow some time for the arm to move to the pose
    p.resetDebugVisualizerCamera( cameraDistance=dist-2, cameraYaw=yaw, cameraPitch=pitch, cameraTargetPosition=target_loc)

    errors = []
    # metadata = dict(title='Movie Test', artist='Matplotlib',
    #             comment='Movie support!')
    # writer = FFMpegWriter(fps=15, metadata=metadata)

    # fig = plt.figure()
    # l, = plt.plot([])

    # with writer.saving(fig, "demo_error.mp4", 10):
    for i in range(50):
        images = []
        depths = []
        
        
        for camera_config, image_path_prefix in zip(
            [(side_camera, side_projection),(front_camera, front_projection),(top_camera, top_projection)],
            # , (corner1_camera, corner1_projection), 
            # (corner2_camera, corner2_projection)],
            ['side', 'front', 'top']):
            # 'corner1', 'corner2']):
            # Capture RGB and Depth images
            _, _, rgba_image, depth_buffer, _ = p.getCameraImage(width, height, camera_config[0], camera_config[1])
            
            # Convert RGBA to RGB (discard alpha channel)
            rgb_image = rgba_image[:, :, :3]
            
            # Convert depth buffer to depth image
            depth_image = get_depth_image(np.array(depth_buffer), 0.1, 100.0, width, height)

            images.append(rgb_image)
            depths.append(depth_image)

        # images.extend(depths)
        
        true_joints = np.array(Extract(p.getJointStates(arm_id,[2,3,4,5]))).reshape(1,4)
        
        pred_joints = predict.get_prediction(images)
        
        error = mean_squared_error(true_joints,pred_joints,squared=False)
        errors.append(error)
        print('True Joint Values:', true_joints)
        print('Predicted Joint Values:' ,pred_joints)

        # indices = np.linspace(0,len(errors),len(errors))

        # l.set_data(errors,indices)
        # writer.grab_frame()

        plt.ion()
        plt.plot(errors)
        plt.xlabel('Step')
        plt.ylabel('RMSE (Radians)')
        plt.title('Running State Estimation Error')
        plt.show()
        
        p.stepSimulation()
        time.sleep(1/240)

        
        
        # time.sleep(1)

    final = p.getLinkState(arm_id,9)
    print(final)

    with open("config3", "wb") as fp:   #Pickling
        pickle.dump(errors, fp)
   
