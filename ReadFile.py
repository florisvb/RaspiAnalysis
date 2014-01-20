import numpy as np
import matplotlib.pyplot as plt
import time

import Kalman

### Define kalman filter properties ########
phi = np.matrix([[1, 1, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 1, 1],
                 [0, 0, 0, 1]])
H   = np.matrix([[1, 0, 0, 0],
                 [0, 0, 1, 0]])
P0  = np.matrix([[10, 0, 0, 0],
                 [0, 10, 0, 0],
                 [0, 0, 10, 0],
                 [0, 0, 0, 10]])
Q   = 1*np.matrix(np.eye(4))
R   = 1*np.matrix(np.eye(2))
#############################################

def read(filename):
    
    f = open(filename)
    lines = f.readlines()
    frames = {}
    
    for line in lines:
        if line[0] in ['#', '$', '%', '/']:
            # This line is part of the header
            continue
        
        vals = line.split(',')
        
        framenumber     = int(vals[0])
        nobjs           = int(vals[1])
        timestamp       = float(vals[2])
        
        frame = {'framenumber': framenumber,
                 'timestamp':   timestamp,
                 'nobjs':       nobjs,
                 'objs':        [],
                 }
        
        for obj in range(nobjs):
            x = float(vals[3+obj*3])
            y = float(vals[4+obj*3])
            position = np.array([[x, y]])
            s = float(vals[5+obj*3])
            obj = { 'x': x,    
                    'y': y, 
                    'size': s, 
                    }
            frame['objs'].append(obj)
            
        frames.setdefault(framenumber, frame)
    
    return frames
    
    
def extract_objects(frames):
    
    dataset = {}
    current_objid = 0
    
    for n, frame in frames.items():
        # keep track of which new objects have been "taken"
        objs_accounted_for = []
        
        # iterate through existing trajectories, in order of size
        tracked_objids = dataset.keys()
        tracked_objid_lengths = [len(dataset[i]['frames']) for i in tracked_objids]
        order = np.array(tracked_objid_lengths).argsort()[::-1]
        for k in order:
            tracked_objid = tracked_objids[k]
            tracked_obj = dataset[tracked_objid]
            ntracked_objs = 0
            
            # if trajectory  (e.g. tracked object) existed last frame, attempt to find which new object should be associated with it
            # TODO: with large datasets this might be slow - better to keep track of which objects are "live"
            if tracked_obj['frames'][-1] == n-1:
                ntracked_objs += 1
                tracked_obj_state = tracked_obj['kalmanfilter'].xhat_apriori   # extract estimate of current position based on Kalman model
                tracked_obj_covariance = tracked_obj['kalmanfilter'].P_apriori # extract current covariance based on Kalman model 
                position_covariance = np.linalg.norm(  tracked_obj_covariance.diagonal().T[tracked_obj['statenames']['position']]  )
                
                # iterate through new objects, and calculate error in position compared to tracked object
                position_errors = {}
                for o, obj in enumerate(frame['objs']):
                    if o not in objs_accounted_for:
                        obj_state = np.matrix([[obj['x'], 0, obj['y'], 0]]).T
                        obj_measurement = np.matrix([[obj['x'], obj['y']]]).T
                        error = np.abs(obj_state - tracked_obj_state)
                        position_covariance = np.linalg.norm(  tracked_obj_covariance.diagonal().T[tracked_obj['statenames']['position']]  )
                        position_error = np.linalg.norm( error[tracked_obj['statenames']['position']] )
                        if position_error < 3*np.sqrt(position_covariance): # if mean position error is < 3*sqrt(mean position covariance), it is a potential candidate (this is true, for a properly kalman tracked object)
                            position_errors.setdefault(position_error, o)
                            
                if len(position_errors.keys()) > 0: # pick the best object, if there are multiple
                    least_error = np.min(position_errors.keys())
                    o = position_errors[least_error]
                    obj = frame['objs'][o]
                    tracked_obj['measurement'] = np.hstack( (tracked_obj['measurement'], np.matrix([[obj['x'], obj['y']]]).T) ) # add object's data to the tracked object
                    tracked_obj['size'] =  np.hstack( (tracked_obj['size'], obj['size']) )   
                    tracked_obj['frames'].append(n)
                    tracked_obj['timestamp'].append(frame['timestamp'])
                    xhat, P, K = tracked_obj['kalmanfilter'].update( tracked_obj['measurement'][:,-1] ) # run kalman filter
                    tracked_obj['state'] = np.hstack( (tracked_obj['state'], xhat) )
                    objs_accounted_for.append(o)
        
        # any unnaccounted for objects should spawn new objects
        for o, obj in enumerate(frame['objs']):
            if o not in objs_accounted_for:
                obj_state = np.matrix([[obj['x'], 0, obj['y'], 0]]).T
                obj_measurement = np.matrix([[obj['x'], obj['y']]]).T
                # If not associated with previous object, spawn a new object
                new_obj = { 'objid':        current_objid,
                            'statenames':   {'position': [0, 2], 'velocity': [1,3]},
                            'state':        obj_state,
                            'measurement':  np.matrix([obj['x'], obj['y']]).T,
                            'timestamp':    [frame['timestamp']],
                            'size':         np.array(obj['size']),
                            'frames':       [n],
                            'kalmanfilter': Kalman.DiscreteKalmanFilter(x0=obj_state, 
                                                                        P0=P0, 
                                                                        phi=phi, 
                                                                        gamma=None, 
                                                                        H=H, 
                                                                        Q=Q, 
                                                                        R=R, 
                                                                        gammaW=None)
                          }
                dataset.setdefault(new_obj['objid'], new_obj)
                current_objid += 1
        
        # cull objects
        for objid, obj in dataset.items():
            if obj['frames'][-1] < n-1:
                if len(obj['frames']) == 1:
                    dataset.pop(objid)
            
    return dataset
            
            
def plot_trajectories_of_persistant_objects(dataset, persistance=20, frames=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    if frames is not None:
        for n, frame in frames.items():
            for obj in frame['objs']:
                ax.plot(obj['x'], obj['y'], '*', color='black')
                
    for key, obj in dataset.items():
        if len(obj['frames']) > persistance:
            ax.plot(obj['state'][ obj['statenames']['position'] ][0,:].T, obj['state'][ obj['statenames']['position'] ][1,:].T) 
            
    
            
def print_key_lengths(dataset):
    for key, obj in dataset.items():
        print key, len(obj['frames'])
            
            
            
            
            
def write_dataset_to_text_files(dataset, nobjs=3):
    filenamebase = time.strftime("trajectory_%Y%m%d_%H%M%S_",time.localtime())    
    
    tracked_objids = dataset.keys()
    tracked_objid_lengths = [len(dataset[i]['frames']) for i in tracked_objids]
    order = np.array(tracked_objid_lengths).argsort()[::-1]
    
    for n, k in enumerate(order):
        if n > nobjs:
            break
        
        tracked_objid = tracked_objids[k]
        tracked_obj = dataset[tracked_objid]
        
        
        filename = filenamebase + '_' + str(tracked_objid) + '.txt'
        f = open(filename, 'w')
        
        f.write('# timestamp, x position (px), y position (px), x velocity (px/frame), y velocity (px/frame)')
        for frame in range( len(tracked_obj['frames']) ):
            f.write('%.05f, %f, %f, %f, %f\n'%( tracked_obj['timestamp'][frame], 
                                                tracked_obj['state'][0,frame],
                                                tracked_obj['state'][3,frame],
                                                tracked_obj['state'][1,frame],
                                                tracked_obj['state'][2,frame],
                                                ))
        
            
            
            
            
            
