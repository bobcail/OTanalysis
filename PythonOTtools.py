import pandas as pd
import scipy.io
from scipy.optimize import curve_fit
import numpy as np
import pandas as pd
import os
import csv
import glob
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from nptdms import TdmsFile

# To read a TDMS file in our format and turn it into a numpy array
def TDMS_reader(filename):
    file_name = TdmsFile.read(filename)
    group_name = []
    channel_names = []
    channel_data = []
    for group in file_name.groups():
        group_name.append(group.name)
        for channel in group.channels():
            channel_names.append(channel.name)
            channel_data.append(channel.data)
    return group_name, channel_names, channel_data


#for files in a list, named the way they're usually formatted for us
def callTDMS(file_name, cov_window=18):
    base_array_name = file_name.strip('.tdms')
    meta_name = base_array_name + '_meta.txt'
    TDMS_array = TDMS_reader(file_name)[2]
    meta_numpyarray = np.loadtxt(meta_name, dtype = str, delimiter="\t")
    length_of_trace = meta_numpyarray[1][1].astype(float)
    kA = meta_numpyarray[1][12].astype(float)
    kB = meta_numpyarray[1][13].astype(float)
    CalA = meta_numpyarray[1][10].astype(float)
    CalB = meta_numpyarray[1][11].astype(float)
    freq = meta_numpyarray[1][0].astype(float)
    a_pos = (TDMS_array[0] * CalA).astype(np.float32)
    b_pos = (TDMS_array[1] * CalB).astype(np.float32)
    dead_time = freq * 0.016
    half_cov = freq * cov_window / 1000
    time_int = length_of_trace/len(TDMS_array[0])
    time_list = []
    for i in range(len(TDMS_array[0])):
        x_time = i*time_int
        time_list.append(x_time)
    time_array = np.asarray(time_list)
    return(base_array_name, length_of_trace, kA, kB, CalA, CalB, freq, a_pos, b_pos, dead_time, half_cov, time_int, time_array)

#to read a TDMS file with feedback information, which includes the other bead positions
def callTDMS_fb(file_name, cov_window=8):
    base_array_name = file_name.strip('.tdms')
    meta_name = base_array_name + '_meta.txt'
    TDMS_array = TDMS_reader(file_name)[2]
    meta_numpyarray = np.loadtxt(meta_name, dtype = str, delimiter="\t")
    length_of_trace = meta_numpyarray[1][1].astype(float)
    kA = meta_numpyarray[1][12].astype(float)
    kB = meta_numpyarray[1][13].astype(float)
    CalA = meta_numpyarray[1][10].astype(float)
    CalB = meta_numpyarray[1][11].astype(float)
    freq = meta_numpyarray[1][0].astype(float)
    a_pos = (TDMS_array[0] * CalA).astype(np.float32)
    b_pos = (TDMS_array[1] * CalB).astype(np.float32)
    a_error = (TDMS_array[2]*CalA).astype(np.float32)
    b_error = (TDMS_array[3]*CalB).astype(np.float32)
    dead_time = freq * 0.016
    half_cov = freq * cov_window / 1000
    time_int = length_of_trace/len(TDMS_array[0])
    time_list = []
    for i in range(len(TDMS_array[0])):
        x_time = i*time_int
        time_list.append(x_time)
    time_array = np.asarray(time_list)
    return(base_array_name, length_of_trace, kA, kB, CalA, CalB, freq, a_pos, b_pos, dead_time, half_cov, time_int, time_array, a_error, b_error)

#to flip a molecule: that is, if the displacement is overall negative
def flip_molecule(trace):
    flipped_trace = (trace[0], trace[1], trace[2], trace[3], trace[4], trace[5], trace[6], -trace[7], -trace[8], trace[9], trace[10], trace[11], trace[12])
    return(flipped_trace)


#calculate covariance with pandas implementation
def covariance_pds(trace, cov_window=18):
    window_size = int(cov_window/(trace[11]*1000))
    halfwindowint = int(window_size/2)
    bead_pos_df = pd.DataFrame(
        {
        "Bead_A":trace[7],
        "Bead_B":trace[8],
        "ABprod":(trace[7]*trace[8])
        })
    rolling_avg_A = bead_pos_df['Bead_A'].rolling(window_size, min_periods=1).mean()
    rolling_avg_B = bead_pos_df['Bead_B'].rolling(window_size, min_periods=1).mean()
    rolling_avg_AB = bead_pos_df['ABprod'].rolling(window_size, min_periods=1).mean()
    covar_AB = rolling_avg_AB - (rolling_avg_A*rolling_avg_B)
    covar_AB = covar_AB.shift(-halfwindowint)
    covar_AB[-halfwindowint:] = covar_AB.iloc[-halfwindowint-1000:-halfwindowint-1].mean()
    covar_AB = covar_AB.to_numpy(dtype=np.float32)
    if os.path.exists(trace[0] + '/Cov_and_trace/'):
        print('Folder already exists')
    else:
        os.makedirs(trace[0] + '/Cov_and_trace/', exist_ok=True)
    np.save(trace[0] + '/Cov_and_trace/covariance', covar_AB, allow_pickle = True)
    return(covar_AB)


#this function fits the covariance to a bimodal distribution
def bimodal_fit(trace, covar_AB, a1 = 50000, m1 = 4, s1 = 2, a2 = 175000, m2 = 20, s2 = 4):
    cov_hist = np.histogram(covar_AB, bins = 2000)
    bounds = ([10, 0, 0, 10, 2, 0], [np.inf, 20, 20, np.inf, 50, 20])
    popt, pcov = curve_fit(bimodal, xdata=cov_hist[1][:-1:], ydata=cov_hist[0], p0=[a1, m1, s1, a2, m2, s2], bounds=bounds, maxfev=5000)
    plt.close('all')
    plt.plot(cov_hist[1][:-1:], cov_hist[0])
    plt.plot(cov_hist[1][:-1:], gauss(cov_hist[1][:-1:], popt[0], popt[1], popt[2]), 'r-', label='Bound', alpha = 0.5)
    plt.plot(cov_hist[1][:-1:], gauss(cov_hist[1][:-1:], popt[3], popt[4], popt[5]), 'g-', label='Unbound', alpha = 0.5)
    plt.xlim(-10, 60)
    plt.axvline(popt[1], color = 'black')
    plt.axvline(popt[4], color = 'black')
    plt.legend()
    plt.gcf()
    plt.savefig(trace[0]+'/Cov_and_trace/CovarianceHistogram')
    plt.close('all')
    return(popt[1], popt[4])


#plot and save the candidate events
def find_prelimEvents(cov, p1, p2):

#find indices where cov drops below p2
    startp2 = np.diff((cov < p2).astype(int))
    istartp2 = (np.where(startp2 == 1) + np.array(1))[0]
#find indices where cov rises above p2
    stopp2 = np.diff((cov < p2).astype(int))
    istopp2 = (np.where(stopp2 == -1) + np.array(1))[0]
#excluding any index without a preceding drop below p2
    stopp2[0:istartp2[0]] = 0
    istopp2 = (np.where(stopp2 == -1) + np.array(1))[0] 

#find indices where cov drops below p1
    istartp1 = (np.where(np.diff((cov < p1).astype(int)) == 1) + np.array(1))[0]

# Consider events to occur when the covariance first drops below peak 2,
# then drops below peak 1, and finally rises above peak 2.
    d = np.diff(np.sign((istartp2 - istartp1[..., None])))
    i_rows, i_columns = np.where(d > 0)
    unique_columns = np.unique(i_columns)
    istart = istartp2[unique_columns]
    istop = istopp2[unique_columns]
    prelimEvents = np.vstack((istart, istop)).T
    return(prelimEvents)

    

def plot_prelimEvents(trace, cov, prelimEvents):
    time_array = trace[12]
    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[2, 1], hspace=0.25)

    ax1 = plt.subplot(gs[0])
    ax1.plot(time_array, trace[7], label = 'Bead A')
    ax1.plot(time_array, trace[8], label = 'Bead B')
    ax1.legend(loc = 'upper right')

    ax2 = plt.subplot(gs[1])
    ax2.plot(time_array, cov, color = 'red', label = 'Bead covariance')
    ax2.legend(loc = 'upper right')

    for i in range(len(prelimEvents)):
        ax1.axvspan(time_array[prelimEvents[i][0]], time_array[prelimEvents[i][1]], alpha=0.5, color='gray', zorder = 3)
        ax2.axvspan(time_array[prelimEvents[i][0]], time_array[prelimEvents[i][1]], alpha=0.5, color='gray', zorder = 3)
    if os.path.exists(trace[0] + '/Cov_and_trace/'):
        print('Folder already exists')
    else:
        os.makedirs(trace[0] + '/Cov_and_trace/', exist_ok=True)
    plt.gcf()
    plt.savefig(trace[0]+'Cov_and_trace/prelimEvents')
    plt.close('all')

def timeFilterEvents(trace, prelimEvents):
    #this function time filters exclusively by lifetime, includes events that are close together
    dead_time = int(trace[9])
    half_cov = int(trace[10])
    if dead_time > half_cov:
        timeFilteredEvents = prelimEvents[prelimEvents[:, 1]-prelimEvents[:, 0]>dead_time]
    else:
        timeFilteredEvents = prelimEvents[prelimEvents[:, 1]-prelimEvents[:, 0]>half_cov]
    np.save(trace[0] + '/Cov_and_trace/filteredEventTimes', timeFilteredEvents, allow_pickle = True)
    return timeFilteredEvents


def plot_prelimEvents_withfilter(trace, cov, prelimEvents):
    time_array = trace[12]
    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[2, 1], hspace=0.25)

    timeFilteredEvents = timeFilterEvents(trace, prelimEvents)

    ax1 = plt.subplot(gs[0])
    ax1.plot(time_array, trace[7], label = 'Bead A')
    ax1.plot(time_array, trace[8], label = 'Bead B')
    

    ax2 = plt.subplot(gs[1])
    ax2.plot(time_array, cov, color = 'red', label = 'Bead covariance')
    

    for i in range(len(prelimEvents)):
        ax2.axvspan(time_array[prelimEvents[i][0]], time_array[prelimEvents[i][1]], alpha=0.5, color='gray', zorder = 3)
    for i in range(len(timeFilteredEvents)):
        ax1.axvspan(time_array[timeFilteredEvents[i][0]], time_array[timeFilteredEvents[i][1]], alpha=0.5, color='cornflowerblue', zorder=3)
    ax1.axvspan(0, 0, alpha = 0.5, color='cornflowerblue', label = 'Filtered events')
    ax2.axvspan(0, 0, alpha = 0.5, color='gray', label = 'Candidate events')
    ax1.legend(loc = 'upper right')
    ax2.legend(loc = 'upper right')
    
    plt.gcf()
    plt.savefig(trace[0]+'/Cov_and_trace/Events_timefiltered')
    plt.show()
    plt.close('all')
    

def pad_events(timeFilteredEvents, trace):
    paddedEvents = np.append(timeFilteredEvents, [[len(trace[7]), len(trace[7])]], axis = 0)
    paddedEvents = np.append(paddedEvents, [[0, 0]], axis = 0)
    paddedEvents = np.sort(paddedEvents, axis = 0)
    sep = paddedEvents[1:,0] - paddedEvents[:1,1]
    dur = paddedEvents[1:-1,1] - paddedEvents[1:-1,0]
    numPtsBefore = np.floor(0.49*np.minimum(sep[:-1], dur)).astype(int)
    numPtsAfter = np.floor(0.49*np.minimum(sep[1:], dur)).astype(int)
    return(numPtsBefore, numPtsAfter)
        

def save_events_padded(trace, timeFilteredEvents, covar_AB, front_pad=4000, back_pad=4000):
    plt.ioff()
    bead_average = np.mean(np.array([trace[7], trace[8]]), axis=0)
    if os.path.exists(trace[0] + '/events/'):
        print('Folder already exists')
    else:
        os.makedirs(trace[0] + '/events/', exist_ok=True)
    if timeFilteredEvents[0][0] < front_pad:
        timeFilteredEvents = timeFilteredEvents[1::]
    if timeFilteredEvents[-1][1] > int(trace[1]*trace[6]) - back_pad:
        timeFilteredEvents = timeFilteredEvents[:-1:]
        
    for i in range(len(timeFilteredEvents)):
        trace_i =  bead_average[timeFilteredEvents[i][0]-front_pad:timeFilteredEvents[i][1]+back_pad]
        event_time = trace[12][timeFilteredEvents[i][0]-front_pad:timeFilteredEvents[i][1]+back_pad]
        np.save(trace[0] + '/events/covmethod_event_' + str(i), trace_i, allow_pickle=True)
        gs = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[2, 1], hspace=0.25)
        ax1 = plt.subplot(gs[0])
        ax1.plot(event_time, trace_i)
        plt.axvline(trace[12][timeFilteredEvents[i][0]], color = 'gray', alpha=0.6)
        plt.axvline(trace[12][timeFilteredEvents[i][1]], color = 'gray', alpha=0.6)

        ax2 = plt.subplot(gs[1])
        ax2.plot(event_time, covar_AB[timeFilteredEvents[i][0]-front_pad:timeFilteredEvents[i][1]+back_pad], color = 'red')
        plt.axvline(trace[12][timeFilteredEvents[i][0]], color = 'gray', label = 'cov threshold')
        plt.axvline(trace[12][timeFilteredEvents[i][1]], color = 'gray')
        ax2.legend(loc = 'lower right')
        plt.savefig(trace[0] + '/events/covmethod_event_' + str(i))
        plt.close('all')
        
  
def save_events_isoFB(trace, timeFilteredEvents, covar_AB, exten_win=4000, motor_bead='B', attach_padding=500, dist_to_detach=1000):
    plt.ioff()
    if os.path.exists(trace[0] + '/events/'):
        print('Folder already exists')
    else:
        os.makedirs(trace[0] + '/events/', exist_ok=True)
    # Calculate differences between the first value in row n+1 and the second value in row n
    differences = timeFilteredEvents[1:, 0] - timeFilteredEvents[:-1, 1]
    # Create a boolean mask to identify rows where the difference is greater than or equal to 1500
    mask = differences >= 1500
    # Append a True at the end of the mask to retain the last row
    mask = np.append(mask, True)
    # Use the mask to filter the array
    TFE_noreattach = timeFilteredEvents[mask]
    for i in range(len(TFE_noreattach)):
        trace_i_a = trace[7][TFE_noreattach[i][0]-exten_win:TFE_noreattach[i][1]+exten_win]
        trace_i_b = trace[8][TFE_noreattach[i][0]-exten_win:TFE_noreattach[i][1]+exten_win]
        event_time = trace[12][TFE_noreattach[i][0]-exten_win:TFE_noreattach[i][1]+exten_win]
        if motor_bead == 'B':
            av_event_force = np.mean(trace[3]*trace_i_b[exten_win+attach_padding:-(exten_win+attach_padding)])
            av_postevent_force = np.mean(trace[3]*trace_i_b[-(exten_win-dist_to_detach):-(exten_win-dist_to_detach-attach_padding):])
        else:
            av_event_force = np.mean(trace[2]*trace_i_a[exten_win+attach_padding:-(exten_win+attach_padding)])
            av_postevent_force = np.mean(trace[3]*trace_i_b[-(exten_win-dist_to_detach):-(exten_win-dist_to_detach-attach_padding):])
        event_force = av_event_force-av_postevent_force
        event_lifetime = TFE_noreattach[i][1] - TFE_noreattach[i][0]
        if i == 0:
            event_forces_lifetimes = np.array([event_force, event_lifetime], dtype=np.float32)
        else:
            event_forces_lifetimes = np.concatenate((event_forces_lifetimes, np.array([event_force, event_lifetime])))
        np.save(trace[0] + '/events/covmethod_event_a_' + str(i), trace_i_a, allow_pickle=True)
        np.save(trace[0] + '/events/covmethod_event_b_' + str(i), trace_i_b, allow_pickle=True)
        gs = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[2, 1], hspace=0.25)
        ax1 = plt.subplot(gs[0])
        ax1.plot(event_time, trace_i_a)
        ax1.plot(event_time, trace_i_b)
        plt.axvline(trace[12][TFE_noreattach[i][0]], color = 'gray', alpha=0.6)
        plt.axvline(trace[12][TFE_noreattach[i][1]], color = 'gray', alpha=0.6)
        
        ax2 = plt.subplot(gs[1])
        ax2.plot(event_time, covar_AB[TFE_noreattach[i][0]-exten_win:TFE_noreattach[i][1]+exten_win], color = 'red')
        plt.axvline(trace[12][TFE_noreattach[i][0]], color = 'gray', label = 'cov threshold')
        plt.axvline(trace[12][TFE_noreattach[i][1]], color = 'gray')
        ax2.legend(loc = 'lower right')
        plt.savefig(trace[0] + '/events/covmethod_event_' + str(i))
        plt.close('all')
    event_forces_lifetimes = event_forces_lifetimes.reshape((len(TFE_noreattach), 2))
    np.save(trace[0] + '/events/force_vs_lifetime', event_forces_lifetimes, allow_pickle=True)
    
        
def extend_events_singletrace(trace, ms_avg = 20, exten_win = 4000):
    num_points_avg = ms_avg * (trace[6]/1000)
    #load all the traces into a list!
    event_array_files = glob.glob(trace[0] + '/events/*.npy')
    event_arrays = []
    for file in event_array_files:
        event_array = np.load(file)
        event_arrays.append(event_array)
    #_ max lifetime
    max_lifetime = len(max(event_arrays, key=lambda arr: len(arr))) - 2 * exten_win
    #pad the shorter events forward to match longest event
    for i in range(len(event_arrays)):
        full_event_i = event_arrays[i][exten_win:-exten_win] #just the event, no padding
        lengthened_event_i_fwd = np.concatenate((full_event_i, \
            np.full((max_lifetime-len(full_event_i)), np.mean(full_event_i[-num_points_avg::]))))
        lengthened_event_i_rev = np.concatenate((np.full((max_lifetime-len(full_event_i)), \
            np.mean(full_event_i[:num_points_avg])), full_event_i))
        lengthened_padded_event_i_fwd = np.concatenate((event_arrays[i][:exten_win], \
            lengthened_event_i_fwd, event_arrays[i][-exten_win:]))
        lengthened_padded_event_i_rev = np.concatenate((event_arrays[i][:exten_win], \
            lengthened_event_i_rev, event_arrays[i][-exten_win:]))
        if i == 0:
            all_events_array_fwd = lengthened_padded_event_i_fwd
            all_events_array_rev = lengthened_padded_event_i_rev
        else:
            all_events_array_fwd = np.vstack((all_events_array_fwd, lengthened_padded_event_i_fwd))
            all_events_array_rev = np.vstack((all_events_array_rev, lengthened_padded_event_i_rev))
    #np.save(trace[0] + '_FwdEventAvg', all_events_array_fwd, allow_pickle=True)
    #np.save(trace[0] + '_RevEventAvg', all_events_array_rev, allow_pickle=True)
    return(all_events_array_fwd, all_events_array_rev)

def extend_events_combined(file_list, freq=250000, ms_avg=20, exten_win=4000):
    num_points_avg = int(ms_avg * int((freq/1000)))
    #load all the traces into a list!
    event_arrays = []
    for file in file_list:
        event_array = np.load(file)
        event_arrays.append(event_array)
    #find max lifetime
    max_lifetime = len(max(event_arrays, key=lambda arr: len(arr))) - 2 * exten_win
    #pad the shorter events forward to match longest event
    for i in range(len(event_arrays)):
        full_event_i = event_arrays[i][exten_win:-exten_win] #just the event, no extension window
        lengthened_event_i_fwd = np.concatenate((full_event_i, \
            np.full((max_lifetime-len(full_event_i)), np.mean(full_event_i[-num_points_avg::]))))
        lengthened_event_i_rev = np.concatenate((np.full((max_lifetime-len(full_event_i)), \
            np.mean(full_event_i[:num_points_avg])), full_event_i))
        lengthened_padded_event_i_fwd = np.concatenate((event_arrays[i][:exten_win], \
            lengthened_event_i_fwd, event_arrays[i][-exten_win:]))
        lengthened_padded_event_i_rev = np.concatenate((event_arrays[i][:exten_win], \
            lengthened_event_i_rev, event_arrays[i][-exten_win:]))
        if i == 0:
            all_events_array_fwd = lengthened_padded_event_i_fwd
            all_events_array_rev = lengthened_padded_event_i_rev
        else:
            all_events_array_fwd = np.vstack((all_events_array_fwd, lengthened_padded_event_i_fwd))
            all_events_array_rev = np.vstack((all_events_array_rev, lengthened_padded_event_i_rev))
    return(all_events_array_fwd, all_events_array_rev)

def extend_events_combined_withprint(file_list, freq=250000, ms_avg=20, exten_win=4000):
    num_points_avg = int(ms_avg * int((freq/1000)))
    #load all the traces into a list!
    event_arrays = []
    for file in file_list:
        event_array = np.load(file)
        event_arrays.append(event_array)
    #find max lifetime
    max_lifetime = len(max(event_arrays, key=lambda arr: len(arr))) - 2 * exten_win
    #pad the shorter events forward to match longest event
    for i in range(len(event_arrays)):
        print('extending event ' + str(i))
        full_event_i = event_arrays[i][exten_win:-exten_win] #just the event, no padding
        lengthened_event_i_fwd = np.concatenate((full_event_i, \
            np.full((max_lifetime-len(full_event_i)), np.mean(full_event_i[-num_points_avg::]))))
        lengthened_event_i_rev = np.concatenate((np.full((max_lifetime-len(full_event_i)), \
            np.mean(full_event_i[:num_points_avg])), full_event_i))
        lengthened_padded_event_i_fwd = np.concatenate((event_arrays[i][:exten_win], \
            lengthened_event_i_fwd, event_arrays[i][-exten_win:]))
        lengthened_padded_event_i_rev = np.concatenate((event_arrays[i][:exten_win], \
            lengthened_event_i_rev, event_arrays[i][-exten_win:]))
        if i == 0:
            print('Forging array')
            all_events_array_fwd = lengthened_padded_event_i_fwd
            all_events_array_rev = lengthened_padded_event_i_rev
        else:
            print('Adding event ' + str(i) + ' to array')
            all_events_array_fwd = np.vstack((all_events_array_fwd, lengthened_padded_event_i_fwd))
            all_events_array_rev = np.vstack((all_events_array_rev, lengthened_padded_event_i_rev))
    #np.save('/Users/bob/Desktop/20230411/FwdEventAvg', all_events_array_fwd, allow_pickle=True)
    #np.save('/Users/bob/Desktop/20230411/RevEventAvg', all_events_array_rev, allow_pickle=True)
    return(all_events_array_fwd, all_events_array_rev)

def extend_events_combined_withpadding(file_list, freq=250000, ms_avg=4, exten_win=4000, padding=2):
    #the number of points to average when extending the events
    num_points_avg = int(ms_avg * int((freq/1000)))
    #the padding, ie how far from the transition point to do the averaging
    num_points_pad = int(padding * int((freq/1000)))
    #load all the traces into a list!
    event_arrays = []
    for file in file_list:
        event_array = np.load(file)
        event_arrays.append(event_array)
    #find max lifetime, so you can fill all the other events to this length
    max_lifetime = len(max(event_arrays, key=lambda arr: len(arr))) - 2 * exten_win
    #pad the shorter events forward to match longest event
    for i in range(len(event_arrays)):
        full_event_i = event_arrays[i][exten_win:-exten_win] #just the event, no extension window
        #lengthen the event forward like this:
        #1 create a full array of length max_lifetime-length of full event, such that final length is max_lifetime
        #calculate the average over num_points_avg, num_points_pad from the end, put this value into full array
        #concatenate full_event_i with new full array
        lengthened_event_i_fwd = np.concatenate((full_event_i, \
            np.full((max_lifetime-len(full_event_i)), np.mean(full_event_i[-(num_points_avg+num_points_pad):-num_points_pad]))))
        lengthened_event_i_rev = np.concatenate((np.full((max_lifetime-len(full_event_i)), \
            np.mean(full_event_i[num_points_pad:(num_points_pad+num_points_avg)])), full_event_i))
        lengthened_padded_event_i_fwd = np.concatenate((event_arrays[i][:exten_win], \
            lengthened_event_i_fwd, event_arrays[i][-exten_win:]))
        lengthened_padded_event_i_rev = np.concatenate((event_arrays[i][:exten_win], \
            lengthened_event_i_rev, event_arrays[i][-exten_win:]))
        if i == 0:
            all_events_array_fwd = lengthened_padded_event_i_fwd
            all_events_array_rev = lengthened_padded_event_i_rev
        else:
            all_events_array_fwd = np.vstack((all_events_array_fwd, lengthened_padded_event_i_fwd))
            all_events_array_rev = np.vstack((all_events_array_rev, lengthened_padded_event_i_rev))
    return(all_events_array_fwd, all_events_array_rev)


def extend_events_combined_covmethod(file_list, freq=250000, ms_avg=1, front_pad = 4000, back_pad = 4000, postbind_exten=4, preUB_exten=4):
    #the number of points to average when extending the events
    num_points_avg = int(ms_avg * int((freq/1000)))
    #the postbind extension, ie how far from the binding point to do the averaging
    postbind_extenpoints = int(postbind_exten * int((freq/1000)))
    #the pre-unbind extension, ie how far from the unbinding point to do the averaging
    preUB_extenpoints = int(preUB_exten * int((freq/1000)))
    #load all the traces into a list!
    event_arrays = []
    for file in file_list:
        event_array = np.load(file)
        event_arrays.append(event_array)
    #find max lifetime, so you can fill all the other events to this length
    max_lifetime = len(max(event_arrays, key=lambda arr: len(arr))) - (front_pad + back_pad)
    #pad the shorter events forward to match longest event
    for i in range(len(event_arrays)):
        full_event_i = event_arrays[i][front_pad:-back_pad] #just the event, no padding
        #lengthen the event forward like this:
        #1 create a full array of length max_lifetime-length of full event, such that final length is max_lifetime
        #calculate the average over num_points_avg, num_points_pad from the end, put this value into full array
        #concatenate full_event_i with new full array
        lengthened_event_i_fwd = np.concatenate((full_event_i, \
            np.full((max_lifetime-len(full_event_i)), np.mean(full_event_i[-(num_points_avg+preUB_extenpoints):-preUB_extenpoints]))))
        lengthened_event_i_rev = np.concatenate((np.full((max_lifetime-len(full_event_i)), \
            np.mean(full_event_i[postbind_extenpoints:(postbind_extenpoints+num_points_avg)])), full_event_i))
        lengthened_padded_event_i_fwd = np.concatenate((event_arrays[i][:front_pad], \
            lengthened_event_i_fwd, event_arrays[i][-back_pad:]))
        lengthened_padded_event_i_rev = np.concatenate((event_arrays[i][:front_pad], \
            lengthened_event_i_rev, event_arrays[i][-back_pad:]))
        if i == 0:
            all_events_array_fwd = lengthened_padded_event_i_fwd
            all_events_array_rev = lengthened_padded_event_i_rev
        else:
            all_events_array_fwd = np.vstack((all_events_array_fwd, lengthened_padded_event_i_fwd))
            all_events_array_rev = np.vstack((all_events_array_rev, lengthened_padded_event_i_rev))
    return(all_events_array_fwd, all_events_array_rev)


def event_ensemble_avg_fromfile(filename, exten_win=4000, freq=250000):
    fwd_traces = np.load(filename + '_FwdEventAvg.npy', allow_pickle=True)
    rev_traces = np.load(filename + '_RevEventAvg.npy', allow_pickle=True)
    all_events_unpad_fwd = fwd_traces[...,exten_win:-exten_win]
    all_events_unpad_rev = rev_traces[...,exten_win:-exten_win]
    ###start here, need to figure out event_times
    time_interval = 1/freq
    ext_event_time_padded = np.arange(0, len(fwd_traces[0])) * time_interval
    #time array for the lengthened events, unpadded, necessary for fitting
    ext_event_time = ext_event_time_padded[exten_win:-exten_win]
    initial_params_fwd = [np.mean(all_events_unpad_fwd[:,-exten_win::])-np.mean(all_events_unpad_fwd[:,:exten_win:]),\
                10, np.mean(all_events_unpad_fwd[:,:exten_win:])]
    initial_params_rev = [np.mean(all_events_unpad_rev[:,-exten_win::])-np.mean(all_events_unpad_rev[:,:exten_win:]),\
                1, np.mean(all_events_unpad_rev[:,:exten_win:])]
    popt1, pcov1 = scipy.optimize.curve_fit(fwd_exponential, \
        xdata = ext_event_time, ydata = np.mean(all_events_unpad_fwd, axis = 0), p0 = initial_params_fwd)
    popt2, pcov2 = scipy.optimize.curve_fit(rev_exponential, \
        xdata = ext_event_time, ydata = np.mean(all_events_unpad_rev, axis = 0), p0 = initial_params_rev)
    yfit_fwd = fwd_exponential(ext_event_time, *popt1)
    yfit_rev = rev_exponential(ext_event_time, *popt2)
    plt.close()
    plt.plot(ext_event_time_padded, np.mean(fwd_traces, axis = 0), alpha = 0.7, label = "Forward average")
    plt.plot(ext_event_time_padded, np.mean(rev_traces, axis = 0), color = 'red', alpha = 0.6, label = "Reverse average")
    plt.plot(ext_event_time, yfit_fwd, 'mediumblue', label = popt1[1])
    plt.plot(ext_event_time, yfit_rev, 'maroon', label = popt2[1])
    plt.legend(loc = 'lower right')
    plt.gcf()
    plt.savefig(filename + '_Ensemble_average')
    plt.close('all')
    
#create a list of event files
def find_event_npyfiles(folder_path, keyword=None, extension=None):
    file_pattern = '*'
    
    if keyword:
        file_pattern += keyword + '*'
    
    if extension:
        file_pattern += extension
    
    file_list = glob.glob(os.path.join(folder_path, '**', file_pattern), recursive=True)
    
    return file_list
    
def gauss(x, a, b, c):
    return a * np.exp(-((x - b) / c) ** 2)

def bimodal(x, a1, b1, c1, a2, b2, c2):
    return gauss(x, a1, b1, c1) + gauss(x, a2, b2, c2)

def fwd_exponential(t, A, k, n_offset):
    return A * (1 - np.exp(-k * t)) + n_offset

def rev_exponential(t, A, k, n_offset):
    return A * (np.exp(k * t)) + n_offset

def make_name_list_inorder(filebase, num_files):
    #this function creates a list in order of files to run over, no skipping or anything
    #following the format of optical trap file names
    filename_list = []
    for i in range(1, num_files + 1):
        file_name = filebase + f"{i:03d}"  
        filename_list.append(file_name)
    return(filename_list)

def make_name_list_fromidx(filebase, idx_list):
    #this function takes a list of indexes and creates a file for each index supplied, so it can skip
    filename_list = []
    for i in idx_list:
        file_name = filebase + f"{i:03d}"  
        filename_list.append(file_name)
    return(filename_list)


#old function for extending for single trace, mostly for testing/troubleshooting purposes
def extend_events_singletrace(trace, ms_avg = 20, exten_win = 4000):
    num_points_avg = ms_avg * (trace[6]/1000)
    #load all the traces into a list!
    event_array_files = glob.glob(trace[0] + '/events/*.npy')
    event_arrays = []
    for file in event_array_files:
        event_array = np.load(file)
        event_arrays.append(event_array)
    #find max lifetime
    max_lifetime = len(max(event_arrays, key=lambda arr: len(arr))) - 2 * exten_win
    #pad the shorter events forward to match longest event
    for i in range(len(event_arrays)):
        full_event_i = event_arrays[i][exten_win:-exten_win] #just the event, no padding
        lengthened_event_i_fwd = np.concatenate((full_event_i, \
            np.full((max_lifetime-len(full_event_i)), np.mean(full_event_i[-num_points_avg::]))))
        lengthened_event_i_rev = np.concatenate((np.full((max_lifetime-len(full_event_i)), \
            np.mean(full_event_i[:num_points_avg])), full_event_i))
        lengthened_padded_event_i_fwd = np.concatenate((event_arrays[i][:exten_win], \
            lengthened_event_i_fwd, event_arrays[i][-exten_win:]))
        lengthened_padded_event_i_rev = np.concatenate((event_arrays[i][:exten_win], \
            lengthened_event_i_rev, event_arrays[i][-exten_win:]))
        if i == 0:
            all_events_array_fwd = lengthened_padded_event_i_fwd
            all_events_array_rev = lengthened_padded_event_i_rev
        else:
            all_events_array_fwd = np.vstack((all_events_array_fwd, lengthened_padded_event_i_fwd))
            all_events_array_rev = np.vstack((all_events_array_rev, lengthened_padded_event_i_rev))
    np.save(trace[0] + '_FwdEventAvg', all_events_array_fwd, allow_pickle=True)
    np.save(trace[0] + '_RevEventAvg', all_events_array_rev, allow_pickle=True)
    return(all_events_array_fwd, all_events_array_rev)


#create an ensemble average from a single file of event averages, this is an old version?
def event_ensemble_avg_fromfile(filename, exten_win=4000, freq=250000):
    fwd_traces = np.load(filename + 'FwdEventAvg.npy', allow_pickle=True)
    rev_traces = np.load(filename + 'RevEventAvg.npy', allow_pickle=True)
    all_events_unpad_fwd = fwd_traces[...,exten_win:-exten_win]
    all_events_unpad_rev = rev_traces[...,exten_win:-exten_win]
    ###start here, need to figure out event_times
    time_interval = 1/freq
    ext_event_time_padded = np.arange(0, len(fwd_traces[0])) * time_interval
    #time array for the lengthened events, unpadded, necessary for fitting
    ext_event_time = ext_event_time_padded[exten_win:-exten_win]
    initial_params_fwd = [np.mean(all_events_unpad_fwd[:,-exten_win::])-np.mean(all_events_unpad_fwd[:,:exten_win:]),\
                10, np.mean(all_events_unpad_fwd[:,:exten_win:])]
    initial_params_rev = [np.mean(all_events_unpad_rev[:,-exten_win::])-np.mean(all_events_unpad_rev[:,:exten_win:]),\
                1, np.mean(all_events_unpad_rev[:,:exten_win:])]
    popt1, pcov1 = scipy.optimize.curve_fit(fwd_exponential, \
        xdata = ext_event_time, ydata = np.mean(all_events_unpad_fwd, axis = 0), p0 = initial_params_fwd)
    popt2, pcov2 = scipy.optimize.curve_fit(rev_exponential, \
        xdata = ext_event_time, ydata = np.mean(all_events_unpad_rev, axis = 0), p0 = initial_params_rev)
    yfit_fwd = fwd_exponential(ext_event_time, *popt1)
    yfit_rev = rev_exponential(ext_event_time, *popt2)
    plt.close()
    plt.plot(ext_event_time_padded, np.mean(fwd_traces, axis = 0), alpha = 0.7, label = "Forward average")
    plt.plot(ext_event_time_padded, np.mean(rev_traces, axis = 0), color = 'red', alpha = 0.6, label = "Reverse average")
    plt.plot(ext_event_time, yfit_fwd, 'mediumblue', label = popt1[1])
    plt.plot(ext_event_time, yfit_rev, 'maroon', label = popt2[1])
    plt.legend(loc = 'lower right')
    plt.gcf()
    plt.savefig(filename + 'Ensemble_average')
    
def export_lifetimes_MEMLET(file_list, folder, exten_win=4000, freq=250000):
    lifetime_list = []
    for file in file_list:
        indiv_event = np.load(file)
        lifetime = ((len(indiv_event)/freq)-(2*exten_win/freq))
        lifetime_list.append(lifetime)
    with open(folder + '/EventDurations.txt', 'w') as f:
        for line in lifetime_list:
            f.write(f"{line}\n")
    return(lifetime_list)

def cov_mask_fb(covariance, cutoff_ratio = 2):
    mask = (np.absolute(covariance[1:-1]/(np.mean(covariance)))>cutoff_ratio)
    covariance[1:-1][mask] = np.mean(covariance)
    return covariance


#find breakpoints for only one bead
def find_breakpoints_isoFB(timeFilteredEvents, trace, padding_win=16, motor_bead = 'B'):
    padding_win = int((padding_win * trace[6])/1000)
    #create a window over which to search
    search_win = 2*padding_win
    #create the padded breakpoint windows
    TFE_padded = find_breakpoint_windows(timeFilteredEvents, trace)
    #create lists of the start and end breaks
    start_breaks = []
    end_breaks = []
    if motor_bead == 'B':
        for i in range(len(TFE_padded)):
            if TFE_padded[i, 1] - TFE_padded[i, 0] > search_win*4:
                break_start_window = trace[8][TFE_padded[i][0]:(TFE_padded[i][0]+search_win)]
                break_end_window = trace[8][(TFE_padded[i][1]-search_win):TFE_padded[i][1]]
                model_start = rpt.Dynp(model="l1")
                model_start.fit(break_start_window)
                break_start, trace_end = model_start.predict(n_bkps=1)
                start_breaks.append(TFE_padded[i][0] + break_start)
                model_end = rpt.Dynp(model="l1")
                model_end.fit(break_end_window)
                break_end, end_trace = model_end.predict(n_bkps=1)
                end_breaks.append(TFE_padded[i][1]-search_win + break_end)
            else:
                shortdur_halfwind = int(np.floor((timeFilteredEvents[i][1]-timeFilteredEvents[i][0])/2)) 
                break_start_window = trace[8][TFE_padded[i][0]:(TFE_padded[i][0]+shortdur_halfwind+padding_win)]
                break_end_window = trace[8][(TFE_padded[i][1]-shortdur_halfwind-padding_win):TFE_padded[i][1]]
                model_start = rpt.Dynp(model="l1")
                model_start.fit(break_start_window)
                break_start, trace_end = model_start.predict(n_bkps=1)
                start_breaks.append(TFE_padded[i][0] + break_start)
                model_end = rpt.Dynp(model="l1")
                model_end.fit(break_end_window)
                break_end, end_trace = model_end.predict(n_bkps=1)
                end_breaks.append(TFE_padded[i][1]- shortdur_halfwind - padding_win + break_end)                
    else:
        for i in range(len(TFE_padded)):
            if TFE_padded[i, 1] - TFE_padded[i, 0] > search_win*4:
                break_start_window = trace[7][TFE_padded[i][0]:(TFE_padded[i][0]+search_win)]
                break_end_window = trace[7][(TFE_padded[i][1]-search_win):TFE_padded[i][1]]
                model_start = rpt.Dynp(model="l1")
                model_start.fit(break_start_window)
                break_start, trace_end = model_start.predict(n_bkps=1)
                start_breaks.append(TFE_padded[i][0] + break_start)
                model_end = rpt.Dynp(model="l1")
                model_end.fit(break_end_window)
                break_end, end_trace = model_end.predict(n_bkps=1)
                end_breaks.append(TFE_padded[i][1]-search_win + break_end)
            else:
                shortdur_halfwind = int(np.floor((timeFilteredEvents[i][1]-timeFilteredEvents[i][0])/2)) 
                break_start_window = trace[7][TFE_padded[i][0]:(TFE_padded[i][0]+shortdur_halfwind+padding_win)]
                break_end_window = trace[7][(TFE_padded[i][1]-shortdur_halfwind-padding_win):TFE_padded[i][1]]
                model_start = rpt.Dynp(model="l1")
                model_start.fit(break_start_window)
                break_start, trace_end = model_start.predict(n_bkps=1)
                start_breaks.append(TFE_padded[i][0] + break_start)
                model_end = rpt.Dynp(model="l1")
                model_end.fit(break_end_window)
                break_end, end_trace = model_end.predict(n_bkps=1)
                end_breaks.append(TFE_padded[i][1]- shortdur_halfwind - padding_win + break_end)
    indices_to_remove = []
    for i in range(len(start_breaks)):
        difference = end_breaks[i] - start_breaks[i]
        if difference <= 2000:
            indices_to_remove.append(i)

    # Remove entries from both lists using the collected indices
    for index in sorted(indices_to_remove, reverse=True):
        del start_breaks[index]
        del end_breaks[index]
    return(start_breaks, end_breaks)


def save_events_isoFB(trace, timeFilteredEvents, covar_AB, exten_win=4000, motor_bead='B', attach_padding=500, dist_to_detach=1000):
    plt.ioff()
    if os.path.exists(trace[0] + '/events/'):
        print('Folder already exists')
    else:
        os.makedirs(trace[0] + '/events/', exist_ok=True)
    # Calculate differences between the first value in row n+1 and the second value in row n
    differences = timeFilteredEvents[1:, 0] - timeFilteredEvents[:-1, 1]
    # Create a boolean mask to identify rows where the difference is greater than or equal to 1500
    mask = differences >= 1500
    # Append a True at the end of the mask to retain the last row
    mask = np.append(mask, True)
    # Use the mask to filter the array
    TFE_noreattach = timeFilteredEvents[mask]
    for i in range(len(TFE_noreattach)):
        trace_i_a = trace[7][TFE_noreattach[i][0]-exten_win:TFE_noreattach[i][1]+exten_win]
        trace_i_b = trace[8][TFE_noreattach[i][0]-exten_win:TFE_noreattach[i][1]+exten_win]
        event_time = trace[12][TFE_noreattach[i][0]-exten_win:TFE_noreattach[i][1]+exten_win]
        if motor_bead == 'B':
            av_event_force = np.mean(trace[3]*trace_i_b[exten_win+attach_padding:-(exten_win+attach_padding)])
            av_postevent_force = np.mean(trace[3]*trace_i_b[-(exten_win-dist_to_detach):-(exten_win-dist_to_detach-attach_padding):])
        else:
            av_event_force = np.mean(trace[2]*trace_i_a[exten_win+attach_padding:-(exten_win+attach_padding)])
            av_postevent_force = np.mean(trace[3]*trace_i_b[-(exten_win-dist_to_detach):-(exten_win-dist_to_detach-attach_padding):])
        event_force = av_event_force-av_postevent_force
        event_lifetime = TFE_noreattach[i][1] - TFE_noreattach[i][0]
        if i == 0:
            event_forces_lifetimes = np.array([event_force, event_lifetime], dtype=np.float32)
        else:
            event_forces_lifetimes = np.concatenate((event_forces_lifetimes, np.array([event_force, event_lifetime])))
        np.save(trace[0] + '/events/covmethod_event_a_' + str(i), trace_i_a, allow_pickle=True)
        np.save(trace[0] + '/events/covmethod_event_b_' + str(i), trace_i_b, allow_pickle=True)
        gs = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[2, 1], hspace=0.25)
        ax1 = plt.subplot(gs[0])
        ax1.plot(event_time, trace_i_a)
        ax1.plot(event_time, trace_i_b)
        plt.axvline(trace[12][TFE_noreattach[i][0]], color = 'gray', alpha=0.6)
        plt.axvline(trace[12][TFE_noreattach[i][1]], color = 'gray', alpha=0.6)
        #the area to be averaged for attached force
        plt.axvspan(trace[12][TFE_noreattach[i][0]+attach_padding], trace[12][TFE_noreattach[i][1]-(attach_padding)], color = 'royalblue', alpha=0.2)
        #the area to be averaged for post-detach force
        plt.axvspan(trace[12][TFE_noreattach[i][1]+dist_to_detach], trace[12][TFE_noreattach[i][1]+dist_to_detach+attach_padding], color='purple', alpha=0.2)
        
        ax2 = plt.subplot(gs[1])
        ax2.plot(event_time, covar_AB[TFE_noreattach[i][0]-exten_win:TFE_noreattach[i][1]+exten_win], color = 'red')
        plt.axvline(trace[12][TFE_noreattach[i][0]], color = 'gray', label = 'cov threshold')
        plt.axvline(trace[12][TFE_noreattach[i][1]], color = 'gray')
        ax2.legend(loc = 'lower right')
        plt.savefig(trace[0] + '/events/covmethod_event_' + str(i))
        plt.close('all')
    event_forces_lifetimes = event_forces_lifetimes.reshape((len(TFE_noreattach), 2))
    np.save(trace[0] + '/events/force_vs_lifetime', event_forces_lifetimes, allow_pickle=True)


#save A and B traces for one-bead breaks
def save_bkp_trace_isoFB(trace, start_breaks, end_breaks, exten_win=4000, motor_bead='B', attach_padding=500):
    plt.ioff()
    if os.path.exists(trace[0] + '/events/'):
        print('Folder already exists')
    else:
        os.makedirs(trace[0] + '/events/', exist_ok=True)
    #event_forces_lifetimes = np.array([], dtype=np.float32)
    for i in range(len(start_breaks)):
        trace_i_a = trace[7][start_breaks[i]-exten_win:end_breaks[i]+exten_win]
        trace_i_b = trace[8][start_breaks[i]-exten_win:end_breaks[i]+exten_win]
        if motor_bead == 'B':
            av_event_force = np.mean(trace[3]*trace_i_b[exten_win+attach_padding:-(exten_win+attach_padding)])
            av_postevent_force = np.mean(trace[3]*trace_i_b[-exten_win:-(exten_win-1600):])
        else:
            av_event_force = np.mean(trace[2]*trace_i_a[exten_win+attach_padding:-(exten_win+attach_padding)])
            av_postevent_force = np.mean(trace[2]*trace_i_a[-exten_win:-(exten_win-1600):])
        event_force = av_event_force-av_postevent_force
        event_lifetime = end_breaks[i] - start_breaks[i]
        if i == 0:
            event_forces_lifetimes = np.array([event_force, event_lifetime], dtype=np.float32)
        else:
            event_forces_lifetimes = np.concatenate((event_forces_lifetimes, np.array([event_force, event_lifetime])))
        np.save(trace[0] + '/events/bkp_event_a_' + str(i), trace_i_a, allow_pickle=True)
        np.save(trace[0] + '/events/bkp_event_b_' + str(i), trace_i_b, allow_pickle=True)
        plt.figure()
        plt.plot(trace[12][start_breaks[i]-exten_win:end_breaks[i]+exten_win], trace[7][start_breaks[i]-exten_win:end_breaks[i]+exten_win])
        plt.plot(trace[12][start_breaks[i]-exten_win:end_breaks[i]+exten_win], trace[8][start_breaks[i]-exten_win:end_breaks[i]+exten_win])
        plt.axvline(trace[12][start_breaks[i]], color = 'black')
        plt.axvline(trace[12][end_breaks[i]], color = 'black')
        plt.savefig(trace[0] + '/events/bkp_event_' + str(i))
        plt.close('all')
    event_forces_lifetimes = event_forces_lifetimes.reshape((len(start_breaks), 2))
    np.save(trace[0] + '/events/force_vs_lifetime', event_forces_lifetimes, allow_pickle=True)
    
    
def find_forcelifetime_files(folder_path, keyword=None, extension=None):
    file_pattern = '*'
    
    if keyword:
        file_pattern += keyword + '*'
    
    if extension:
        file_pattern += extension
    
    file_list = glob.glob(os.path.join(folder_path, '**', file_pattern), recursive=True)
    
    return file_list


def step_size_dist(file_list, freq=250000, ms_avg=1, ms_to_trans=4, exten_win=4000):
    num_points_avg = int(ms_avg * int((freq/1000)))
    pts_to_trans = int(ms_to_trans * int((freq/1000)))
    substep_1_array = np.zeros(shape = len(file_list))
    substep_2_array = np.zeros(shape = len(file_list))
    totalstep_array = np.zeros(shape = len(file_list))
    for index, file in enumerate(file_list):
        event_array = np.load(file)
        pre_binding_array = event_array[0:exten_win]
        bound_array = event_array[exten_win:-exten_win]
        post_release_array = event_array[-exten_win::]
        prestroke_pos = np.mean(pre_binding_array[-(num_points_avg+pts_to_trans):-pts_to_trans:])
        substep1_pos = np.mean(bound_array[pts_to_trans:(ms_avg+pts_to_trans)])
        substep2_pos = np.mean(bound_array[-(ms_avg+pts_to_trans):-pts_to_trans:])
        poststroke_pos = np.mean(post_release_array[pts_to_trans:(ms_avg+pts_to_trans)])
        substep1_size = substep1_pos - prestroke_pos
        substep2_size = substep2_pos - substep1_pos
        totalstep_size = substep2_pos - poststroke_pos
        substep_1_array[index] = substep1_size
        substep_2_array[index] = substep2_size
        totalstep_array[index] = totalstep_size
    np.save(file_list[0][:-38] + 'Substep1_dist', substep_1_array)
    np.save(file_list[0][:-38] + 'Substep2_dist', substep_2_array)
    np.save(file_list[0][:-38] + 'Totalstep_dist', totalstep_array)
    plt.hist(substep_1_array, bins = 40)
    mean_value1 = np.mean(substep_1_array)
    plt.axvline(mean_value1, color = 'black')
    plt.legend([f'Mean substep 1 size = {mean_value1:.2f} nm'], loc='upper left')
    plt.savefig(file_list[0][:-38] + 'Substep1_hist')
    plt.close()
    plt.hist(substep_2_array, bins = 40)
    mean_value2 = np.mean(substep_2_array)
    plt.axvline(mean_value2, color = 'black')
    plt.legend([f'Mean substep 2 size = {mean_value2:.2f} nm'], loc='upper left')
    plt.savefig(file_list[0][:-38] + 'Substep2_hist')
    plt.close()
    plt.hist(totalstep_array, bins = 40)
    mean_value3 = np.mean(totalstep_array)
    plt.axvline(mean_value3, color = 'black')
    plt.legend([f'Mean total step size = {mean_value3:.2f} nm'], loc='upper left')
    plt.savefig(file_list[0][:-38] + 'Totalstep_hist')
    plt.close()
    

    

def lifetime_listgen(file_list, freq=250000, front_pad=4000, back_pad=4000)
    lifetime_list = []
    for file in M493I_1uM_2:
        indiv_event = np.load(file)
        lifetime = ((len(indiv_event)/freq)-(2*front_pad/freq))
        lifetime_list.append(lifetime)
    return lifetime_list


def find_event_times(folder_path, keyword='filteredEventTimes', extension='.npy'):
    file_pattern = '*'
    
    if keyword:
        file_pattern += keyword + '*'
    
    if extension:
        file_pattern += extension
    
    file_list = glob.glob(os.path.join(folder_path, '**', file_pattern), recursive=True)
    
    return file_list


def reattachmentrate(folder, freq = 250000):
    reattach_array = np.array([], dtype=np.int32)
    for file in folder:
        eventtime_array = np.load(file)
        for i in range(len(eventtime_array)-1):
            time_bet = (eventtime_array[i+1][0] - eventtime_array[i][1])/freq
            reattach_array = np.append(reattach_array, time_bet)
    return reattach_array

def bell_equation(F, k1, d):
    F_k = k1*np.exp((-F*d)/4.1)
    return F_k