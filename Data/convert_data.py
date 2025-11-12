
import os
from tempfile import mkdtemp
import numpy as np
from sklearn.preprocessing import OneHotEncoder

def load_files_memory_efficient(directory, type_="standard", temp_dir=None,pressures=""):
    if temp_dir is None:
        temp_dir = mkdtemp()
    
    # First pass: collect labels and create mapping
    file_paths = []
    labels=['Carpet', 'LacedMatt', 'wool', 'Cork', 'Felt', 'LongCarpet', 'cotton', 'Plastic', 'Flat', 'foamf', 'foamg', 'bubble', 'foame', 'jeans', 'Leather']
    keys = {labels[i].lower():i for i in range(len(labels))}
    for item in os.listdir(directory):
        full_path = os.path.join(directory, item)
        if not os.path.isfile(full_path):
            continue
            
        filename = os.path.basename(full_path)
        if pressures!="" and pressures in filename:
            newlabel = filename.split("_")[2].lower()
            if newlabel not in keys:
                keys[newlabel] = len(keys)
            file_paths.append(full_path)
        else:
            newlabel = filename.split("_")[1].lower()
            if newlabel not in keys:
                keys[newlabel] = len(keys)
            file_paths.append(full_path)
        
    print(keys,file_paths)
    # Initialize memory-mapped arrays for final output
    sample_file = np.load(file_paths[0])
    if type_ == "circle" or type_ == "pressure":
        sample_file = sample_file[:, :, :len(np.arange(10, 100, 10))]
        sample_shape = (1 * 2 * len(np.arange(10, 100, 10)), 50, *sample_file.shape[4:])
    else:
        sample_file = sample_file
        sample_shape = (1 * 2 * 10 * 10,*sample_file.shape[4:])
    print("expected shape",sample_shape)
    # Create memory-mapped arrays for data and labels
    data_memmap_path = os.path.join(temp_dir, 'data_memmap.dat')
    label_memmap_path = os.path.join(temp_dir, 'label_memmap.dat')
    
    # Calculate total size needed
    total_samples = 0
    total_y=0
    for file_path in file_paths:
        try:
            data = np.load(file_path)
            if type_ == "circle" or type_ == "pressure":
                data = data[:, :, :20].reshape((20, 20, 355, 328))
                total_samples += np.prod(data.shape)
            else:
                data = data.reshape((200, 20, 355, 328))
                product = np.prod(data.shape)
                total_samples += product
        except ValueError:
            print("issue with",file_path)
            file_paths.remove(file_path)
        
        
    print("required size:",total_samples)
    # Initialize memmap files
    if type_=="pressure" or type_=="circle":
        data_memmap = np.memmap(data_memmap_path, dtype=np.uint8, mode='w+', 
                           shape=(total_samples))
    else:
        data_memmap = np.memmap(data_memmap_path, dtype=np.uint8, mode='w+', shape=(total_samples,))

    labels=[]
    # Second pass: process files one at a time
    current_idx = 0
    ones_made_id=0
    for file_path in file_paths:
        filename = os.path.basename(file_path)
        #print("Processing:", filename)
        print("processing...",filename)
        data = np.load(file_path).astype(np.uint8)
        if type_=="pressure" or type_=="circle":
            newlabel = filename.split("_")[2].lower()
        else:
            newlabel = filename.split("_")[1].lower()
        try:
            num = keys[newlabel]
        except:
            num = keys[newlabel[1:]+str(newlabel[0])] #if forams are labelled incorrectly
        
        if type_ == "circle" or type_ == "pressure":
            data = data[:, :, :20]
            data = data.reshape((20, 20, 355, 328))
            num_samples = np.prod(data.shape)
        else:
            data=data.reshape((200, 20, 355, 328))
            num_samples = np.prod(data.shape)
        
        # Write to memmap
        try:
            data_memmap[current_idx:current_idx + num_samples] = data.flatten()
            labels.append([num for i in range(200)])
            ones_made_id+=1
        except:
            pass
        
        current_idx += num_samples
    
    # Flush changes to disk
    data_memmap.flush()
    
    # Reload as regular arrays (or keep as memmap if you prefer)
    final_data = np.array(data_memmap)
    final_labels = np.array(labels).flatten()
    
    # Clean up temporary files
    del data_memmap
    try:
        os.remove(data_memmap_path)
    except:
        pass
    final_data=final_data.reshape((ones_made_id*data.shape[0], 20, 355, 328))

    encoder = OneHotEncoder()
    final_labels = encoder.fit_transform(final_labels.reshape((-1,1)))
    return final_data, final_labels
