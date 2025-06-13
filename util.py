import os, bz2, gzip, pickle

def check_path_created(path):
    """
    Checks whether the folders in the path are created. If not, it creates the folders.
    """
    filename = os.path.basename(path)
    if '.' in filename:
        # the file path ends with a folder; not a file!
        only_directories = os.path.dirname(path)
    else:
        only_directories = path
        # we will assume the path ends with a file name (not necessarily an existing file)
    if only_directories != '': # no directories in the path
        os.makedirs(only_directories, exist_ok=True)

def save_pickle(obj, path):
    # check if all folders in the path exist; if not create them
    check_path_created(path)

    # based on the file extension, we support pickling with compression and pickling with no compression
    filepath, extension = os.path.splitext(path)
    if extension == '.bz2pkl': # use bz2 compression
        with bz2.BZ2File(path, 'wb') as f: 
            pickle.dump(obj, f)
    elif extension == '.gzpkl': # use gzip compression
        with gzip.open(path, 'wb') as f:
            pickle.dump(obj, f)
    else: # use classic pickle with no compression
        with open(path, 'wb') as fil:
            pickle.dump(obj, fil)

def load_pickle(path):
    filepath, extension = os.path.splitext(path)
    print (f'the extension is {extension}')
    if extension == '.bz2pkl':
        with bz2.BZ2File(path, 'rb') as f:
            obj = pickle.load(f)
    elif extension == '.gzpkl':
        with gzip.open(path, 'rb') as f:
            obj = pickle.load(f)
    else:
        with open(path, 'rb') as fil:
            obj = pickle.load(fil)
    
    return obj

def has_method(inst,func_name):
    """ 
    Checks whether an object (instance) has a method with a specific name.
    """
    attr = getattr(inst, func_name, None)
    if attr is not None:
        if callable(attr):
            return True
    return False