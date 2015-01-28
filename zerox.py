from pysndfile import *

def find_zerox_pysndfile(
    audio_file,
    position,
    search_before = False
):
    """
    Returns the index of the sample in the PySndfile that is the nearest
    zero-crossing to the position provided
    position:       Sample to begin searching for zero-crossing from
    search_before:  whether the algorithm should search for the zero-crossing
    before or after the position
    If no zerocrossing is found then None is returned
    """
    print audio_file.frames()
    def pos_neg_zero(x):
        if x > 0:
            return 0
        elif x < 0:
            return 1
        elif x == 0:
            return 2

    chunksize = 4096
    #save original seek position to reset after
    original_seek = audio_file.seek(0, 1)
    zero_found = None
    frames_read = 0
    if not search_before:
        #algorithm for searching after the position
        while zero_found is None:
            #if the search has reached the end of the file,
            #only read the number of samples left
            if chunksize + position + frames_read > audio_file.frames():
                chunksize = audio_file.frames() - (position + frames_read)
                #if there are no samples left to read then break
                if chunksize == 0:
                    break
            #create array of samples
            audio_file.seek(position + frames_read, 0)
            chunk = audio_file.read_frames(chunksize)
            #search for the zero crossing by comparing each sample to its
            #predecessor
            prev_j = chunk[0]
            print chunk
            print frames_read
            for index, j in enumerate(chunk, start = 0):
                if pos_neg_zero(j) is not pos_neg_zero(prev_j):
                    zero_found = position + frames_read + index - 1
                    break
                prev_j = j
            
            if zero_found is None:
                frames_read += chunksize
            else:
                break
    
    else:
        #algorithm for searching before the sample
        while zero_found is None:
            #if the search has reached the begining of the file,
            #only read the number of samples left
            if position - (chunksize + frames_read) < 0:
                chunksize = position - frames_read
                #if there are no samples left to read then break
                if chunksize == 0:
                    break
            #create array of samples
            print chunksize
            print position - (frames_read + chunksize)
            audio_file.seek(position - (frames_read + chunksize), 0)
            chunk = audio_file.read_frames(chunksize)
            print frames_read
            print chunk
            #search for the zero crossing by comparing each sample to its
            #predecessor
            prev_j = chunk[-1]
            for index, j in reversed(list(enumerate(chunk))):
                if pos_neg_zero(j) is not pos_neg_zero(prev_j):
                    zero_found = position - (frames_read + index + 1)
                    break
                prev_j = j
            if zero_found is None:
                frames_read += chunksize
            else:
                break

    #reset audio seeker to position
    audio_file.seek(original_seek, 0)
    return zero_found

def find_zero_crossing_list(sample_list,
                            position,
                            search_before = False,
                            ):
    """
    Return the index in samples of the nearest point where the signal crosses 0.
    Function searches before position given if search_before is true.
    If no value is found then the first/last sample in the list is
    returned

    """

    def pos_neg_zero(x):
        if x > 0:
            return 0
        elif x < 0:
            return 1
        elif x == 0:
            return 2

    zero_found = False
    if search_before:
        i = 1
        prev_i = 0
        while zero_found == False:
            try:
                if pos_neg_zero(sample_list[position - i]) is not pos_neg_zero(sample_list[position - prev_i]):
                    return position - i
                else:
                    prev_i = i
                    i += 1
            except IndexError:
                return 0
    else:
        i = 1
        prev_i = 0
        while zero_found == False:
            try:
                if pos_neg_zero(sample_list[position + i]) is not pos_neg_zero(sample_list[position + prev_i]):
                    return position + i
                else:
                    prev_i = i
                    i += 1
            except IndexError:
                return len(sample_list) - 1

