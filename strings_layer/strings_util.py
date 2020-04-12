import subprocess

def get_strings_with_offset(file_path):
    """ Obtains strings from a file
    
    Arguments:
    file_path -- path from the binary to get strings from

    Returns:
    strings_with_offset -- strings found. The first word of each string is the offset
        on the file
    """
    #EXECUTE STRINGS COMMAND
    process = subprocess.Popen(["strings","--radix=x",file_path],  stdout=subprocess.PIPE)
    strings_with_offset=[]
    while True:
        line =process.stdout.readline()
        if not line:
            break
        strings_with_offset.append(line.strip().decode("UTF-8"))
    return strings_with_offset