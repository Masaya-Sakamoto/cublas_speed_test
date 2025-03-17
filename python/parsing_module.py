def parse_output(output):
    lines = output.strip().split('\n')
    if len(lines) != 1:
        raise ValueError("Unexpected output format")
    
    values = lines[0].split(',')
    if len(values) != 6:
        raise ValueError("Unexpected number of values in output")
    
    execution_time_avg = float(values[0])
    execution_time_err = float(values[1])
    device_copy_time_avg = float(values[2])
    device_copy_time_err = float(values[3])
    host_copy_time_avg = float(values[4])
    host_copy_time_err = float(values[5])
    
    return {
        "execution_time_avg": execution_time_avg,
        "execution_time_err": execution_time_err,
        "device_copy_time_avg": device_copy_time_avg,
        "device_copy_time_err": device_copy_time_err,
        "host_copy_time_avg": host_copy_time_avg,
        "host_copy_time_err": host_copy_time_err
    }
