from scikits import audiolab
import numpy
sin_a3 = audiolab.Sndfile('saw_a1.aiff')

print sin_a3.nframes

window_size = 4096 #256

num_windows = sin_a3.nframes / window_size # Truncated int

#shuffled_indices = numpy.random.permutation(num_windows)

window = numpy.hamming(window_size)

file_data = [window * sin_a3.read_frames(window_size) for _ in range(num_windows)]
output_data = numpy.random.permutation(file_data)

output_file = audiolab.Sndfile(
    'shuffled_windowed_shifted_saw_a1_'+str(window_size)+'.aiff',
    'w',
    audiolab.Format(type='aiff', encoding = sin_a3.encoding),
    sin_a3.channels,
    sin_a3.samplerate)
#for i in range(num_windows):
#    output_file.write_frames(output_data[i])
for i in range(num_windows - 1):
    output_file.write_frames(output_data[i][window_size/2:] + output_data[i+1][:window_size/2])

output_file.sync()

