from scikits import audiolab
import numpy
#import pyfftw
sin_a3 = audiolab.Sndfile('sin_a3.aiff')

print sin_a3.nframes

window_size = 4096 #256

num_windows = sin_a3.nframes / window_size # Truncated int

window = numpy.hamming(window_size)

file_data = [window * sin_a3.read_frames(window_size) for _ in range(num_windows)]

#file_data = [pyfftw.n_byte_align(x, 16) for x in file_data]
ffts = [numpy.fft.fft(x) for x in file_data] #pyfftw.interfaces.numpy_fft.fft(x)
output_ffts = numpy.random.permutation(ffts)
output_data = [numpy.fft.ifft(x) for x in output_ffts] #pfftw.interfaces.numpy_fft.ifft(x)

output_file = audiolab.Sndfile(
    'shuffled_fft_sin_a3_'+str(window_size)+'.aiff',
    'w',
    audiolab.Format(type='aiff', encoding = sin_a3.encoding),
    sin_a3.channels,
    sin_a3.samplerate)
for i in range(num_windows):
    output_file.write_frames(numpy.array(output_data[i], dtype='float64'))
#for i in range(num_windows - 1):
#    output_file.write_frames(output_data[i][window_size/2:] + output_data[i+1][:window_size/2])

output_file.sync()
