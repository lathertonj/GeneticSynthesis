from scikits import audiolab
import numpy

def close_to_zero(val):
    return abs(val) <= pow(10, -3)

sin_a3 = audiolab.Sndfile('saw_a1.aiff')

print sin_a3.nframes

window_size = 1024 * 4 #256

num_windows = sin_a3.nframes / window_size # Truncated int

#shuffled_indices = numpy.random.permutation(num_windows)

window = numpy.hamming(window_size)

def count_zeros(audio_file):
    count = 0
    for _ in range(audio_file.nframes):
        val = audio_file.read_frames(1)
        if close_to_zero(val):
            count += 1

    percent_zero = 1.0 * count / audio_file.nframes

    print percent_zero

chunks = []
current_chunk = [sin_a3.read_frames(1)[0]]
for _ in range(sin_a3.nframes - 1):
    val = sin_a3.read_frames(1)[0]
    if (val >= 0.0 and current_chunk[-1] < 0.0):# or (val < 0 and current_chunk[-1] >= 0):    ###close_to_zero(val): #and len(current_chunk) > 0 and current_chunk[-1] < val:
        chunks.append(numpy.array(current_chunk))
        current_chunk = []
    current_chunk.append(val)

output_data = numpy.random.permutation(chunks)


output_file = audiolab.Sndfile(
     'zero_chunks_twoways_saw_a1.aiff',
     'w',
     audiolab.Format(type='aiff', encoding = sin_a3.encoding),
     sin_a3.channels,
     sin_a3.samplerate)
for i in range(len(output_data)):
    output_file.write_frames(output_data[i])
# for i in range(num_windows - 1):
#     output_file.write_frames(output_data[i][window_size/2:] + output_data[i+1][:window_size/2])

output_file.sync()