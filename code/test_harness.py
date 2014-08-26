from scikits import audiolab
import numpy as np
import fitness
import random

sin_a3 = audiolab.Sndfile('early_experiments/sin_a3.aiff')
sin_a3_data = sin_a3.read_frames(sin_a3.nframes)

comparison = audiolab.Sndfile('early_experiments/zero_chunks_sin_a3.aiff')
comparison_data = comparison.read_frames(comparison.nframes)

print "Testing spectral centroid (should be ~220):"
for offset in [50, 100, 110, 120, 150]:
    print fitness.spectral_centroid(sin_a3_data[offset/4 * fitness.window_size : (offset/4 + 1) * fitness.window_size])

print "Testing whether d(x,x) == 0:"
print fitness.distance(sin_a3_data, sin_a3_data)

print "Testing distance between two similar ish files:"
print fitness.distance(sin_a3_data, comparison_data)

print "Testing distance between file and mutated file:"

frames_to_mutate = 100
sin_a3_mutated = np.array(sin_a3_data)
rand1 = random.randint(frames_to_mutate, sin_a3.nframes - 4 * frames_to_mutate)
rand2 = random.randint(rand1 + frames_to_mutate, sin_a3.nframes - 2 * frames_to_mutate)

sin_a3_mutated[rand2:rand2 + frames_to_mutate] = sin_a3_data[rand1:rand1 + frames_to_mutate]
sin_a3_mutated[rand1:rand1 + frames_to_mutate] = sin_a3_data[rand2:rand2 + frames_to_mutate]

print fitness.distance(sin_a3_data, sin_a3_mutated)
