from random import randint
from random import random as randdouble
from scikits import audiolab
import numpy as np
import fitness as fitness_metric

num_zero_crossings_per_chunk = ((10,20), (20, 40), (40, 80)) #
fitness_metric.use_rms_centroid()

randint_spread = lambda x: int(round(randdouble()*2*x - x))

class Chromosome:
    def __init__(self, genes):
        self.genes = genes
    
    def crossover(self, other):
        max_crossing_point = min(len(self.genes), len(other.genes))
        crossing_point = randint(0, max_crossing_point)
        child1 = Chromosome(self.genes[:crossing_point] + other.genes[crossing_point:])
        child2 = Chromosome(other.genes[:crossing_point] + self.genes[crossing_point:])
        return (child1, child2)
    
    def mutate(self, max_value):
        index = randint(0, len(self.genes) - 1)
        value = randint(0, max_value)
        self.genes[index] = value
    
    def nudge(self, max_value, max_nudge, nudge_rate):
        for i in range(len(self.genes)):
            if randdouble() < nudge_rate:
                new_value = self.genes[i] + randint_spread(max_nudge)
                new_value = max(new_value, 0)
                new_value = min(new_value, max_value)
                self.genes[i] = new_value

def load_sound_file(filepath):
    sf = audiolab.Sndfile(filepath)
    frames = sf.read_frames(sf.nframes)
    # Necessary if reading a stereo or higher file
    #frames = np.array([x[0] for x in frames])
    return frames

# Splits by ascending zero crossings
def split_by_zero_crossings(arr, num_to_skip=(0,0)):
    chunks = []
    current_chunk = [arr[0]]
    skipped = 0
    to_skip = randint(*num_to_skip)
    for i in range(1, len(arr)):
        val = arr[i]
        if (val >= 0.0 and current_chunk[-1] < 0.0):
            if skipped == to_skip:
                chunks.append(np.array(current_chunk))
                current_chunk = []
                skipped = 0
                to_skip = randint(*num_to_skip)
            else:
                skipped += 1
        current_chunk.append(val)
    # Don't append last chunk.  It's not a full zero crossing so we can't use it.
    return chunks

class MasterChromosome:
    def __init__(self, source_file, target_file):
        sf = audiolab.Sndfile(source_file)
        self.encoding = sf.encoding
        self.channels = 1 # We only load first channel
        self.samplerate = sf.samplerate
        sf.close()
        
        self.source = load_sound_file(source_file)
        self.target = load_sound_file(target_file)
        self.split_source = []
        for split_range in num_zero_crossings_per_chunk:
            print split_range
            self.split_source += split_by_zero_crossings(self.source, split_range)
        self.max_gene = len(self.split_source) - 1
        self.ideal_length = len(self.target)
    
    def generate(self):
        genes = []
        running_length = 0
        while running_length < self.ideal_length:
            genes += [randint(0, self.max_gene)]
            running_length += len(self.split_source[genes[-1]])
        return Chromosome(genes)
        return Chromosome(range(self.max_gene))
    
    def anti_fitness(self, c):
        # We have a distance metric, which we want to minimize.
        # Call it "anti-fitness" i.e. higher anti-fitness = worse candidate.
        return fitness_metric.distance(self.target, self.to_numpy(c))
    
    def to_numpy(self, c):
        return np.concatenate([self.split_source[x] for x in c.genes])
        
    # Precondition: filepath ends in '.aiff', is a valid filepath
    def to_sound_file(self, c, filepath):
        output_file = audiolab.Sndfile(
            filepath,
            'w',
            audiolab.Format(type='aiff', encoding=self.encoding),
            self.channels,
            self.samplerate)
        output_file.write_frames(self.to_numpy(c))
        output_file.close()

class Population:
    # Precondition: turnover size is less than half of population size
    def __init__(self, source_file, target_file, population_size, turnover_size, mutation_rate, nudge_rate):
        self.master_chromosome = MasterChromosome(source_file, target_file)
        self.population_size = population_size
        # Make turnover size even
        self.turnover_size = (int(turnover_size) / 2) * 2
        self.mutation_rate = mutation_rate
        self.nudge_rate = nudge_rate
        self.populate()
        self.sort_population()
    
    def populate(self):
        self.chromosomes = []
        for _ in range(self.population_size):
            self.chromosomes += [self.master_chromosome.generate()]
    
    def sort_population(self):
        self.chromosomes = sorted(self.chromosomes, key=lambda x: self.master_chromosome.anti_fitness(x))
    
    # Pre- and post-condition: self.chromosomes is sorted by anti-fitness
    def run_generation(self):
        evaluated_chromosomes = self.chromosomes[:self.population_size - self.turnover_size]
        
        # Crossover works best on two that are both pretty good.
        # Currently crossing 0 and 1, 2 and 3, etc. so that none
        #   get their offspring into the pool more than once.
        for i in range(self.turnover_size / 2):
            parentA = evaluated_chromosomes[(2 * i)]
            parentB = evaluated_chromosomes[(2 * i) + 1]
            evaluated_chromosomes += parentA.crossover(parentB)
        
        if len(evaluated_chromosomes) != self.population_size:
            raise Exception("Noooo we lost or gained chromosomes")
        
        for i in range(self.population_size):
            if randdouble() < self.mutation_rate:
                evaluated_chromosomes[i].mutate(self.master_chromosome.max_gene)
        
        self.chromosomes = evaluated_chromosomes
        self.sort_population()
        
        best = self.chromosomes[0]
        best_anti_fitness = self.master_chromosome.anti_fitness(best)
        return best, best_anti_fitness
    
    def nudge_population(self, max_nudge):
        for c in self.chromosomes:
            c.nudge(self.master_chromosome.max_gene, max_nudge, self.nudge_rate)
        
        self.sort_population()
        best = self.chromosomes[0]
        best_anti_fitness = self.master_chromosome.anti_fitness(best)
        return best, best_anti_fitness
        
def run_genetic_algorithm(source_file, target_file):
    population_size = 100
    turnover_size = 40
    mutation_rate = 0.5
    nudge_rate = 0.1
    nudge_threshold = 2
    p = Population(source_file, target_file, population_size, turnover_size, mutation_rate, nudge_rate)
    
    generations_without_change = 0
    num_nudges = 0
    max_nudge_addition = 20
    previous_anti_fitness = 1.0
    for j in range(4):
        for k in range(100):
            (best, best_anti_fitness) = p.run_generation()
            print j * 100 + k, ':\t', best_anti_fitness
            
            percent_change = (previous_anti_fitness - best_anti_fitness) / previous_anti_fitness
            if percent_change == 0.0:
                generations_without_change += 1
            else:
                generations_without_change = 0
            if generations_without_change >= nudge_threshold:
                num_to_nudge = generations_without_change + max(max_nudge_addition - num_nudges, 0)
                print "\t\t\tNudging population", num_to_nudge
                p.nudge_population(num_to_nudge)
                num_nudges += 1
                p.nudge_rate += 0.005
            previous_anti_fitness = best_anti_fitness
        filename = 'best_sp_rms_centroid_nudged'+str(num_zero_crossings_per_chunk)+'_'+str(j)+'.aiff'
        p.master_chromosome.to_sound_file(best, filename)

s = 'early_experiments/sin_a3.aiff'
r = run_genetic_algorithm
ss = 'synth_source.aiff'
st = 'violin_target.aiff'
sst = 'violin_spiccato_target.aiff'
sls = 'sax_long.aiff'
sst2 = 'violin_spiccato_target_2.aiff'
sv = 'violin_target_original.aiff'

if __name__ == "__main__":
    r(st, sst)
