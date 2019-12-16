import math
import random
from deap import tools


def sel_random(individuals, buckets, k):
    chosen_inds = []
    chosen_buckets = []
    for i in range(k):
        r = random.randrange(len(individuals))
        chosen_inds.append(individuals[r])
        chosen_buckets.append(buckets[r])

    return chosen_inds, chosen_buckets


def sel_least_complex(individuals, complexity_func):
    if len(individuals) == 1:
        return individuals[0]
    else:
        lowest_complexity = math.inf
        for ind in individuals:
            complexity = complexity_func(ind)
            if complexity < lowest_complexity:
                lowest_complexity = complexity
                least_complex = ind
        return least_complex


def selElitistAndTournament(individuals, k, tournsize, elitism):
    return tools.selBest(individuals, elitism) + tools.selTournament(individuals, k-elitism, tournsize)


def parsimony_tournament(individuals, k, tournsize, toolbox):
        chosen = []
        ordered = sorted(individuals, key=lambda x: x.fitness.values[0])
        buckets = toolbox.bucket(ordered)

        for i in range(k):
            aspirants, asp_buckets = sel_random(ordered, buckets, tournsize)
            max_bucket = max(*asp_buckets)

            # add all aspirants in max bucket to final group.
            final = []
            for j in range(len(aspirants)):
                if asp_buckets[j] == max_bucket:
                    final.append(aspirants[j])

            # select least complex from highest-order individuals
            chosen.append(sel_least_complex(final, toolbox.eval_complexity))

        return chosen


def ratio_bucketing(r, individuals):
    buckets = [0]
    curr_in_bucket = 1
    bucket = 0
    length = len(individuals)

    bucket_size = math.ceil(1/r * length)

    for i in range(1, len(individuals)):
        if curr_in_bucket == 0 and individuals[i].fitness.values[0] == individuals[i-1].fitness.values[0]:
            buckets.append(buckets[i-1])
        else:
            buckets.append(bucket)
            curr_in_bucket += 1

            if curr_in_bucket >= bucket_size:
                bucket += 1
                curr_in_bucket = 0
                bucket_size = math.ceil(1/r * (length - i))

    return buckets


def direct_bucketing(b, individuals):
    per_bucket = math.ceil(len(individuals) / b)

    # start with first element in first bucket
    buckets = [0]
    curr_in_bucket = 1
    bucket = 0

    for i in range(1, len(individuals)):
        # if previous bucket is full but fitness is equal to the highest from there, add it to maintain consistency
        if curr_in_bucket == 0 and individuals[i].fitness.values[0] == individuals[i-1].fitness.values[0]:
            buckets.append(buckets[i-1])
        else:
            buckets.append(bucket)
            curr_in_bucket += 1

            if curr_in_bucket >= per_bucket:
                bucket += 1
                curr_in_bucket = 0

    return buckets


def no_bucketing(b, individuals):

    buckets = [0]
    bucket = 0

    for i in range(1, len(individuals)):
        if individuals[i].fitness.values[0] == individuals[i-1].fitness.values[0]:
            buckets.append(bucket)
        else:
            bucket += 1
            buckets.append(bucket)

    return buckets
