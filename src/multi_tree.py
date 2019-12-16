import random
import numpy as np

from deap import gp, creator
from deap import tools


def process_data(individual, toolbox, data):
    no_instances = data.shape[0]
    no_trees = len(individual)
    feature_major = data.T
    # [no_trees x no_instances]
    # we do it this way so we can assign rows (constructed features) efficiently.
    result = np.zeros(shape=(no_trees, no_instances))
    for i, expr in enumerate(individual):
        func = toolbox.compile(expr=expr)
        vec = func(*feature_major)
        if (not isinstance(vec, np.ndarray)) or vec.ndim == 0:
            # it decided to just give us a constant back...
            vec = np.repeat(vec, no_instances)
        result[i] = vec

    return result.T


def init_primitives(pset):

    pset.addPrimitive(np.add, 2)
    pset.addPrimitive(np.subtract, 2)
    pset.addPrimitive(np.multiply, 2)
    # pset.addPrimitive(add_abs, 2)
    # pset.addPrimitive(sub_abs, 2)
    pset.addPrimitive(protected_div, 2)
    pset.addPrimitive(np.maximum, 2)
    pset.addPrimitive(np.minimum, 2)
    pset.addPrimitive(mt_if, 3)

    pset.addEphemeralConstant("rand", ephemeral=lambda: random.uniform(-1, 1))


def init_toolbox(toolbox, pset, crossover, n_trees):
    creator.create("Individual", list, fitness=creator.FitnessMax, pset=pset)

    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=0, max_=6)
    toolbox.register("tree", tools.initIterate, gp.PrimitiveTree, toolbox.expr)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.tree, n=n_trees)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    if crossover == "ric":
        toolbox.register("mate", xmate_ric)
    elif crossover == "aic":
        toolbox.register("mate", xmate_aic)
    elif crossover == "sic":
        toolbox.register("mate", xmate_sic)

    toolbox.register("expr_mut", gp.genFull, min_=0, max_=8)
    toolbox.register("mutate", xmut, expr=toolbox.expr_mut)

    # TODO: depth limiting for multi-tree
    # toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=9))
    # toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=9))


def xmate_ric(ind1, ind2):
    i1 = random.randrange(len(ind1))
    i2 = random.randrange(len(ind2))
    ind1[i1], ind2[i2] = gp.cxOnePoint(ind1[i1], ind2[i2])
    return ind1, ind2


def xmate_aic(ind1, ind2):
    assert len(ind1) == len(ind2)
    for i in range(len(ind1)):
        ind1[i], ind2[i] = gp.cxOnePoint(ind1[i], ind2[i])
    return ind1, ind2


def xmate_sic(ind1, ind2):
    i = random.randrange(len(ind1))
    ind1[i], ind2[i] = gp.cxOnePoint(ind1[i], ind2[i])
    return ind1, ind2


def xmut(ind, expr):
    i1 = random.randrange(len(ind))
    indx = gp.mutUniform(ind[i1], expr,pset=ind.pset)
    ind[i1] = indx[0]
    return ind,


def add_abs(a, b):
    return np.abs(np.add(a, b))


def sub_abs(a, b):
    return np.abs(np.subtract(a, b))


def mt_if(a, b, c):
    return np.where(a < 0, b, c)


def protected_div(left, right):
    with np.errstate(divide='ignore', invalid='ignore'):
        x = np.divide(left, right)
        if isinstance(x, np.ndarray):
            x[np.isinf(x)] = 1
            x[np.isnan(x)] = 1
        elif np.isinf(x) or np.isnan(x):
            x = 1
    return x
