'''
Implementation of the solution using neural networks and genetic algorithm.
'''
from nn import *
from implementations import *
from helpers import *

import numpy as np

import sys
from multiprocessing import Pool
from functools import partial


# Columns to ignore from the dataset
DROP_COLUMNS = [
    "FRUITJU1", "_AIDTST3", "HIVTST6", "_FRUTSUM", "FRUTDA1_", "_FRUTSUM", "_BMI5",  "HIVTST6",
    "CTELENUM",
    "PVTRESD1",
    "COLGHOUS",
    "STATERES",
    "CELLFON3",
    "LADULT",
    "NUMADULT",
    "NUMMEN",
    "NUMWOMEN",
    "CTELNUM1",
    "CELLFON2",
    "CADULT",
    "PVTRESD2",
    "CCLGHOUS",
    "CSTATE",
    "LANDLINE",
    "HHADULT",
    "POORHLTH",
    "BPMEDS",
    "ASTHNOW",
    "DIABAGE2",
    "NUMHHOL2",
    "NUMPHON2",
    "CPDEMO1",
    "PREGNANT",
    "SMOKDAY2",
    "STOPSMK2",
    "LASTSMK2",
    "AVEDRNK2",
    "DRNK3GE5",
    "MAXDRNKS",
    "EXRACT11",
    "EXEROFT1",
    "EXERHMM1",
    "EXRACT21",
    "EXEROFT2",
    "EXERHMM2",
    "LMTJOIN3",
    "ARTHDIS2",
    "ARTHSOCL",
    "JOINPAIN",
    "FLSHTMY2",
    "IMFVPLAC",
    "HIVTSTD3",
    "WHRTST10",
    "PDIABTST",
    "PREDIAB1",
    "INSULIN",
    "BLDSUGAR",
    "FEETCHK2",
    "DOCTDIAB",
    "CHKHEMO3",
    "FEETCHK",
    "EYEEXAM",
    "DIABEYE",
    "DIABEDU",
    "CAREGIV1",
    "CRGVREL1",
    "CRGVLNG1",
    "CRGVHRS1",
    "CRGVPRB1",
    "CRGVPERS",
    "CRGVHOUS",
    "CRGVMST2",
    "CRGVEXPT",
    "VIDFCLT2",
    "VIREDIF3",
    "VIPRFVS2",
    "VINOCRE2",
    "VIEYEXM2",
    "VIINSUR2",
    "VICTRCT4",
    "VIGLUMA2",
    "VIMACDG2",
    "CIMEMLOS",
    "CDHOUSE",
    "CDASSIST",
    "CDHELP",
    "CDSOCIAL",
    "CDDISCUS",
    "WTCHSALT",
    "LONGWTCH",
    "DRADVISE",
    "ASTHMAGE",
    "ASATTACK",
    "ASERVIST",
    "ASDRVIST",
    "ASRCHKUP",
    "ASACTLIM",
    "ASYMPTOM",
    "ASNOSLEP",
    "ASTHMED3",
    "ASINHALR",
    "HAREHAB1",
    "STREHAB1",
    "CVDASPRN",
    "ASPUNSAF",
    "RLIVPAIN",
    "RDUCHART",
    "RDUCSTRK",
    "ARTTODAY",
    "ARTHWGT",
    "ARTHEXER",
    "ARTHEDU",
    "TETANUS",
    "HPVADVC2",
    "HPVADSHT",
    "SHINGLE2",
    "HADMAM",
    "HOWLONG",
    "HADPAP2",
    "LASTPAP2",
    "HPVTEST",
    "HPLSTTST",
    "HADHYST2",
    "PROFEXAM",
    "LENGEXAM",
    "BLDSTOOL",
    "LSTBLDS3",
    "HADSIGM3",
    "HADSGCO1",
    "LASTSIG3",
    "PCPSAAD2",
    "PCPSADI1",
    "PCPSARE1",
    "PSATEST1",
    "PSATIME",
    "PCPSARS1",
    "PCPSADE1",
    "PCDMDECN",
    "SCNTMNY1",
    "SCNTMEL1",
    "SCNTPAID",
    "SCNTWRK1",
    "SCNTLPAD",
    "SCNTLWK1",
    "SXORIENT",
    "TRNSGNDR",
    "RCSGENDR",
    "RCSRLTN2",
    "CASTHDX2",
    "CASTHNO2",
    "EMTSUPRT",
    "LSATISFY",
    "ADPLEASR",
    "ADDOWN",
    "ADSLEEP",
    "ADENERGY",
    "ADEAT1",
    "ADFAIL",
    "ADTHINK",
    "ADMOVE",
    "MISTMNT",
    "ADANXEV",
    "MSCODE",
    "_CRACE1",
    "_CPRACE",
    "_CLLCPWT",
    "_DUALCOR",
    "METVL11_",
    "METVL21_",
    "ACTIN11_",
    "ACTIN21_",
    "PADUR1_",
    "PADUR2_",
    "PAFREQ1_",
    "PAFREQ2_",
    "_MINAC11",
    "_MINAC21",
    "PAMIN11_",
    "PAMIN21_",
    "PA1MIN_",
    "PAVIG11_",
    "PAVIG21_",
    "PA1VIGM_",
    "_FLSHOT6",
    "_PNEUMO2"
]


def _worker(w, x_train, y_train, nn_shape):
    return (NN(nn_shape, w).error(x_train, y_train), w)


def gen_alg(x_train, y_train, nn_shape, popsize, elitism, mutation_prob, mutation_scale, iter):
    '''
    Genetic algorithm for optimization.

    Args:
        x_train: training dataset
        y_train: training dataset outputs
        nn_shape: neural network shape
        popsize: population size
        elitism: elitism
        mutation_prob: probability of mutation
        mutation_scale: standard deviation of mutations
        iter: number of iterations

    Returns:
        neural network weights received from the optimization algorithm.
    '''
    rng = np.random.default_rng()

    population = [NN(nn_shape).get_weight_vector() for i in range(popsize)]
    population = [(NN(nn_shape, x).error(x_train, y_train), x) for x in population]
    population.sort(key=lambda x: x[0])

    fitness_func = lambda x: 10000 / (1 + x)
    total_fitness = sum(fitness_func(x[0]) for x in population)

    f = partial(_worker, x_train=x_train, y_train=y_train, nn_shape=nn_shape)

    with Pool(processes=16) as pool:
        for iter_num in range(1, iter + 1):
            print(f"running iter {iter_num} / {iter}")
            new_population = [population[i][1] for i in range(elitism)]

            if iter_num % 2000 == 0:
                print(f"[Train error @{iter_num}]: {population[0][0]}")

            while len(new_population) < popsize:
                i1, i2 = rng.choice(
                    list(range(len(population))),
                    2,
                    p=[fitness_func(x[0]) / total_fitness for x in population],
                    replace=False
                )

                t1, t2 = population[i1], population[i2]

                parent_1, parent_2 = t1[1], t2[1]
                w = (t1[1] + t2[1]) / 2
                for i in range(len(w)):
                    if rng.random() <= mutation_prob:
                        w[i] += rng.normal(0.0, mutation_scale) 

                new_population.append(w)
            
            population = new_population
            population = pool.map(f, population)
            population.sort(key=lambda x: x[0])

            total_fitness = sum(fitness_func(x[0]) for x in population)
            
    return population[0][1]


def main():
    '''
    Entry point.
    '''

    # Load the data.
    PATH_TO_DATASET = "data/dataset"
    x_train, x_test, y_train, train_ids, test_ids = load_csv_data(
        PATH_TO_DATASET, NaNstrat="fill", remove_columns=DROP_COLUMNS
    )

    # Take only a prefix of the rows, it's too slow with the entire dataset.
    MAX_ROWS = 70000
    x_train = x_train[:MAX_ROWS]
    y_train = y_train[:MAX_ROWS]
    
    # Balance the dataset because no. of 1s is way smaller than no. of 0s.
    x_train, y_train = balance_dataset(x_train, y_train, 2)

    # Preprocess the train dataset.
    x_train, mask = preprocess_structural(x_train, ones=False)
    x_train = preprocess_unstructural(x_train)
    
    # Preprocess the test dataset.
    x_test = x_test[:, mask]
    x_test = preprocess_unstructural(x_test)

    # Apply dimensionality reduction.
    MAX_PCA_DIMS = 35
    x_train, x_mean, top_comps = pca_fit(x_train, MAX_PCA_DIMS)
    x_test = pca_transform(x_test, x_mean, top_comps)

    # Split the training dataset for evaluation.
    tx_train_split, tx_test_split, y_train_split, y_test_split = split_data(x_train, y_train)

    y_train_split = y_train_split.reshape((-1, 1))
    y_test_split = y_test_split.reshape((-1, 1))
    y_train_split = (1 + y_train_split) / 2
    y_test_split = (1 + y_test_split) / 2

    print(np.shape(tx_train_split))
    print(np.shape(y_train_split))

    D = tx_train_split.shape[1]
    NUM_ITER = 1000

    # Run the genetic algorithm.
    nn_shape = [D, 8, 8, 8, 8, 1]
    w = gen_alg(
        tx_train_split, y_train_split,
        nn_shape,
        16, 1, 0.1, 0.1, NUM_ITER
    )

    # Calculate the confusion matrix and graph it.
    y_pred = (NN(nn_shape, w).evaluate(tx_test_split) > 0.5).astype(np.float128)
    print(f"predicted {y_pred[:16]}")
    print(f"true {y_test_split[:16]}")

    # Evaluate on actual test dataset for submission.
    y_submit = (NN(nn_shape, w).evaluate(x_test) > 0.5).astype(np.int32) * 2 - 1
    create_csv_submission(test_ids, y_submit, "y_pred.csv")


if __name__ == "__main__":
    main()