import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import concurrent.futures
import time
from sklearn.model_selection import cross_validate, learning_curve, StratifiedKFold
import mlrose


# # Part 1-Discrete Problems

# Define problem domains & fitness functions

# Multi Peak
def multipeak_eq(state):
    x = state[0] + 2*state[1] + 4*state[2] + 8*state[3] + 16*state[4] + 32*state[5] + 64*state[6] + 128*state[7] + 256*state[8] + 512*state[9]
    y = state[10] + 2*state[11] + 4*state[12] + 8*state[13] + 16*state[14] + 32*state[15] + 64*state[16] + 128*state[17] + 256*state[18] + 512*state[19]
    z = max(0, 10*np.sin(x/20+2.3) + 4*(x%10) + 10*np.sin(y/25+1) + 4*(y%15) + 20)
    return z
multipeak_fn = mlrose.CustomFitness(multipeak_eq)


# Knapsack
knapsack_weights = [5, 10, 15, 20, 25, 30, 35, 40, 45]
knapsack_values = np.arange(1, len(knapsack_weights)+1)
knapsack_max_weight = 0.7
knapsack_fn = mlrose.Knapsack(knapsack_weights, knapsack_values, knapsack_max_weight)


# K-colors
kcolor_edges = [(0,1),(0,2),(0,3),(0,4),(0,5),(0,6),(0,7),(0,8),(0,9),(1,4),(1,5),(1,6),(1,7),(2,3),(2,5),(2,7),(3,5),(3,6),(3,7),(3,9),(4,5),(5,6),(5,7),(5,8),(6,7),(7,8),(8,9)]
kcolor_edges = list({tuple(sorted(edge)) for edge in kcolor_edges})
kcolor_fn = mlrose.MaxKColor(kcolor_edges)

def multipeak_prob_wrapper():
    return mlrose.DiscreteOpt(length=20, fitness_fn=multipeak_fn)

def knapsack_prob_wrapper():
    return mlrose.DiscreteOpt(length=len(knapsack_weights), fitness_fn=knapsack_fn)

def kcolor_prob_wrapper():
    return mlrose.DiscreteOpt(length=10, fitness_fn=kcolor_fn, max_val=2)

probs = [{'name':"Multipeak", 'obj': multipeak_prob_wrapper},
         {'name': "Knapsack", 'obj': knapsack_prob_wrapper},
         {'name': "K-Color", 'obj': kcolor_prob_wrapper}]

def hill_climb(prob, **kwargs):
    start = time.time()
    _, best_score, curve, fit_evals = mlrose.random_hill_climb(prob(), curve=True, **kwargs)
    end = time.time()
    return np.array([best_score, len(curve), fit_evals, end-start])

def sim_anneal(prob, **kwargs):
    start = time.time()
    _, best_score, curve, fit_evals = mlrose.simulated_annealing(prob(), curve=True, **kwargs)
    end = time.time()
    return np.array([best_score, len(curve), fit_evals, end-start])

def gen_alg(prob, **kwargs):
    start = time.time()
    _, best_score, curve, fit_evals = mlrose.genetic_alg(prob(), curve=True, **kwargs)
    end = time.time()
    return np.array([best_score, len(curve), fit_evals, end-start])

def mimic(prob, **kwargs):
    start = time.time()
    _, best_score, curve, fit_evals = mlrose.mimic(prob(), curve=True, fast_mimic=True, **kwargs)
    end = time.time()
    return np.array([best_score, len(curve), fit_evals, end-start])


# ## SA param search
runs = 15
temps = np.arange(1,32,2)
sa_results = {}
with concurrent.futures.ThreadPoolExecutor() as exe:
    for prob in probs:
        if prob['name'] not in sa_results: sa_results[prob['name']] = {}
        for temp in temps:
            if temp not in sa_results[prob['name']]: sa_results[prob['name']][temp] = []
            for run in range(runs+1):
                schedule = mlrose.ExpDecay(init_temp=temp, min_temp=0)
                sa_results[prob['name']][temp].append(exe.submit(sim_anneal, prob['obj'], max_iters=500, schedule=schedule))


# ### SA param search plotting
sa_plot_data = {}
for prob in sa_results:
    sa_plot_data[prob] = None
    for temp in sa_results[prob]:
        data = None
        for run in sa_results[prob][temp]:
            # DATA: temp, score, iters, fit_evals, runtime
            if data is None:
                data = np.hstack([temp, run.result()])
            else:
                data = np.vstack([data, np.hstack([temp, run.result()])])
        if sa_plot_data[prob] is None:
            sa_plot_data[prob] = np.mean(data, axis=0)
        else:
            sa_plot_data[prob] = np.vstack([sa_plot_data[prob], np.mean(data, axis=0)])

for prob in sa_plot_data:
    plt.figure(figsize=(15,4))
    plt.suptitle(f'Simulated Annealing Param Search for {prob}', fontsize=15)
    plt.style.use('seaborn')
    plt.tight_layout(pad=2)

    ax1 = plt.subplot(221)
    plt.plot(sa_plot_data[prob][:,0], sa_plot_data[prob][:,1])
    ax1.xaxis.get_label().set_visible(False)
    plt.ylabel('best fitness score', fontsize=13)

    ax2 = plt.subplot(222)
    plt.plot(sa_plot_data[prob][:,0], sa_plot_data[prob][:,2])
    ax2.xaxis.get_label().set_visible(False)
    plt.ylabel('iteration count', fontsize=13)

    ax3 = plt.subplot(223)
    plt.plot(sa_plot_data[prob][:,0], sa_plot_data[prob][:,3])
    plt.xlabel('starting temperature', fontsize=13)
    plt.ylabel("fitness evaluations", fontsize=13)

    ax4 = plt.subplot(224)
    plt.plot(sa_plot_data[prob][:,0], sa_plot_data[prob][:,4])
    plt.xlabel('starting temperature', fontsize=13)
    plt.ylabel("wall time [s]", fontsize=13)

    plt.show()


# ## GA param search
runs = 15
popsizes = np.hstack([2., np.arange(50,500,50)])
ga_results = {}
with concurrent.futures.ThreadPoolExecutor() as exe:
    for prob in probs:
        if prob['name'] not in ga_results: ga_results[prob['name']] = {}
        for popsize in popsizes:
            if popsize not in ga_results[prob['name']]: ga_results[prob['name']][popsize] = []
            for run in range(runs+1):
                ga_results[prob['name']][popsize].append(exe.submit(gen_alg, prob['obj'], max_iters=500, pop_size=popsize, mutation_prob=0.1))


# ### GA param seach plotting
ga_plot_data = {}
for prob in ga_results:
    ga_plot_data[prob] = None
    for popsize in ga_results[prob]:
        data = None
        for run in ga_results[prob][popsize]:
            # DATA: popsize, score, iters, fit_evals, runtime
            if data is None:
                data = np.hstack([popsize, run.result()])
            else:
                data = np.vstack([data, np.hstack([popsize, run.result()])])
        if ga_plot_data[prob] is None:
            ga_plot_data[prob] = np.mean(data, axis=0)
        else:
            ga_plot_data[prob] = np.vstack([ga_plot_data[prob], np.mean(data, axis=0)])

for prob in ga_plot_data:
    plt.figure(figsize=(15,4))
    plt.suptitle(f'Genetic Algorithm Param Search for {prob}', fontsize=15)
    plt.style.use('seaborn')
    plt.tight_layout(pad=2)

    ax1 = plt.subplot(221)
    plt.plot(ga_plot_data[prob][:,0], ga_plot_data[prob][:,1])
    ax1.xaxis.get_label().set_visible(False)
    plt.ylabel('best fitness score', fontsize=13)

    ax2 = plt.subplot(222)
    plt.plot(ga_plot_data[prob][:,0], ga_plot_data[prob][:,2])
    ax2.xaxis.get_label().set_visible(False)
    plt.ylabel('iteration count', fontsize=13)

    ax3 = plt.subplot(223)
    plt.plot(ga_plot_data[prob][:,0], ga_plot_data[prob][:,3])
    plt.xlabel('population size', fontsize=13)
    plt.ylabel("fitness evaluations", fontsize=13)

    ax4 = plt.subplot(224)
    plt.plot(ga_plot_data[prob][:,0], ga_plot_data[prob][:,4])
    plt.xlabel('population size', fontsize=13)
    plt.ylabel("wall time [s]", fontsize=13)

    plt.show()


# ## MIMIC param search
runs = 5
popsizes = np.hstack([2., np.arange(50,500,50)])
mi_results = {}
with concurrent.futures.ThreadPoolExecutor() as exe:
    for prob in probs:
        if prob['name'] not in mi_results: mi_results[prob['name']] = {}
        for popsize in popsizes:
            if popsize not in mi_results[prob['name']]: mi_results[prob['name']][popsize] = []
            for run in range(runs+1):
                mi_results[prob['name']][popsize].append(exe.submit(mimic, prob['obj'], max_iters=500, pop_size=popsize, keep_pct=0.4))


# ### MIMIC param search plotting
mi_plot_data = {}
for prob in mi_results:
    mi_plot_data[prob] = None
    for popsize in mi_results[prob]:
        data = None
        for run in mi_results[prob][popsize]:
            # DATA: popsize, score, iters, fit_evals, runtime
            if data is None:
                data = np.hstack([popsize, run.result()])
            else:
                data = np.vstack([data, np.hstack([popsize, run.result()])])
        if mi_plot_data[prob] is None:
            mi_plot_data[prob] = np.mean(data, axis=0)
        else:
            mi_plot_data[prob] = np.vstack([mi_plot_data[prob], np.mean(data, axis=0)])

for prob in mi_plot_data:
    plt.figure(figsize=(15,4))
    plt.suptitle(f'MIMIC Param Search for {prob}', fontsize=15)
    plt.style.use('seaborn')
    plt.tight_layout(pad=2)

    ax1 = plt.subplot(221)
    plt.plot(mi_plot_data[prob][:,0], mi_plot_data[prob][:,1])
    ax1.xaxis.get_label().set_visible(False)
    plt.ylabel('best fitness score', fontsize=13)

    ax2 = plt.subplot(222)
    plt.plot(mi_plot_data[prob][:,0], mi_plot_data[prob][:,2])
    ax2.xaxis.get_label().set_visible(False)
    plt.ylabel('iteration count', fontsize=13)

    ax3 = plt.subplot(223)
    plt.plot(mi_plot_data[prob][:,0], mi_plot_data[prob][:,3])
    plt.xlabel('population size', fontsize=13)
    plt.ylabel("fitness evaluations", fontsize=13)

    ax4 = plt.subplot(224)
    plt.plot(mi_plot_data[prob][:,0], mi_plot_data[prob][:,4])
    plt.xlabel('population size', fontsize=13)
    plt.ylabel("wall time [s]", fontsize=13)

    plt.show()


# ## All algorithms compared
runs = 15
results = {}
schedule = mlrose.ExpDecay(init_temp=1, min_temp=0)
with concurrent.futures.ThreadPoolExecutor() as exe:
    for prob in probs:
        if prob['name'] not in results: results[prob['name']] = {}
        results[prob['name']]['rh'] = []
        results[prob['name']]['sa'] = []
        results[prob['name']]['ga'] = []
        results[prob['name']]['mi'] = []
        for run in range(runs):
            results[prob['name']]['rh'].append(exe.submit(hill_climb, prob['obj'], max_iters=500))
            results[prob['name']]['sa'].append(exe.submit(sim_anneal, prob['obj'], max_iters=500, schedule=schedule))
            results[prob['name']]['ga'].append(exe.submit(gen_alg, prob['obj'], max_iters=500, pop_size=400, mutation_prob=0.1))
            results[prob['name']]['mi'].append(exe.submit(mimic, prob['obj'], max_iters=500, pop_size=200, keep_pct=0.2))


# ### All algorithms plotting
plot_data = {}
for prob in results:
    if prob not in plot_data: plot_data[prob] = {}
    for alg in results[prob]:
        data = None
        for run in results[prob][alg]:
            # DATA: score, iters, fit_evals, runtime
            if data is None:
                data = run.result()
            else:
                data = np.vstack([data, run.result()])

        plot_data[prob][alg] = np.mean(data, axis=0)

print("Multipeak")
print(plot_data['Multipeak'])
print("Knapsack")
print(plot_data['Knapsack'])
print("K-Color")
print(plot_data['K-Color'])


# # Part 2-Neural Net

# ## load data
phoneme = pd.read_csv("data/phoneme.csv")

def get_x_labels(df, y_label):
    all_labels = list(df)
    all_labels.remove(y_label)
    return all_labels
phoneme_y_label = "Class"
phoneme_x_labels = get_x_labels(phoneme, phoneme_y_label)
phoneme_x = phoneme[phoneme_x_labels]
phoneme_y = phoneme[phoneme_y_label]
phoneme_y = phoneme_y.replace({1: int(-1), 2: int(1)})

# ## fit network
algs = ['random_hill_climb', 'genetic_alg', 'simulated_annealing']
train_sizes=[500, 1000, 1500, 2000, 4000]

nn_results_re = {}
cnts = [20, 100, 500, 1000]
with concurrent.futures.ThreadPoolExecutor() as exe:
    for alg in algs:
        nn_results_re[alg] = {}
        for cnt in cnts:
            nn = mlrose.NeuralNetwork(hidden_nodes=[4], activation='relu',
                                algorithm=algs,
                                max_iters=cnt,
                                bias=True,
                                is_classifier=True,
                                learning_rate=0.01,
                                early_stopping=False,
                                clip_max=6,
                                max_attempts=10)

            nn_results_re[alg][cnt] = exe.submit(learning_curve, nn, phoneme_x, phoneme_y, cv=5, train_sizes=train_sizes, scoring='accuracy', return_times=True)


# ## learning curves plotting
alg_names = {'random_hill_climb': "Random Hill Climb", 'genetic_alg': "Genetic Algorithm", 'simulated_annealing': "Simulated Annealing"}
for alg in nn_results_re:
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15,6))
    fig.suptitle(f"Neural Net Learning Curves for {alg_names[alg]}", fontsize=19)
    fig.subplots_adjust(hspace=0.5, wspace=0.1)
    i = 0
    for ax, iters in zip(axes.flatten(), nn_results_re[alg]):
        sizes, train_scores, test_scores, fit_times, _ = nn_results_re[alg][iters].result()
        ax.plot(sizes, 1-np.mean(train_scores, axis=1), label="Training accuracy")
        ax.plot(sizes, 1-np.mean(test_scores, axis=1), label="Validation accuracy")
        ax.set_ylim(0,1)
        ax.set_ylabel("accuracy", fontsize=14)
        if i > 1: ax.set_xlabel("training set size", fontsize=14)
        i+=1
        ax.legend()
        ax.set_title(f'{iters} iterations', fontsize=15)


# ## fit times plotting
alg_names = {'random_hill_climb': "Random Hill Climb", 'genetic_alg': "Genetic Algorithm", 'simulated_annealing': "Simulated Annealing"}
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(15,8))
fig.subplots_adjust(hspace=0.6, top=0.9)
i = 0
for ax, alg in zip(axes.flatten(), nn_results_re):
    ax.set_ylim(0,30)
    ax.set_xlim(400,4100)
    ax.set_title(f'Fit times for {alg_names[alg]}', fontsize=15)
    ax.set_ylabel("fit time [s]", fontsize=14)
    if i==2: ax.set_xlabel("training set size", fontsize=14)
    i+=1

    for iters in nn_results_re[alg]:
        sizes, train_scores, test_scores, fit_times, _ = nn_results_re[alg][iters].result()
        ax.plot(sizes, np.mean(fit_times, axis=1), label=(f'{iters} iterations'))
    ax.legend(bbox_to_anchor=(0.65, 0.5, 0.48, 0.4))

