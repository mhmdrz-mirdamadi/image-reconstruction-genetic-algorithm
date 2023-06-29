from Individual import Individual
from Genetic import Genetic

if __name__ == '__main__':
    target_img = Individual('test/mario.png')

    model = Genetic(target_img, population_size=100,
            tournament_size=0.08, parents_size=0.2,
            mutation_rate=0.3)

    print('Initializing...')
    model.initialize_population()
    print('Population Initialized.')
    
    print('Running...')
    model.run(generations=10000)

    print('Calculating Top Individuals...')
    top, fitness = model.top_individuals(3)
    for i, (t, f) in enumerate(zip(top, fitness), 4):
        t.show(f'Rank {i}, Fitness: {f}')
        t.save(f'out/mario_{i}.png')
