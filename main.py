from Individual import Individual
from Genetic import Genetic

if __name__ == '__main__':
    target_img = Individual('test/angry_bird.png')

    model = Genetic(target_img, population_size=120,
            tournament_size=0.08, parents_size=0.2,
            mutation_rate=0.3)

    print('Initializing...')
    model.initialize_population()
    print('Population Initialized.')
    
    print('Running...')
    model.run(generations=100)

    print('Calculating Top Individuals...')
    top, fitness = model.top_individuals()
    top.show(f'Fitness: {fitness}')
    top.save(f'out/angry_bird_reconstructed.png')
