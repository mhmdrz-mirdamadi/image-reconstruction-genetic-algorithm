from json import load
from Individual import Individual
from Genetic import Genetic

if __name__ == '__main__':

    with open('params.json') as params_json:
        params = load(params_json)

    target_img = Individual(params['input_path'])

    model = Genetic(
            target_img,
            population_size=params['population_size'],
            tournament_size=params['tournament_size'],
            parents_size=params['parents_size'],
            mutation_rate=params['mutation_rate'],
            blend_use=params['blend_use'],
            block_size=tuple(params['block_size']),
            num_blocks_initialize=tuple(params['num_blocks_initialize']),
            max_blocks_mutation=params['max_blocks_mutation']
            )

    print('Initializing...')
    model.initialize_population()
    print('Population Initialized.')
    
    print('Running...')
    model.run(
        generations=params['generations'],
        print_every=params['print_every'])

    print('Calculating Top Individual...')
    top, fitness = model.top_individuals()
    top.show(f'Fitness: {fitness}')
    top.save(params['output_path'])
