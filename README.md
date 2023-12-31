# Image Recreation by Genetic Algorithm

<img src="resource/example.jpg" alt="Example">

## Usage

1. Install requirements by:

```
pip install -r requirements.txt
```

2. Configure parameters in JSON file.
3. Run the code:

```
python main.py PATH_TO_PARAMS PATH_TO_INPUT PATH_TO_OUTPUT
```

## Parameters

Parameters should be in a JSON file, like `params.json`.

- `population_size`: Size of the initial population
- `tournament_size`: Size of each tournament to sample the population. ($\in (0, 1)$)
- `parents_size`: Determines how many individuals in the new population should be parents selected in the tournament and then crossover to produce new individuals. ($\in (0, 1)$)
- `mutation_rate`: Probability of mutation occurring. ($\in (0, 1)$)
- `generations`: Number of generations to run.
- `blend_use`: Probability of using blend crossover against two-point method. ($\in (0, 1)$)
- `block_size`: The lower and upper bound on the size of random blocks. ($\in (0, 1)$)
- `num_blocks_initialize`: The lower and upper bound on the number of random blocks in population initializing.
- `max_blocks_mutation`: The maximum number of random blocks to be added in mutation operation.

## Initial Population

Each Individual is an image with size of target image. When a random individual is created, it is given a random background color. Then, a random number of rectangular blocks with a random size of a specific random color for each individual is added to that background.

<img src="resource/initial_population.png" alt="Initial Population">

## Fitness Function

A relatively straightforward implementation for the fitness function is pixel-wise Mean Square Error (MSE) between two images. It may sound so strict to use this parameter because of its sensitivity to small differences which are not recognizable in Human Visual System (HVS) but actually, it works fine in practice.

## Selection

The tournament selection method is implemented to pick up the best individuals as parents for the next generation and then for crossover operation. It randomly samples the population given a tournament size and selects the fittest individual to be the winner of the tournament. In practice, a tournament size between 6-8% of the total population size would be a reasonable choice.

## Crossover

For crossover operation, it is used two different approaches which could help the diversity of the population. Both of these operations in one generation are used for creating the child which each has a probability of choosing.

### Blending Crossover

In this method, a random uniformly distributed variable $x \in [0, 1]$ is generated and the child is overlayed of a first parent with opacity $x$ and the second parent with opacity $1-x$.

### Two-Point Crossover

In this approach, two random points on the row and column are selected to be the crossover points. These two points divide the individual into four areas. In areas 1 and 4, the first parent, and in areas 2 and 3, the second parent will be placed without any further changes

<img src="resource/crossover.png" alt="Crossover">

## Mutation

To perform a mutation, a random number of rectangular blocks of random color (like something there was in the producing initial population) superimpose onto an individual.

<img src="resource/mutation.png" alt="Mutation">
