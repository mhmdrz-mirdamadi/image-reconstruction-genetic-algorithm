from Individual import Individual
from Genetic import Genetic


if __name__ == '__main__':
    target_img = Individual('mario.png')
#     target_img.show()

    model = Genetic(target_img, population_size=50,
            tournament_size=0.08, parents_size=0.2,
            mutation_rate=0.3)
    model.initialize_population()
    model.run(generations=500)