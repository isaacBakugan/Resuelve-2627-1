# Nombre del integrante: Tomás Paraco
# Cédula del integrante: 30497481

# Tarea 4 - Genetic algorithm
#
# Finds the maximum of a function read from a separate Python file
# (BOUNDS, VARIABLE_COUNT and fitness(*variables), see funcion_a_3.py and funcion_b_3.py).
#
# Two representations of the population can be used:
#   - "real": each gene is the value of the variable (a float).
#     Mutation adds gaussian noise with standard deviation = mutation variability.
#   - "binary": each variable is a string of BITS bits mapped to the bounds.
#     Each bit flips with probability variability / BITS, so the variability is the
#     expected number of flipped bits per variable.
#
# Difference threshold: the population is considered converged, and the evolution stops,
# when the difference between the best and the worst fitness of the population is
# smaller than the threshold. If that never happens it runs until the maximum number
# of generations.

import importlib.util
import json
import math
import random
from pathlib import Path

BITS = 16  # bits per variable in the binary representation
TOURNAMENT_SIZE = 3
CROSSOVER_RATE = 0.9

DEFAULTS = {
    "population_size": 10,
    "difference_threshold": 0.05,
    "max_generations": 500,
    "mutation_variability": 1.0,
}


# ---------- Files ----------

def find_file(path):
    # Looks for the file where the program is run and, if it is not there,
    # next to this .py (so it works from any folder, without absolute paths)
    path = Path(path)
    if path.exists():
        return path
    return Path(__file__).resolve().parent / path


def load_function(path):
    file = find_file(path)
    spec = importlib.util.spec_from_file_location(file.stem, file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------- Representations ----------

def random_genes(function, representation):
    low, high = function.BOUNDS
    if representation == "binary":
        return "".join(random.choice("01") for _ in range(BITS * function.VARIABLE_COUNT))
    return [random.uniform(low, high) for _ in range(function.VARIABLE_COUNT)]


def decode(genes, function, representation):
    # Returns the values of the variables
    if representation == "real":
        return list(genes)
    low, high = function.BOUNDS
    top = 2 ** BITS - 1
    values = []
    for i in range(function.VARIABLE_COUNT):
        number = int(genes[i * BITS:(i + 1) * BITS], 2)
        values.append(low + (high - low) * number / top)
    return values


def crossover(genes_1, genes_2, representation):
    if random.random() > CROSSOVER_RATE:
        return genes_1
    if representation == "binary":
        # one point crossover
        point = random.randint(1, len(genes_1) - 1)
        return genes_1[:point] + genes_2[point:]
    # blend crossover: a random point between both parents for every variable
    child = []
    for a, b in zip(genes_1, genes_2):
        weight = random.random()
        child.append(weight * a + (1 - weight) * b)
    return child


def mutate(genes, function, representation, variability):
    low, high = function.BOUNDS
    if representation == "binary":
        probability = min(1.0, variability / BITS)
        bits = []
        for bit in genes:
            if random.random() < probability:
                bit = "1" if bit == "0" else "0"
            bits.append(bit)
        return "".join(bits)
    return [min(high, max(low, value + random.gauss(0, variability))) for value in genes]


# ---------- Population ----------

def make_member(genes, function, representation):
    variables = decode(genes, function, representation)
    return {"variables": variables, "fitness": function.fitness(*variables), "genes": genes}


def create_population(function_path, representation, parameters):
    function = load_function(function_path)
    members = []
    for _ in range(parameters["population_size"]):
        members.append(make_member(random_genes(function, representation), function, representation))
    members.sort(key=lambda m: m["fitness"], reverse=True)
    parameters = dict(parameters, generations_elapsed=0)
    return {"function": str(function_path), "representation": representation,
            "parameters": parameters, "members": members, "module": function}


def tournament(members):
    competitors = random.sample(members, min(TOURNAMENT_SIZE, len(members)))
    return max(competitors, key=lambda m: m["fitness"])


def next_generation(population):
    function = population["module"]
    representation = population["representation"]
    variability = population["parameters"]["mutation_variability"]
    members = population["members"]

    new_members = [members[0]]  # elitism: the best member always survives
    while len(new_members) < len(members):
        parent_1, parent_2 = tournament(members), tournament(members)
        genes = crossover(parent_1["genes"], parent_2["genes"], representation)
        genes = mutate(genes, function, representation, variability)
        new_members.append(make_member(genes, function, representation))

    new_members.sort(key=lambda m: m["fitness"], reverse=True)
    population["members"] = new_members


def median(values):
    values = sorted(values)
    middle = len(values) // 2
    if len(values) % 2 == 1:
        return values[middle]
    return (values[middle - 1] + values[middle]) / 2


def evolve(population, generations):
    parameters = population["parameters"]
    for _ in range(generations):
        next_generation(population)
        parameters["generations_elapsed"] += 1
        values = [m["fitness"] for m in population["members"]]
        print(f"Generación {parameters['generations_elapsed']}: "
              f"máximo = {max(values):.6f}, mediana = {median(values):.6f}, "
              f"mínimo = {min(values):.6f}")
        if max(values) - min(values) < parameters["difference_threshold"]:
            print("La población convergió (diferencia menor que el umbral).")
            break


def show_population(population):
    parameters = population["parameters"]
    print(f"\nFunción: {population['function']} (representación {population['representation']})")
    print(f"Tamaño de la población: {parameters['population_size']}")
    print(f"Umbral de diferencia: {parameters['difference_threshold']}")
    print(f"Generaciones transcurridas: {parameters['generations_elapsed']} "
          f"(máximo {parameters['max_generations']})")
    print(f"Variabilidad de la mutación: {parameters['mutation_variability']}")
    print("Miembros, de mejor a peor:")
    for i, member in enumerate(population["members"], start=1):
        values = ", ".join(f"{v:.6f}" for v in member["variables"])
        print(f"  {i:>3}. ({values}) -> {member['fitness']:.6f}")


# ---------- Save and load ----------

def save_population(population, path):
    data = {
        "function": population["function"],
        "representation": population["representation"],
        "parameters": population["parameters"],
        "members": population["members"],
    }
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def load_population(path):
    with open(find_file(path), encoding="utf-8") as file:
        data = json.load(file)
    representation = data.get("representation", "real")
    function = load_function(data["function"])
    members = []
    for member in data["members"]:
        genes = member.get("genes")
        if genes is None:  # no genes saved: rebuild them from the variables (real numbers)
            representation, genes = "real", member["variables"]
        members.append(make_member(genes, function, representation))
    members.sort(key=lambda m: m["fitness"], reverse=True)
    return {"function": data["function"], "representation": representation,
            "parameters": data["parameters"], "members": members, "module": function}


# ---------- Console ----------

def ask_number(message, default, integer=False, minimum=None):
    while True:
        text = input(f"{message} [{default}]: ").strip().replace(",", ".")
        if text == "":
            return default
        try:
            value = int(text) if integer else float(text)
        except ValueError:
            print("Valor inválido, escriba un número.")
            continue
        if minimum is not None and value < minimum:
            print(f"El valor debe ser mayor o igual a {minimum}.")
            continue
        return value


def ask_option(message, options, default=1):
    print(message)
    for i, text in enumerate(options, start=1):
        print(f"  {i}) {text}")
    while True:
        text = input(f"Opción [{default}]: ").strip()
        if text == "":
            return default
        if text.isdigit() and 1 <= int(text) <= len(options):
            return int(text)
        print(f"Escriba un número del 1 al {len(options)}.")


def ask_text(message, default=""):
    suffix = f" [{default}]" if default else ""
    text = input(f"{message}{suffix}: ").strip().strip('"')
    return text if text else default


def create_menu():
    while True:
        function_path = ask_text("Archivo de la función", "funcion_a_3.py")
        try:
            load_function(function_path)
            break
        except (OSError, AttributeError, SyntaxError) as error:
            print(f"No se pudo cargar la función: {error}")
    representation = ["real", "binary"][ask_option(
        "Representación de la población:", ["Real (números decimales)", "Binaria (cadenas de bits)"]) - 1]
    parameters = {
        "population_size": ask_number("Tamaño de la población", DEFAULTS["population_size"], True, 2),
        "difference_threshold": ask_number("Umbral de diferencia", DEFAULTS["difference_threshold"], minimum=0),
        "max_generations": ask_number("Número máximo de generaciones", DEFAULTS["max_generations"], True, 1),
        "mutation_variability": ask_number("Variabilidad de la mutación", DEFAULTS["mutation_variability"], minimum=0),
    }
    return create_population(function_path, representation, parameters)


def load_menu():
    while True:
        path = ask_text("Archivo de la población (.json)")
        try:
            population = load_population(path)
            show_population(population)
            return population
        except (OSError, KeyError, ValueError) as error:
            print(f"No se pudo cargar la población: {error}")


def choose_population():
    option = ask_option("\n¿Qué desea hacer?", ["Crear una nueva población", "Cargar una población de un archivo"])
    return create_menu() if option == 1 else load_menu()


def main():
    population = choose_population()
    while True:
        option = ask_option("\n--- Menú ---", [
            "Evolucionar",
            "Mostrar la población",
            "Guardar la población (.json)",
            "Crear o cargar otra población",
            "Salir",
        ])
        if option == 1:
            parameters = population["parameters"]
            remaining = parameters["max_generations"] - parameters["generations_elapsed"]
            if remaining <= 0:
                remaining = parameters["max_generations"]
            generations = ask_number("Generaciones a evolucionar", remaining, True, 1)
            evolve(population, generations)
            show_population(population)
        elif option == 2:
            show_population(population)
        elif option == 3:
            path = ask_text("Archivo donde guardar", "poblacion.json")
            try:
                save_population(population, path)
                print(f"Población guardada en {path}")
            except OSError as error:
                print(f"No se pudo guardar: {error}")
        elif option == 4:
            population = choose_population()
        else:
            print("Fin del programa.")
            break


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\nPrograma terminado.")
