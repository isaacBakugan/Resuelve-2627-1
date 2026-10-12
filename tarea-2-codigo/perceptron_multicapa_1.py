# Arturo Pulgar
# 29920034 


from __future__ import annotations

import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


LEARNING_RATE = 0.1


@dataclass
class MultilayerPerceptron:
    input_size: int
    output_size: int
    hidden_layers: int
    neurons_per_layer: int
    epochs_trained: int
    weights: list[np.ndarray]

    @classmethod
    def create(
        cls, input_size: int, output_size: int, hidden_layers: int, neurons_per_layer: int
    ) -> "MultilayerPerceptron":
        sizes = [input_size] + [neurons_per_layer] * hidden_layers + [output_size]
        weights = []
        random_generator = np.random.default_rng()
        for previous_size, current_size in zip(sizes, sizes[1:]):
            limit = math.sqrt(6 / (previous_size + current_size))
            weights.append(
                random_generator.uniform(
                    -limit, limit, size=(previous_size, current_size)
                )
            )
        return cls(input_size, output_size, hidden_layers, neurons_per_layer, 0, weights)

    def forward(self, inputs: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
        inputs = np.asarray(inputs, dtype=float)
        if inputs.ndim not in (1, 2) or inputs.shape[-1] != self.input_size:
            raise ValueError(f"La entrada debe terminar con {self.input_size} valores.")
        if not np.isfinite(inputs).all():
            raise ValueError("Las entradas de la red deben ser valores finitos.")

        activations = [inputs]
        for layer_weights in self.weights[:-1]:
            hidden_values = np.clip(activations[-1] @ layer_weights, -60, 60)
            activations.append(1.0 / (1.0 + np.exp(-hidden_values)))

        logits = activations[-1] @ self.weights[-1]
        logits -= np.max(logits, axis=-1, keepdims=True)
        exponentials = np.exp(logits)
        outputs = exponentials / np.sum(exponentials, axis=-1, keepdims=True)
        return activations, outputs

    def predict_indices(self, inputs: np.ndarray) -> np.ndarray:
        _, outputs = self.forward(inputs)
        return np.argmax(outputs, axis=1)

    def train(self, inputs: np.ndarray, targets: np.ndarray, epochs: int) -> list[float]:
        if inputs.ndim != 2 or inputs.shape[1] != self.input_size:
            raise ValueError(f"El entrenamiento requiere vectores de {self.input_size} entradas.")
        if targets.ndim != 1 or len(targets) != len(inputs) or len(inputs) == 0:
            raise ValueError("El entrenamiento requiere una clase por cada ejemplo no vacio.")
        if epochs < 1:
            raise ValueError("El numero de epocas debe ser mayor que cero.")
        if not np.isfinite(inputs).all():
            raise ValueError("Los datos de entrenamiento deben ser valores finitos.")
        if not np.issubdtype(targets.dtype, np.integer) or np.any(
            (targets < 0) | (targets >= self.output_size)
        ):
            raise ValueError("Las clases codificadas deben estar entre 0 y v-1.")

        losses = []
        for _ in range(epochs):
            order = np.random.permutation(len(inputs))
            total_loss = 0.0
            for sample_index in order:
                sample = inputs[sample_index]
                target = targets[sample_index]
                activations, output = self.forward(sample)
                total_loss -= math.log(max(float(output[target]), 1e-12))

                deltas = [output.copy()]
                deltas[0][target] -= 1.0
                for layer_index in range(len(self.weights) - 1, 0, -1):
                    hidden = activations[layer_index]
                    deltas.insert(
                        0,
                        (self.weights[layer_index] @ deltas[0]) * hidden * (1.0 - hidden),
                    )

                for layer_index, delta in enumerate(deltas):
                    self.weights[layer_index] -= LEARNING_RATE * np.outer(
                        activations[layer_index], delta
                    )

            losses.append(total_loss / len(inputs))
        self.epochs_trained += epochs
        return losses

    def save(self, filename: str | Path) -> None:
        column_count = max(self.input_size, self.neurons_per_layer)
        header = ["tipo", "capa", "posicion"] + [
            f"peso_{index}" for index in range(column_count)
        ]
        rows: list[list[str]] = [header]
        parameters = [
            ("u", self.input_size),
            ("v", self.output_size),
            ("L", self.hidden_layers),
            ("b", self.neurons_per_layer),
            ("e", self.epochs_trained),
        ]
        for name, value in parameters:
            rows.append([name, str(value)] + [""] * (column_count + 1))

        for layer_index, layer_weights in enumerate(self.weights):
            is_output = layer_index == len(self.weights) - 1
            neuron_type = "o" if is_output else "h"
            layer_number = self.hidden_layers + 1 if is_output else layer_index + 1
            for neuron_index in range(layer_weights.shape[1]):
                neuron_weights = [
                    format(float(weight), ".17g")
                    for weight in layer_weights[:, neuron_index]
                ]
                rows.append(
                    [neuron_type, str(layer_number), str(neuron_index)]
                    + neuron_weights
                    + [""] * (column_count - len(neuron_weights))
                )

        with Path(filename).open("w", newline="", encoding="utf-8") as output_file:
            csv.writer(output_file).writerows(rows)

    @classmethod
    def load(cls, filename: str | Path) -> "MultilayerPerceptron":
        with Path(filename).open("r", newline="", encoding="utf-8-sig") as input_file:
            rows = list(csv.reader(input_file))
        if len(rows) < 6 or len(rows[0]) < 4:
            raise ValueError("El archivo no tiene el formato de perceptron multicapa esperado.")

        parameters: dict[str, int] = {}
        for row in rows[1:6]:
            if len(row) < 2 or row[0] not in {"u", "v", "L", "b", "e"}:
                raise ValueError("Faltan hiperparametros validos en el archivo de red.")
            parameters[row[0]] = int(row[1])
        if set(parameters) != {"u", "v", "L", "b", "e"}:
            raise ValueError("El archivo debe incluir los hiperparametros u, v, L, b y e.")

        input_size = parameters["u"]
        output_size = parameters["v"]
        hidden_layers = parameters["L"]
        neurons_per_layer = parameters["b"]
        if min(input_size, output_size, neurons_per_layer) < 1 or hidden_layers < 0:
            raise ValueError("Los hiperparametros del archivo no son validos.")

        layer_sizes = [input_size] + [neurons_per_layer] * hidden_layers + [output_size]
        weights = [
            np.zeros((previous_size, current_size), dtype=float)
            for previous_size, current_size in zip(layer_sizes, layer_sizes[1:])
        ]
        seen: set[tuple[str, int, int]] = set()
        for row in rows[6:]:
            if not row or not any(cell.strip() for cell in row):
                continue
            if len(row) < 4 or row[0] not in {"h", "o"}:
                raise ValueError("Una fila de neurona tiene un formato invalido.")
            neuron_type, layer_text, position_text = row[:3]
            layer_number, neuron_index = int(layer_text), int(position_text)
            is_output = neuron_type == "o"
            expected_layer = hidden_layers + 1 if is_output else layer_number
            layer_index = expected_layer - 1
            if layer_index < 0 or layer_index >= len(weights):
                raise ValueError("Una neurona referencia una capa inexistente.")
            if (is_output and layer_index != len(weights) - 1) or (
                not is_output and layer_index >= hidden_layers
            ):
                raise ValueError("El tipo de neurona no corresponde a su capa.")
            if neuron_index < 0 or neuron_index >= weights[layer_index].shape[1]:
                raise ValueError("La posicion de una neurona esta fuera de rango.")
            identity = (neuron_type, layer_number, neuron_index)
            if identity in seen:
                raise ValueError("El archivo contiene neuronas duplicadas.")
            seen.add(identity)

            incoming_size = weights[layer_index].shape[0]
            values = [float(value) for value in row[3 : 3 + incoming_size]]
            if len(values) != incoming_size or not np.isfinite(values).all():
                raise ValueError("La cantidad de pesos no coincide con la capa.")
            weights[layer_index][:, neuron_index] = values

        expected_neurons = sum(layer.shape[1] for layer in weights)
        if len(seen) != expected_neurons:
            raise ValueError("El archivo no contiene todas las neuronas de la red.")
        return cls(
            input_size,
            output_size,
            hidden_layers,
            neurons_per_layer,
            parameters["e"],
            weights,
        )


def read_dataset(filename: str | Path) -> tuple[np.ndarray, np.ndarray, list[float]]:
    inputs: list[list[float]] = []
    labels: list[float] = []
    with Path(filename).open("r", newline="", encoding="utf-8-sig") as input_file:
        reader = csv.reader(input_file)
        header = next(reader, None)
        if header is None or len(header) < 2:
            raise ValueError("El CSV debe tener encabezado, entradas y una columna de clase.")
        for line_number, row in enumerate(reader, start=2):
            if not row or not any(value.strip() for value in row):
                continue
            if len(row) != len(header):
                raise ValueError(f"La fila {line_number} no tiene {len(header)} columnas.")
            try:
                inputs.append([float(value) for value in row[:-1]])
                labels.append(float(row[-1]))
            except ValueError as error:
                raise ValueError(f"La fila {line_number} contiene un valor no numerico.") from error
    if not inputs:
        raise ValueError("El archivo CSV no contiene ejemplos.")
    if not np.isfinite(inputs).all() or not np.isfinite(labels).all():
        raise ValueError("El archivo contiene valores NaN o infinitos.")
    classes = sorted(set(labels))
    return np.asarray(inputs, dtype=float), np.asarray(labels, dtype=float), classes


def validate_dataset(
    network: MultilayerPerceptron, inputs: np.ndarray, labels: np.ndarray, classes: list[float]
) -> np.ndarray:
    if inputs.shape[1] != network.input_size or len(classes) != network.output_size:
        raise ValueError(
            "El archivo contiene "
            f"{inputs.shape[1]} entradas y {len(classes)} clases; configure la red con "
            f"u={inputs.shape[1]} neuronas de entrada y v={len(classes)} neuronas de salida."
        )
    return np.searchsorted(classes, labels)


def print_hyperparameters(network: MultilayerPerceptron) -> None:
    print("\nHiperparametros de la red:")
    print(f"  Neuronas de entrada (u): {network.input_size}")
    print(f"  Neuronas de salida (v): {network.output_size}")
    print(f"  Capas ocultas (L): {network.hidden_layers}")
    print(f"  Neuronas por capa oculta (b): {network.neurons_per_layer}")
    print(f"  Epocas de entrenamiento acumuladas (e): {network.epochs_trained}")


def show_results(
    inputs: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
    classes: list[float],
    losses: list[float] | None = None,
    first_epoch: int = 1,
) -> None:
    figure = plt.figure(figsize=(15, 4.8), constrained_layout=True)
    try:
        _draw_results(figure, inputs, labels, predictions, classes, losses, first_epoch)
        plt.show()
    finally:
        plt.close(figure)


def _draw_results(
    figure: plt.Figure,
    inputs: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
    classes: list[float],
    losses: list[float] | None,
    first_epoch: int,
) -> None:
    data_axis = figure.add_subplot(1, 3, 1, projection="3d" if inputs.shape[1] >= 3 else None)
    if inputs.shape[1] >= 3:
        points = data_axis.scatter(
            inputs[:, 0], inputs[:, 1], inputs[:, 2], c=labels, cmap="viridis"
        )
        data_axis.set_zlabel("Entrada 3")
    elif inputs.shape[1] == 2:
        plt.sca(data_axis)
        points = plt.scatter(inputs[:, 0], inputs[:, 1], c=labels, cmap="viridis")
    else:
        points = data_axis.scatter(np.arange(len(inputs)), inputs[:, 0], c=labels, cmap="viridis")
        data_axis.set_xlabel("Ejemplo")
        data_axis.set_ylabel("Entrada 1")
    data_axis.set_title("Datos por clase real")
    if inputs.shape[1] >= 2:
        data_axis.set_xlabel("Entrada 1")
        data_axis.set_ylabel("Entrada 2")
    figure.colorbar(points, ax=data_axis, label="Clase real", shrink=0.75)

    correctness_axis = figure.add_subplot(1, 3, 2)
    correct = predictions == labels
    correctness_axis.scatter(
        np.arange(len(labels)),
        predictions,
        c=np.where(correct, "#238636", "#d1242f"),
        marker="o",
        label="Prediccion",
    )
    correctness_axis.scatter(
        np.arange(len(labels)), labels, facecolors="none", edgecolors="#1f6feb", label="Real"
    )
    correctness_axis.set_title(f"Predicciones correctas: {np.mean(correct):.1%}")
    correctness_axis.set_xlabel("Ejemplo")
    correctness_axis.set_ylabel("Clase")
    correctness_axis.legend(fontsize="small")
    correctness_axis.grid(alpha=0.2)

    error_axis = figure.add_subplot(1, 3, 3)
    if losses is not None:
        error_axis.plot(
            range(first_epoch, first_epoch + len(losses)), losses, color="#bf3989"
        )
        error_axis.set_title("Error de entrenamiento")
        error_axis.set_xlabel("Epoca")
        error_axis.set_ylabel("Entropia cruzada")
        error_axis.grid(alpha=0.2)
    else:
        class_indices = {label: index for index, label in enumerate(classes)}
        confusion = np.zeros((len(classes), len(classes)), dtype=int)
        for expected, predicted in zip(labels, predictions):
            confusion[class_indices[expected], class_indices[predicted]] += 1
        image = error_axis.matshow(confusion, cmap="Blues")
        error_axis.set_title("Matriz de confusion", pad=18)
        error_axis.set_xlabel("Predicha")
        error_axis.set_ylabel("Real")
        error_axis.set_xticks(range(len(classes)), labels=[str(value) for value in classes])
        error_axis.set_yticks(range(len(classes)), labels=[str(value) for value in classes])
        for row_index in range(len(classes)):
            for column_index in range(len(classes)):
                error_axis.text(
                    column_index,
                    row_index,
                    str(confusion[row_index, column_index]),
                    ha="center",
                    va="center",
                )
        figure.colorbar(image, ax=error_axis, fraction=0.046, pad=0.04)


def display_results(
    inputs: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
    classes: list[float],
    losses: list[float] | None = None,
    first_epoch: int = 1,
) -> None:
    try:
        show_results(inputs, labels, predictions, classes, losses, first_epoch)
    except Exception as error:
        print(f"No se pudieron mostrar los graficos: {error}")


def ask_positive_integer(prompt: str, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    while True:
        try:
            value = int(input(prompt))
            if value >= minimum:
                return value
        except ValueError:
            pass
        comparison = "cero o mayor" if allow_zero else "mayor que cero"
        print(f"Ingrese un entero {comparison}.")


def create_network() -> MultilayerPerceptron:
    print("\nConfiguracion del nuevo perceptron multicapa")
    input_size = ask_positive_integer("Numero de neuronas de entrada: ")
    output_size = ask_positive_integer("Numero de neuronas de salida: ")
    hidden_layers = ask_positive_integer("Numero de capas ocultas: ", allow_zero=True)
    neurons_per_layer = ask_positive_integer("Numero de neuronas por capa: ")
    return MultilayerPerceptron.create(
        input_size, output_size, hidden_layers, neurons_per_layer
    )


def select_network() -> MultilayerPerceptron | None:
    while True:
        print("\n1. Crear un nuevo perceptron multicapa")
        print("2. Cargar un perceptron multicapa desde un archivo")
        option = input("Seleccione una opcion: ").strip()
        if option == "1":
            return create_network()
        if option == "2":
            filename = input("Archivo del perceptron multicapa: ").strip().strip('"')
            try:
                network = MultilayerPerceptron.load(filename)
                print(f"Red cargada desde {filename}.")
                return network
            except (OSError, ValueError) as error:
                print(f"No se pudo cargar la red: {error}")
        else:
            print("Opcion no valida.")


def run_network(network: MultilayerPerceptron) -> None:
    while True:
        print_hyperparameters(network)
        print("\n1. Entrenar la red")
        print("2. Probar la red")
        print("3. Guardar la red")
        print("4. Salir")
        option = input("Seleccione una opcion: ").strip()
        if option in {"1", "2"}:
            prompt = "Archivo de entrenamiento: " if option == "1" else "Archivo de prueba: "
            filename = input(prompt).strip().strip('"')
            try:
                inputs, labels, classes = read_dataset(filename)
                targets = validate_dataset(network, inputs, labels, classes)
                if option == "1":
                    epochs = ask_positive_integer("Numero de epocas: ")
                    losses = network.train(inputs, targets, epochs)
                    predictions = np.asarray(classes)[network.predict_indices(inputs)]
                    print(f"Entrenamiento finalizado; epocas acumuladas: {network.epochs_trained}.")
                    display_results(
                        inputs,
                        labels,
                        predictions,
                        classes,
                        losses,
                        network.epochs_trained - len(losses) + 1,
                    )
                else:
                    predictions = np.asarray(classes)[network.predict_indices(inputs)]
                    display_results(inputs, labels, predictions, classes)
            except (OSError, ValueError) as error:
                print(f"No se pudo procesar el conjunto de datos: {error}")
        elif option == "3":
            filename = input("Archivo donde guardar la red: ").strip().strip('"')
            try:
                network.save(filename)
                print(f"Red guardada en {filename}.")
            except OSError as error:
                print(f"No se pudo guardar la red: {error}")
        elif option == "4":
            return
        else:
            print("Opcion no valida.")


def main() -> None:
    network = select_network()
    run_network(network)


if __name__ == "__main__":
    try:
        main()
    except (EOFError, KeyboardInterrupt):
        print("\nPrograma finalizado.")
        sys.exit(0)
