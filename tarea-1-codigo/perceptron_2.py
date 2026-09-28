from matplotlib import pyplot as plt
#Tarea 1 Arturo Pulgar

def leer_csv(ruta_archivo):
    with open(ruta_archivo, "r", encoding="utf-8") as archivo:
        lineas = archivo.read().splitlines()

    datos = []
    for linea in lineas:
        if not linea.strip():
            continue

        partes = linea.split(",")
        if len(partes) < 2:
            continue

        fila = []
        valido = True
        for parte in partes:
            try:
                fila.append(float(parte.strip()))
            except ValueError:
                valido = False
                break

        if valido:
            datos.append(fila)

    if not datos:
        raise ValueError("El archivo CSV no contiene datos numéricos válidos.")

    return datos


def suma_perceptron(vector_entrada, pesos, sesgo):
    total = sesgo
    for i in range(len(vector_entrada)):
        total += pesos[i] * vector_entrada[i]
    return total


def activacion_signo(valor):
    if valor >= 0:
        return 1
    return 0


def activacion_escalon(valor):
    if valor >= 0:
        return 1
    return 0


def obtener_activacion(nombre):
    nombre = nombre.strip().lower()
    if nombre in ("s", "signo", "sign", "sgn"):
        return activacion_signo
    if nombre in ("h", "heaviside", "escalon", "escalón", "step", "step_function"):
        return activacion_escalon
    raise ValueError("La función de activación debe ser 's' para signo o 'h' para escalon/heaviside.")


def pedir_numero(mensaje):
    while True:
        valor = input(mensaje).strip()
        if valor == "":
            print("Debe ingresar un valor.")
            continue
        try:
            return float(valor)
        except ValueError:
            print("Debe ingresar un número válido.")


def pedir_pesos(cantidad):
    pesos = []
    for i in range(cantidad):
        pesos.append(pedir_numero(f"Ingrese el peso para x{i + 1}: "))
    return pesos


def predecir(vector_entrada, pesos, sesgo, funcion_activacion):
    valor_total = suma_perceptron(vector_entrada, pesos, sesgo)
    return funcion_activacion(valor_total)


def medir_precision(datos, pesos, sesgo, funcion_activacion):
    aciertos = 0
    total = 0

    for fila in datos:
        entrada = fila[:-1]
        salida_esperada = int(fila[-1])
        salida_predicha = predecir(entrada, pesos, sesgo, funcion_activacion)

        total += 1
        if salida_predicha == salida_esperada:
            aciertos += 1

    if total == 0:
        return 0.0

    porcentaje = (aciertos / total) * 100
    return porcentaje


def graficar_resultados(datos, pesos, sesgo, funcion_activacion):
    x1 = []
    x2 = []
    esperados = []
    predichos = []
    coincidencias = []

    for fila in datos:
        entrada = fila[:-1]
        salida_esperada = int(fila[-1])

        if len(entrada) >= 2:
            x1.append(float(entrada[0]))
            x2.append(float(entrada[1]))
        elif len(entrada) == 1:
            x1.append(float(entrada[0]))
            x2.append(0.0)
        else:
            x1.append(0.0)
            x2.append(0.0)

        salida_predicha = predecir(entrada, pesos, sesgo, funcion_activacion)
        esperados.append(salida_esperada)
        predichos.append(salida_predicha)
        coincidencias.append(salida_esperada == salida_predicha)

    figura, ejes = plt.subplots(1, 3, figsize=(15, 5))

    color_esperado = ["blue" if valor == 1 else "orange" for valor in esperados]
    ejes[0].scatter(x1, x2, c=color_esperado, s=60)
    ejes[0].set_title("Valor esperado")
    ejes[0].set_xlabel("x1")
    ejes[0].set_ylabel("x2")

    color_predicho = ["green" if valor == 1 else "red" for valor in predichos]
    ejes[1].scatter(x1, x2, c=color_predicho, s=60)
    ejes[1].set_title("Valor predicho")
    ejes[1].set_xlabel("x1")
    ejes[1].set_ylabel("x2")

    color_coincidencia = ["green" if valor else "red" for valor in coincidencias]
    ejes[2].scatter(x1, x2, c=color_coincidencia, s=60)
    ejes[2].set_title("Coincidencia")
    ejes[2].set_xlabel("x1")
    ejes[2].set_ylabel("x2")

    precision = medir_precision(datos, pesos, sesgo, funcion_activacion)
    print(f"Precisión del perceptrón: {precision:.2f}%")

    figura.tight_layout()
    plt.show()




def ejecutar_programa():
    print("Programa interactivo del perceptrón")
    print("La última columna del CSV indica la salida esperada.")
    print("Las demás columnas son las variables de entrada.")

    while True:
        try:
            ruta_csv = input("Ingrese la ruta del archivo CSV: ").strip()
            datos = leer_csv(ruta_csv)

            cantidad_columnas = len(datos[0])
            if cantidad_columnas < 2:
                raise ValueError("El archivo debe tener al menos dos columnas.")

            pesos = pedir_pesos(cantidad_columnas - 1)
            sesgo = pedir_numero("Ingrese el valor del sesgo: ")

            nombre_activacion = input("Ingrese la función de activación (s = signo, h = escalon/heaviside): ").strip()
            funcion_activacion = obtener_activacion(nombre_activacion)

            graficar_resultados(datos, pesos, sesgo, funcion_activacion)

            respuesta = input("¿Desea probar con otros pesos? (s/n): ").strip().lower()
            if respuesta not in ("s", "si", "y", "yes"):
                break

        except FileNotFoundError:
            print("No se encontró el archivo. Verifique la ruta.")
            respuesta = input("¿Desea intentar nuevamente? (s/n): ").strip().lower()
            if respuesta not in ("s", "si", "y", "yes"):
                break
        except ValueError as error:
            print(error)
            respuesta = input("¿Desea intentar nuevamente? (s/n): ").strip().lower()
            if respuesta not in ("s", "si", "y", "yes"):
                break

    print("Fin del programa.")


if __name__ == "__main__":
    ejecutar_programa()

