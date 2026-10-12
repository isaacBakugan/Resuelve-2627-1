# Nombre del integrante: Tomás Paraco
# Cédula del integrante: 30497481

# Tarea 2 - Perceptrón multicapa
# Parte del código de la tarea 1 (lectura del CSV, entrada por consola y
# gráficas) y le agrega una red con capas ocultas que se entrena con
# retropropagación. Se puede crear, entrenar, probar, guardar y cargar.

import matplotlib.pyplot as plt
import numpy as np

TASA_APRENDIZAJE = 0.5

COLOR_ACIERTO = "green"
COLOR_FALLO = "red"


# ---------- Lectura de archivos (igual que en la tarea 1) ----------

def carpeta_del_programa():
    # Carpeta donde está este .py, para encontrar assets/ desde cualquier lado
    ruta = __file__.replace("\\", "/")
    if "/" not in ruta:
        return "."
    return ruta.rsplit("/", 1)[0]


def abrir_csv(ruta):
    # Primero prueba la ruta tal cual; si no existe, la busca junto a este archivo
    try:
        return open(ruta, encoding="utf-8-sig")
    except FileNotFoundError:
        return open(carpeta_del_programa() + "/" + ruta, encoding="utf-8-sig")


def leer_lineas(ruta):
    with abrir_csv(ruta) as archivo:
        return [linea.strip() for linea in archivo.read().splitlines() if linea.strip() != ""]


def leer_datos(ruta):
    # Devuelve una matriz con las entradas y una lista con la clase de cada fila
    lineas = leer_lineas(ruta)
    entradas = []
    clases = []
    for linea in lineas[1:]:  # la primera línea es el encabezado
        valores = [float(v) for v in linea.split(",")]
        entradas.append(valores[:-1])
        clases.append(valores[-1])

    if len(entradas) == 0:
        raise ValueError("no tiene filas de datos")
    for fila in entradas:
        if len(fila) == 0 or len(fila) != len(entradas[0]):
            raise ValueError("todas las filas deben tener las mismas columnas, mínimo 2")
    return np.array(entradas), clases


# ---------- Clases de salida ----------

def clases_de_la_red(clases, v):
    # Cada neurona de salida representa una clase. Normalmente las clases son
    # las que aparecen en el archivo, ordenadas (-1 y 1, o 1, 2, 3 y 4).
    distintas = sorted(set(clases))
    if len(distintas) == v:
        return distintas
    # Si al archivo le falta alguna clase, se prueba con las numeraciones usuales
    for posibles in ([-1, 1], list(range(1, v + 1)), list(range(v))):
        if len(posibles) == v and all(c in posibles for c in distintas):
            return posibles
    return None


def revisar_archivo(red, entradas, clases):
    # Devuelve la lista de clases si el archivo sirve para la red; si no, None
    columnas = entradas.shape[1]
    lista_clases = clases_de_la_red(clases, red["v"])
    if columnas != red["u"] or lista_clases is None:
        print(f"Error: el archivo tiene {columnas} entradas y {len(set(clases))} clases, "
              f"pero la red tiene u = {red['u']} neuronas de entrada y v = {red['v']} de salida.")
        print(f"Configure la red con {columnas} neuronas de entrada y "
              f"{len(set(clases))} neuronas de salida.")
        return None
    return lista_clases


def a_vectores(clases, lista_clases):
    # Convierte cada clase en un vector con un 1 en su neurona y 0 en las demás
    objetivos = np.zeros((len(clases), len(lista_clases)))
    for i in range(len(clases)):
        objetivos[i][lista_clases.index(clases[i])] = 1
    return objetivos


# ---------- Perceptrón multicapa ----------

def crear_red(u, v, L, b):
    # pesos[k] es una matriz (neuronas que entran) x (neuronas de la capa k + 1)
    tamanos = [u] + [b] * L + [v]
    pesos = []
    for k in range(len(tamanos) - 1):
        pesos.append(np.random.uniform(-1, 1, (tamanos[k], tamanos[k + 1])))
    return {"u": u, "v": v, "L": L, "b": b, "e": 0, "pesos": pesos}


def sigmoide(z):
    z = np.clip(z, -500, 500)  # evita que exp se desborde
    return 1 / (1 + np.exp(-z))


def alimentar(red, x):
    # Alimentación hacia delante: guarda la salida de cada capa
    salidas = [x]
    for matriz in red["pesos"]:
        salidas.append(sigmoide(salidas[-1] @ matriz))
    return salidas


def entrenar(red, entradas, objetivos, epocas):
    # Retropropagación: se ajustan los pesos después de cada vector
    errores = []
    for _ in range(epocas):
        error_epoca = 0
        for i in np.random.permutation(len(entradas)):
            salidas = alimentar(red, entradas[i])
            y = salidas[-1]
            error_epoca += np.sum((objetivos[i] - y) ** 2) / 2

            # delta de la capa de salida
            delta = (y - objetivos[i]) * y * (1 - y)
            # se recorre la red de atrás hacia delante
            for k in range(len(red["pesos"]) - 1, -1, -1):
                anterior = salidas[k]
                delta_anterior = (red["pesos"][k] @ delta) * anterior * (1 - anterior)
                red["pesos"][k] -= TASA_APRENDIZAJE * np.outer(anterior, delta)
                delta = delta_anterior
        errores.append(error_epoca / len(entradas))
    red["e"] += epocas
    return errores


def predecir(red, entradas, lista_clases):
    # La clase predicha es la de la neurona de salida con el valor más alto
    predichos = []
    for x in entradas:
        salida = alimentar(red, x)[-1]
        predichos.append(lista_clases[int(np.argmax(salida))])
    return predichos


def contar_aciertos(esperados, predichos):
    aciertos = 0
    for i in range(len(esperados)):
        if esperados[i] == predichos[i]:
            aciertos += 1
    return aciertos


def imprimir_hiperparametros(red):
    print("Hiperparámetros de la red:")
    print(f"  u (entradas): {red['u']}")
    print(f"  v (salidas): {red['v']}")
    print(f"  L (capas ocultas): {red['L']}")
    print(f"  b (neuronas por capa oculta): {red['b']}")
    print(f"  e (épocas entrenadas): {red['e']}")


# ---------- Guardar y cargar la red ----------

def guardar_red(red, ruta):
    c = max(red["u"], red["b"])
    lineas = ["tipo,capa,posicion," + ",".join(f"peso_{i}" for i in range(c))]
    # primeras 5 filas: hiperparámetros, con las columnas de pesos vacías
    for letra in ["u", "v", "L", "b", "e"]:
        lineas.append(f"{letra},{red[letra]}" + "," * (c + 1))
    # después, una fila por neurona
    for k in range(len(red["pesos"])):
        matriz = red["pesos"][k]
        tipo = "o" if k == len(red["pesos"]) - 1 else "h"
        for j in range(matriz.shape[1]):
            pesos = [repr(float(p)) for p in matriz[:, j]]
            pesos += [""] * (c - len(pesos))
            lineas.append(f"{tipo},{k + 1},{j}," + ",".join(pesos))
    with open(ruta, "w", encoding="utf-8") as archivo:
        archivo.write("\n".join(lineas) + "\n")


def cargar_red(ruta):
    lineas = leer_lineas(ruta)
    if len(lineas) < 7:
        raise ValueError("el archivo no tiene hiperparámetros y neuronas")

    hiper = {}
    for linea in lineas[1:6]:
        partes = linea.split(",")
        hiper[partes[0]] = int(partes[1])
    for letra in ["u", "v", "L", "b", "e"]:
        if letra not in hiper:
            raise ValueError(f"falta el hiperparámetro {letra}")

    red = crear_red(hiper["u"], hiper["v"], hiper["L"], hiper["b"])
    red["e"] = hiper["e"]
    for linea in lineas[6:]:
        partes = linea.split(",")
        k = int(partes[1]) - 1  # la capa 1 es la primera oculta -> pesos[0]
        j = int(partes[2])
        entradas = red["pesos"][k].shape[0]
        red["pesos"][k][:, j] = [float(p) for p in partes[3:3 + entradas]]
    return red


# ---------- Entrada por consola ----------

def pedir_entero(mensaje, minimo=1):
    while True:
        texto = input(mensaje).strip()
        if texto.isdigit() and int(texto) >= minimo:
            return int(texto)
        print(f"Valor inválido, escriba un entero mayor o igual a {minimo}.")


def pedir_texto(mensaje, sugerido=""):
    if sugerido != "":
        mensaje += f"[{sugerido}]: "
    texto = input(mensaje).strip().strip('"')
    if texto == "":
        return sugerido
    return texto


def pedir_opcion(opciones):
    for i in range(len(opciones)):
        print(f"  {i + 1}) {opciones[i]}")
    while True:
        eleccion = input("Elija una opción: ").strip()
        if eleccion.isdigit() and 1 <= int(eleccion) <= len(opciones):
            return int(eleccion)
        print(f"Opción inválida, escriba un número del 1 al {len(opciones)}.")


def pedir_datos(mensaje, sugerido):
    # Repite hasta que el archivo de datos se pueda leer
    while True:
        ruta = pedir_texto(mensaje, sugerido)
        try:
            return leer_datos(ruta)
        except OSError:
            print("No se pudo abrir el archivo. Intente de nuevo.")
        except ValueError as error:
            print(f"El archivo no es válido ({error}). Intente de nuevo.")


# ---------- Gráficas ----------

def crear_eje_datos(figura, posicion, entradas):
    # Con 3 o más entradas se usa proyección 3D con x1, x2 y x3
    if entradas.shape[1] >= 3:
        eje = figura.add_subplot(1, 3, posicion, projection="3d")
        eje.set_zlabel("x3")
    else:
        eje = figura.add_subplot(1, 3, posicion)
    eje.set_xlabel("x1")
    eje.set_ylabel("x2")
    return eje


def dibujar_puntos(eje, entradas, color, etiqueta, marcador="o"):
    if len(entradas) == 0:
        return
    if entradas.shape[1] >= 3:
        eje.scatter(entradas[:, 0], entradas[:, 1], entradas[:, 2],
                    color=color, s=20, marker=marcador, label=etiqueta)
    elif entradas.shape[1] == 2:
        eje.scatter(entradas[:, 0], entradas[:, 1], color=color, s=20,
                    marker=marcador, label=etiqueta)
    else:
        # con una sola entrada, x2 se toma como 0
        eje.scatter(entradas[:, 0], np.zeros(len(entradas)), color=color, s=20,
                    marker=marcador, label=etiqueta)


def graficar(entradas, esperados, predichos, lista_clases, errores, titulo):
    figura = plt.figure(figsize=(17, 5.5))

    # 1) datos del archivo, un color por clase
    eje = crear_eje_datos(figura, 1, entradas)
    colores = ["blue", "orange", "purple", "brown", "pink", "gray", "olive", "cyan"]
    for i in range(len(lista_clases)):
        filas = [j for j in range(len(esperados)) if esperados[j] == lista_clases[i]]
        dibujar_puntos(eje, entradas[filas], colores[i % len(colores)], f"clase {lista_clases[i]:g}")
    eje.set_title("Datos del archivo")
    eje.legend(fontsize="small")

    # 2) predicción correcta o incorrecta
    eje = crear_eje_datos(figura, 2, entradas)
    bien = [j for j in range(len(esperados)) if esperados[j] == predichos[j]]
    mal = [j for j in range(len(esperados)) if esperados[j] != predichos[j]]
    dibujar_puntos(eje, entradas[bien], COLOR_ACIERTO, "correcta")
    dibujar_puntos(eje, entradas[mal], COLOR_FALLO, "incorrecta", "x")
    eje.set_title("Predicción (verde = correcta, rojo = incorrecta)")
    eje.legend(fontsize="small")

    # 3) error: por época si se entrenó, matriz de confusión si se probó
    eje = figura.add_subplot(1, 3, 3)
    if errores is not None:
        eje.plot(range(1, len(errores) + 1), errores, color="purple")
        eje.set_title("Error por época")
        eje.set_xlabel("Época")
        eje.set_ylabel("Error cuadrático medio")
    else:
        n = len(lista_clases)
        matriz = np.zeros((n, n), dtype=int)
        for j in range(len(esperados)):
            matriz[lista_clases.index(esperados[j])][lista_clases.index(predichos[j])] += 1
        eje.matshow(matriz, cmap="Blues")
        for fila in range(n):
            for col in range(n):
                color = "white" if matriz[fila][col] > matriz.max() / 2 else "black"
                eje.text(col, fila, str(matriz[fila][col]), ha="center", va="center", color=color)
        nombres = [f"{c:g}" for c in lista_clases]
        eje.set_xticks(range(n))
        eje.set_xticklabels(nombres)
        eje.set_yticks(range(n))
        eje.set_yticklabels(nombres)
        eje.set_xlabel("Clase predicha")
        eje.set_ylabel("Clase real")
        eje.set_title("Matriz de confusión")

    figura.suptitle(titulo)
    figura.tight_layout()
    plt.show()
    plt.close(figura)


# ---------- Opciones del menú ----------

def archivo_sugerido(red, tipo):
    # Sugiere el dataset de assets/ que corresponde al tamaño de la red
    if red["u"] >= 3:
        return f"assets/3-d_4-class_{tipo}.csv"
    return f"assets/2-d_2-class_{tipo}.csv"


def opcion_entrenar(red):
    entradas, clases = pedir_datos("Archivo de entrenamiento ", archivo_sugerido(red, "train"))
    lista_clases = revisar_archivo(red, entradas, clases)
    if lista_clases is None:
        return
    epocas = pedir_entero("Número de épocas: ")
    objetivos = a_vectores(clases, lista_clases)

    print("Entrenando...")
    errores = entrenar(red, entradas, objetivos, epocas)
    predichos = predecir(red, entradas, lista_clases)
    aciertos = contar_aciertos(clases, predichos)
    porcentaje = 100 * aciertos / len(clases)
    print(f"Error de la última época: {errores[-1]:.4f}")
    print(f"Aciertos sobre el entrenamiento: {aciertos} de {len(clases)} ({porcentaje:.1f}%)")
    imprimir_hiperparametros(red)
    graficar(entradas, clases, predichos, lista_clases, errores,
             f"Entrenamiento: {epocas} épocas, aciertos = {porcentaje:.1f}%")


def opcion_probar(red):
    entradas, clases = pedir_datos("Archivo de prueba ", archivo_sugerido(red, "test"))
    lista_clases = revisar_archivo(red, entradas, clases)
    if lista_clases is None:
        return
    predichos = predecir(red, entradas, lista_clases)
    aciertos = contar_aciertos(clases, predichos)
    porcentaje = 100 * aciertos / len(clases)
    print(f"Aciertos: {aciertos} de {len(clases)} ({porcentaje:.1f}%)")
    imprimir_hiperparametros(red)
    graficar(entradas, clases, predichos, lista_clases, None,
             f"Prueba: aciertos = {porcentaje:.1f}%")


def opcion_guardar(red):
    ruta = pedir_texto("Archivo donde guardar la red ", "red.csv")
    try:
        guardar_red(red, ruta)
        print(f"Red guardada en {ruta}")
    except OSError:
        print("No se pudo guardar el archivo.")


def crear_o_cargar():
    while True:
        print("¿Qué desea hacer?")
        opcion = pedir_opcion(["Crear un nuevo perceptrón multicapa", "Cargarlo de un archivo"])
        if opcion == 1:
            u = pedir_entero("Número de neuronas de entrada: ")
            v = pedir_entero("Número de neuronas de salida: ")
            L = pedir_entero("Número de capas ocultas: ", 0)
            b = pedir_entero("Número de neuronas por capa: ")
            return crear_red(u, v, L, b)
        ruta = pedir_texto("Archivo de la red: ")
        try:
            red = cargar_red(ruta)
            print("Red cargada.")
            imprimir_hiperparametros(red)
            return red
        except OSError:
            print("No se pudo abrir el archivo.")
        except (ValueError, IndexError):
            print("El archivo no tiene el formato de un perceptrón multicapa.")


# ---------- Programa principal ----------

def main():
    red = crear_o_cargar()
    while True:
        print()
        opcion = pedir_opcion(["Entrenar", "Probar", "Guardar", "Salir"])
        if opcion == 1:
            opcion_entrenar(red)
        elif opcion == 2:
            opcion_probar(red)
        elif opcion == 3:
            opcion_guardar(red)
        else:
            break


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\nPrograma terminado.")
