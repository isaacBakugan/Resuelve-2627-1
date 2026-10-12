import numpy as np
import matplotlib.pyplot as plt

TASA = 0.5  # tasa de aprendizaje

def leer_csv(nombre_archivo):
    entradas = []
    clases = []
    archivo = open(nombre_archivo, "r", encoding="utf-8-sig")
    lineas = archivo.readlines()
    archivo.close()
    encabezado = lineas[0].strip().split(",")
    for linea in lineas[1:]:
        linea = linea.strip()
        if linea == "":
            continue
        fila = linea.split(",")
        x = []
        for i in range(len(fila) - 1):
            x.append(float(fila[i]))
        entradas.append(x)
        clases.append(int(round(float(fila[-1]))))  # se redondea por si es difuso

    # si las clases empiezan en 1 (ej. 1,2,3,4) se les resta 1 para que empiecen en 0
    if min(clases) == 1:
        print("Las clases empiezan en 1, se usaran desde 0 (clase 1 -> 0, clase 2 -> 1, ...)")
        for i in range(len(clases)):
            clases[i] = clases[i] - 1
    return encabezado, entradas, clases

def sigmoide(z):
    if z < -500:
        return 0
    return 1 / (1 + np.exp(-z))

def suma(x, pesos, sesgo):
    total = sesgo
    for i in range(len(x)):
        total = total + pesos[i] * x[i]
    return total

def crear_red(u, v, L, b):
    pesos = []
    sesgos = []
    entradas_capa = u
    for capa in range(L + 1):
        if capa == L:
            neuronas = v  # la ultima capa es la de salida
        else:
            neuronas = b
        pesos_capa = []
        sesgos_capa = []
        for n in range(neuronas):
            w = []
            for i in range(entradas_capa):
                w.append(np.random.uniform(-1, 1))
            pesos_capa.append(w)
            sesgos_capa.append(np.random.uniform(-1, 1))
        pesos.append(pesos_capa)
        sesgos.append(sesgos_capa)
        entradas_capa = neuronas
    return pesos, sesgos

def alimentar(pesos, sesgos, x):
    salidas = [x]
    for capa in range(len(pesos)):
        salida_capa = []
        for n in range(len(pesos[capa])):
            z = suma(salidas[capa], pesos[capa][n], sesgos[capa][n])
            salida_capa.append(sigmoide(z))
        salidas.append(salida_capa)
    return salidas

def salida_deseada(clase, v):
    if v == 1:
        return [clase]
    deseado = []
    for i in range(v):
        if i == clase:
            deseado.append(1)
        else:
            deseado.append(0)
    return deseado

def obtener_clase(salida):
    if len(salida) == 1:
        if salida[0] >= 0.5:
            return 1
        else:
            return 0
    mayor = 0
    for i in range(len(salida)):
        if salida[i] > salida[mayor]:
            mayor = i
    return mayor



def retropropagacion(pesos, sesgos, x, deseado): # Retropropagacion para un vector. Devuelve el error de ese vector.
    salidas = alimentar(pesos, sesgos, x)
    ultima = len(pesos) - 1
    error = 0
    deltas = []
    for n in range(len(pesos[ultima])):
        y = salidas[ultima + 1][n]
        error = error + (deseado[n] - y) ** 2
        deltas.append((deseado[n] - y) * y * (1 - y))

    for capa in range(ultima, -1, -1):
        entrada = salidas[capa]
        deltas_anteriores = []
        if capa > 0:
            for i in range(len(entrada)):
                s = 0
                for n in range(len(pesos[capa])):
                    s = s + pesos[capa][n][i] * deltas[n]
                deltas_anteriores.append(s * entrada[i] * (1 - entrada[i]))

        for n in range(len(pesos[capa])):
            for i in range(len(entrada)):
                pesos[capa][n][i] = pesos[capa][n][i] + TASA * deltas[n] * entrada[i]
            sesgos[capa][n] = sesgos[capa][n] + TASA * deltas[n]

        deltas = deltas_anteriores

    return error


def entrenar(pesos, sesgos, entradas, clases, epocas, v):
    errores = []
    for epoca in range(epocas):
        error_total = 0
        for i in range(len(entradas)):
            error_total = error_total + retropropagacion(pesos, sesgos, entradas[i], salida_deseada(clases[i], v))
        errores.append(error_total / len(entradas))
        if (epoca + 1) % 100 == 0:
            print("Epoca", epoca + 1, "error:", errores[-1])
    return errores


def predecir(pesos, sesgos, entradas):
    predichos = []
    for x in entradas:
        salidas = alimentar(pesos, sesgos, x)
        predichos.append(obtener_clase(salidas[-1]))
    return predichos


def imprimir_hiperparametros(u, v, L, b, e):
    print("--- Hiperparametros ---")
    print("Neuronas de entrada (u):", u)
    print("Neuronas de salida (v):", v)
    print("Capas ocultas (L):", L)
    print("Neuronas por capa (b):", b)
    print("Epocas (e):", e)
    print("Tasa de aprendizaje:", TASA)


def graficar(encabezado, entradas, clases, predichos, errores, matriz): # Se hace los 3 graficos, si se entreno se pasa errores, si se probo se pasa matriz
    x1 = []
    x2 = []
    x3 = []
    for x in entradas:
        x1.append(x[0])
        if len(x) > 1:
            x2.append(x[1])
        else:
            x2.append(0)
        if len(x) > 2:
            x3.append(x[2])

    colores = []
    for i in range(len(clases)):
        if clases[i] == predichos[i]:
            colores.append("green")
        else:
            colores.append("red")

    figura = plt.figure(figsize=(17, 5))

    # grafico 1: datos del archivo
    # grafico 2: prediccion correcta o incorrecta
    # grafico 3: error de la red
    if len(x3) > 0: 
        g1 = figura.add_subplot(1, 3, 1, projection="3d")
        g1.scatter(x1, x2, x3, c=clases)
        g1.set_zlabel(encabezado[2])
        g2 = figura.add_subplot(1, 3, 2, projection="3d")
        g2.scatter(x1, x2, x3, c=colores)
        g2.set_zlabel(encabezado[2])
    else:
        g1 = figura.add_subplot(1, 3, 1)
        g1.scatter(x1, x2, c=clases)
        g2 = figura.add_subplot(1, 3, 2)
        g2.scatter(x1, x2, c=colores)
    g1.set_title("Datos del archivo")
    g2.set_title("Prediccion (verde = bien, rojo = mal)")
    g1.set_xlabel(encabezado[0])
    g2.set_xlabel(encabezado[0])
    g1.set_ylabel(encabezado[1])
    g2.set_ylabel(encabezado[1])
    g3 = figura.add_subplot(1, 3, 3)
    if errores != None:
        g3.plot(errores)
        g3.set_title("Error por epoca")
        g3.set_xlabel("Epoca")
        g3.set_ylabel("Error")
    else:
        plt.sca(g3)
        plt.matshow(matriz, fignum=0, cmap="Blues")
        for i in range(len(matriz)):
            for j in range(len(matriz)):
                g3.text(j, i, matriz[i][j], ha="center", va="center")
        g3.set_title("Matriz de confusion")
        g3.set_xlabel("Clase predicha")
        g3.set_ylabel("Clase real")

    plt.show()


def guardar_red(nombre_archivo, pesos, sesgos, u, v, L, b, e):
    c = max(u, b)
    archivo = open(nombre_archivo, "w")

    linea = "tipo,capa,posicion"
    for i in range(c):
        linea = linea + ",w" + str(i)
    archivo.write(linea + ",sesgo\n")

    vacias = "," * (c + 1)
    archivo.write("u," + str(u) + vacias + "\n")
    archivo.write("v," + str(v) + vacias + "\n")
    archivo.write("L," + str(L) + vacias + "\n")
    archivo.write("b," + str(b) + vacias + "\n")
    archivo.write("e," + str(e) + vacias + "\n")

    for capa in range(len(pesos)):
        if capa < L:
            tipo = "h"
        else:
            tipo = "o"
        for n in range(len(pesos[capa])):
            linea = tipo + "," + str(capa + 1) + "," + str(n)
            for i in range(c):
                if i < len(pesos[capa][n]):
                    linea = linea + "," + str(pesos[capa][n][i])
                else:
                    linea = linea + ","
            linea = linea + "," + str(sesgos[capa][n])
            archivo.write(linea + "\n")
    archivo.close()

def cargar_red(nombre_archivo):
    archivo = open(nombre_archivo, "r", encoding="utf-8-sig")
    lineas = archivo.readlines()
    archivo.close()

    u = int(lineas[1].split(",")[1])
    v = int(lineas[2].split(",")[1])
    L = int(lineas[3].split(",")[1])
    b = int(lineas[4].split(",")[1])
    e = int(lineas[5].split(",")[1])
    pesos, sesgos = crear_red(u, v, L, b)

    for linea in lineas[6:]:
        linea = linea.strip()
        if linea == "":
            continue
        fila = linea.split(",")
        capa = int(fila[1]) - 1
        n = int(fila[2])
        for i in range(len(pesos[capa][n])):
            pesos[capa][n][i] = float(fila[3 + i])
        sesgos[capa][n] = float(fila[-1])
    return pesos, sesgos, u, v, L, b, e


print("1. Crear una red nueva")
print("2. Cargar una red de un archivo")
opcion = input("Opcion: ")

if opcion == "1":
    u = int(input("Numero de neuronas de entrada: "))
    v = int(input("Numero de neuronas de salida: "))
    L = int(input("Numero de capas ocultas: "))
    b = int(input("Numero de neuronas por capa: "))
    e = 0
    pesos, sesgos = crear_red(u, v, L, b)
else:
    pesos, sesgos, u, v, L, b, e = cargar_red(input("Nombre del archivo de la red: "))
imprimir_hiperparametros(u, v, L, b, e)

opcion = ""
while opcion != "4":
    print()
    print("1. Entrenar la red")
    print("2. Probar la red")
    print("3. Guardar la red")
    print("4. Salir")
    opcion = input("Opcion: ")

    if opcion == "1" or opcion == "2":
        encabezado, entradas, clases = leer_csv(input("Nombre del archivo de datos: "))

        num_entradas = len(entradas[0])# revisar que la red corresponda con el archivo
        num_clases = max(clases) + 1
        if num_clases <= 2:
            salida_bien = (v == 1 or v == 2)
        else:
            salida_bien = (v == num_clases)
        if num_entradas != u or not salida_bien:
            print("ERROR: la red no corresponde con el archivo.")
            if num_clases <= 2:
                print("La red debe tener", num_entradas, "neuronas de entrada y 1 o 2 neuronas de salida.")
            else:
                print("La red debe tener", num_entradas, "neuronas de entrada y", num_clases, "neuronas de salida.")
            continue

        errores = None
        matriz = None
        if opcion == "1":
            epocas = int(input("Numero de epocas: "))
            errores = entrenar(pesos, sesgos, entradas, clases, epocas, v)
            e = e + epocas

        predichos = predecir(pesos, sesgos, entradas)

        if opcion == "2":
            n = max(v, 2) # matriz de confusion: filas = clase real, columnas = clase predicha
            matriz = []
            for i in range(n):
                matriz.append([0] * n)
            for i in range(len(clases)):
                matriz[clases[i]][predichos[i]] = matriz[clases[i]][predichos[i]] + 1

        aciertos = 0
        for i in range(len(clases)):
            if clases[i] == predichos[i]:
                aciertos = aciertos + 1
        print("Aciertos:", aciertos, "de", len(clases))
        imprimir_hiperparametros(u, v, L, b, e)
        graficar(encabezado, entradas, clases, predichos, errores, matriz)

    elif opcion == "3":
        guardar_red(input("Nombre del archivo: "), pesos, sesgos, u, v, L, b, e)
        print("Red guardada.")

print("Fin :)")
