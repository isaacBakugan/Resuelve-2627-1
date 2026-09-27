import matplotlib.pyplot 
#Tarea 1 Christian Goncalves
E = 2.718281828459045  # numero de euler

def leer_csv(nombre_archivo):
    entradas = []
    esperados = []
    archivo = open(nombre_archivo, "r", encoding="utf-8-sig")
    lineas = archivo.readlines()
    archivo.close()
    encabezado = lineas[0].strip().split(",")  # la primera linea son los nombres
    for linea in lineas[1:]:
        linea = linea.strip()
        if linea == "":
            continue
        fila = linea.split(",")
        x = []
        for i in range(len(fila) - 1):
            x.append(float(fila[i]))
        entradas.append(x)
        esperados.append(float(fila[-1]))
    return encabezado, entradas, esperados

# Funcion suma: sesgo + w1*x1 + w2*x2 + ...
def suma(x, pesos, sesgo):
    total = sesgo
    for i in range(len(x)):
        total = total + pesos[i] * x[i]
    return total

# Funcion escalon
def escalon(z):
    if z >= 0:
        return 1
    else:
        return 0

# Funcion sigmoide
def sigmoide(z):
    if z < -500:  # si z es muy negativo da error por numero muy grande
        return 0
    return 1 / (1 + E ** (-z))

def predecir(entradas, pesos, sesgo, activacion):# Calcula la salida del perceptron para todas las entradas
    predichos = []
    for x in entradas:
        z = suma(x, pesos, sesgo)
        if activacion == 1:
            predichos.append(escalon(z))
        else:
            predichos.append(sigmoide(z))
    return predichos

def redondear(valor):# Convierte un valor a 0 o 1 (para comparar valores difusos o de la sigmoide)
    if valor >= 0.5:
        return 1
    else:
        return 0

def graficar(encabezado, entradas, esperados, predichos): # Hace los 3 graficos de dispersion
    x1 = []
    x2 = []  # solo se usan las dos primeras entradas para graficar
    for x in entradas:
        x1.append(x[0])
        if len(x) > 1:
            x2.append(x[1])
        else:
            x2.append(0)

    colores = []
    for i in range(len(esperados)):
        if redondear(esperados[i]) == redondear(predichos[i]):
            colores.append("green")
        else:
            colores.append("red")

    matplotlib.pyplot.figure(figsize=(15, 5))
    matplotlib.pyplot.subplot(1, 3, 1)
    matplotlib.pyplot.scatter(x1, x2, c=esperados, cmap="coolwarm", vmin=0, vmax=1)
    matplotlib.pyplot.title("Valor esperado")
    matplotlib.pyplot.xlabel(encabezado[0])
    matplotlib.pyplot.ylabel(encabezado[1])
    matplotlib.pyplot.subplot(1, 3, 2)
    matplotlib.pyplot.scatter(x1, x2, c=predichos, cmap="coolwarm", vmin=0, vmax=1)
    matplotlib.pyplot.title("Valor predicho")
    matplotlib.pyplot.xlabel(encabezado[0])
    matplotlib.pyplot.ylabel(encabezado[1])
    matplotlib.pyplot.subplot(1, 3, 3)
    matplotlib.pyplot.scatter(x1, x2, c=colores)
    matplotlib.pyplot.title("Coincidencia (verde = si, rojo = no)")
    matplotlib.pyplot.xlabel(encabezado[0])
    matplotlib.pyplot.ylabel(encabezado[1])
    matplotlib.pyplot.show()


    
nombre_archivo = input("Nombre del archivo csv: ")
encabezado, entradas, esperados = leer_csv(nombre_archivo)
print("Se leyeron", len(entradas), "filas")

continuar = "s"
while continuar == "s":
    # pedir los pesos
    sesgo = float(input("Peso del sesgo (w0): "))
    pesos = []
    for i in range(len(encabezado) - 1):
        w = float(input("Peso para " + encabezado[i] + ": "))
        pesos.append(w)

    # pedir la funcion de activacion
    print("1. Escalon")
    print("2. Sigmoide")
    activacion = int(input("Funcion de activacion: "))

    predichos = predecir(entradas, pesos, sesgo, activacion)

    # contar aciertos
    aciertos = 0
    for i in range(len(esperados)):
        if redondear(esperados[i]) == redondear(predichos[i]):
            aciertos = aciertos + 1
    print("Aciertos:", aciertos, "de", len(esperados))

    graficar(encabezado, entradas, esperados, predichos)

    continuar = input("Quiere probar con otros pesos? (s/n): ")

print("Fin :)")
