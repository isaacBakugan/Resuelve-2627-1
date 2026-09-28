# Nombre del integrante: Tomás Paraco
# Cédula del integrante: 30497481

# haga su tarea aqui

import matplotlib.pyplot as plt

RUTA_SUGERIDA = "assets/fuzzy_separables.csv"
E = 2.718281828459045  # número de Euler, para la sigmoide

COLOR_CLASE_POSITIVA = "blue"  # azul
COLOR_CLASE_NEGATIVA = "orange"  # naranja
COLOR_ACIERTO = "green"  # verde
COLOR_FALLO = "red"  # rojo


# ---------- Lectura del archivo ----------

def carpeta_del_programa():
    # Carpeta donde está este .py (sin usar os, se corta la ruta a mano)
    ruta = __file__.replace("\\", "/")
    if "/" not in ruta:
        return "."
    return ruta.rsplit("/", 1)[0]


def abrir_csv(ruta):
    # Primero prueba la ruta tal cual (relativa a donde se ejecuta);
    # si no existe, la busca relativa a la carpeta de este archivo.
    # utf-8-sig quita el BOM que traen los CSV al inicio.
    try:
        return open(ruta, encoding="utf-8-sig")
    except FileNotFoundError:
        return open(carpeta_del_programa() + "/" + ruta, encoding="utf-8-sig")


def leer_csv(ruta):
    # Devuelve las entradas de cada fila (lista de listas) y la salida esperada
    with abrir_csv(ruta) as archivo:
        lineas = archivo.read().splitlines()

    entradas = []
    esperados = []
    for linea in lineas[1:]:  # la primera línea es el encabezado
        linea = linea.strip()
        if linea == "":
            continue
        valores = [float(v) for v in linea.split(",")]
        entradas.append(valores[:-1])
        esperados.append(valores[-1])

    if len(entradas) == 0:
        raise ValueError("no tiene filas de datos")
    for fila in entradas:
        if len(fila) == 0 or len(fila) != len(entradas[0]):
            raise ValueError("todas las filas deben tener las mismas columnas, mínimo 2")
    return entradas, esperados


# ---------- Clases de la salida esperada ----------

def detectar_clases(esperados):
    # Si hay valores cercanos a -1 las clases son -1/1, si no son 0/1
    if min(esperados) < -0.5:
        return (-1, 1)
    return (0, 1)


def a_clase(valor, clases):
    # Lleva un valor a la clase más cercana. Hace falta porque
    # fuzzy_separables.csv trae valores como 0.2 o -0.1 que son clase 0.
    umbral = (clases[0] + clases[1]) / 2
    if valor >= umbral:
        return clases[1]
    return clases[0]


def convertir_a_clases(valores, clases):
    resultado = []
    difusos = 0
    for valor in valores:
        clase = a_clase(valor, clases)
        if clase != valor:
            difusos += 1
        resultado.append(clase)
    if difusos > 0:
        print(f"Aviso: {difusos} valores de y no eran exactamente {clases[0]} o {clases[1]}, "
              f"se tomaron como la clase más cercana.")
    return resultado


# ---------- Perceptrón ----------

def suma(entradas, pesos, sesgo):
    # z = b + w1*x1 + w2*x2 + ... + wn*xn
    total = sesgo
    for i in range(len(entradas)):
        total += pesos[i] * entradas[i]
    return total


def escalon(z):
    if z >= 0:
        return 1
    return 0


def sigmoide(z):
    # 1 / (1 + e^-z). Si z es muy negativo, e^-z se desborda y el resultado es 0
    if z < -700:
        return 0.0
    return 1 / (1 + E ** (-z))


def sigmoide_umbral(z):
    # La sigmoide da un valor entre 0 y 1; con umbral 0.5 se obtiene la clase
    if sigmoide(z) >= 0.5:
        return 1
    return 0


def signo(z):
    if z >= 0:
        return 1
    return -1


def tangente_hiperbolica(z):
    # tanh(z) = 2*sigmoide(2z) - 1, da un valor entre -1 y 1
    return 2 * sigmoide(2 * z) - 1


def tangente_umbral(z):
    if tangente_hiperbolica(z) >= 0:
        return 1
    return -1


def opciones_activacion(clases):
    # Solo se ofrecen funciones cuya salida coincide con las clases del CSV
    if clases == (0, 1):
        return [("Escalón", escalon), ("Sigmoide (umbral 0.5)", sigmoide_umbral)]
    return [("Signo", signo), ("Tangente hiperbólica (umbral 0)", tangente_umbral)]


def predecir(entradas, pesos, sesgo, activacion):
    predichos = []
    for x in entradas:
        predichos.append(activacion(suma(x, pesos, sesgo)))
    return predichos


def contar_aciertos(esperados, predichos):
    aciertos = 0
    for i in range(len(esperados)):
        if esperados[i] == predichos[i]:
            aciertos += 1
    return aciertos


# ---------- Entrada por consola ----------

def pedir_ruta():
    # Repite hasta que el archivo se pueda leer
    while True:
        ruta = input(f"Ruta del archivo CSV [{RUTA_SUGERIDA}]: ").strip().strip('"')
        if ruta == "":
            ruta = RUTA_SUGERIDA
        try:
            return leer_csv(ruta)
        except OSError:
            print("No se pudo abrir el archivo. Intente de nuevo.")
        except ValueError as error:
            print(f"El archivo no es válido ({error}). Intente de nuevo.")


def pedir_numero(mensaje):
    # Repite hasta que se escriba un número válido (acepta coma decimal)
    while True:
        texto = input(mensaje).strip().replace(",", ".")
        try:
            numero = float(texto)
        except ValueError:
            print("Valor inválido, escriba un número (por ejemplo -0.25).")
            continue
        return numero


def pedir_pesos(cantidad):
    sesgo = pedir_numero("Sesgo (b): ")
    pesos = []
    for i in range(cantidad):
        pesos.append(pedir_numero(f"Peso w{i + 1} (de x{i + 1}): "))
    return sesgo, pesos


def pedir_activacion(clases):
    opciones = opciones_activacion(clases)
    print("Funciones de activación:")
    for i in range(len(opciones)):
        print(f"  {i + 1}) {opciones[i][0]}")
    while True:
        eleccion = input("Elija una opción: ").strip()
        if eleccion.isdigit() and 1 <= int(eleccion) <= len(opciones):
            return opciones[int(eleccion) - 1]
        print(f"Opción inválida, escriba un número del 1 al {len(opciones)}.")


def pedir_si_no(mensaje):
    while True:
        respuesta = input(mensaje).strip().lower()
        if respuesta in ("s", "si", "sí"):
            return True
        if respuesta in ("n", "no"):
            return False
        print("Respuesta inválida, escriba s o n.")


# ---------- Gráficas ----------

def columna(entradas, indice):
    # Saca la columna x1 (indice 0) o x2 (indice 1).
    # Si el CSV tiene una sola entrada, x2 se toma como 0.
    valores = []
    for fila in entradas:
        if indice < len(fila):
            valores.append(fila[indice])
        else:
            valores.append(0)
    return valores


def filtrar(x1, x2, etiquetas, valor):
    # Devuelve solo los puntos cuya etiqueta es igual a valor
    xs = []
    ys = []
    for i in range(len(etiquetas)):
        if etiquetas[i] == valor:
            xs.append(x1[i])
            ys.append(x2[i])
    return xs, ys


def poner_textos(eje, titulo):
    eje.set_title(titulo)
    eje.set_xlabel("x1")
    eje.set_ylabel("x2")
    eje.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)


def graficar_clases(eje, x1, x2, valores, clases, titulo):
    xs, ys = filtrar(x1, x2, valores, clases[1])
    eje.scatter(xs, ys, color=COLOR_CLASE_POSITIVA, s=20, label=f"clase {clases[1]}")
    xs, ys = filtrar(x1, x2, valores, clases[0])
    eje.scatter(xs, ys, color=COLOR_CLASE_NEGATIVA, s=20, label=f"clase {clases[0]}")
    poner_textos(eje, titulo)


def graficar_coincidencia(eje, x1, x2, esperados, predichos):
    coincide = []
    for i in range(len(esperados)):
        coincide.append(esperados[i] == predichos[i])
    xs, ys = filtrar(x1, x2, coincide, True)
    eje.scatter(xs, ys, color=COLOR_ACIERTO, s=20, marker="o", label="coincide")
    xs, ys = filtrar(x1, x2, coincide, False)
    eje.scatter(xs, ys, color=COLOR_FALLO, s=40, marker="x", label="no coincide")
    poner_textos(eje, "Coincidencia (verde = acierta, rojo = falla)")


def graficar(entradas, esperados, predichos, clases, titulo_general):
    # Solo se grafican x1 y x2 aunque el CSV tenga más entradas
    x1 = columna(entradas, 0)
    x2 = columna(entradas, 1)
    figura, ejes = plt.subplots(1, 3, figsize=(16, 5.5))
    graficar_clases(ejes[0], x1, x2, esperados, clases, "Valor esperado")
    graficar_clases(ejes[1], x1, x2, predichos, clases, "Valor predicho")
    graficar_coincidencia(ejes[2], x1, x2, esperados, predichos)
    figura.suptitle(titulo_general)
    figura.tight_layout()
    plt.show()
    plt.close(figura)


# Main

def main():
    entradas, valores_y = pedir_ruta()
    clases = detectar_clases(valores_y)
    esperados = convertir_a_clases(valores_y, clases)
    cantidad = len(entradas[0])
    print(f"Se leyeron {len(entradas)} filas con {cantidad} entradas. "
          f"Clases: {clases[0]} y {clases[1]}.")
    if cantidad > 2:
        print("En las gráficas solo se muestran x1 y x2.")

    while True:
        sesgo, pesos = pedir_pesos(cantidad)
        nombre, activacion = pedir_activacion(clases)
        predichos = predecir(entradas, pesos, sesgo, activacion)

        aciertos = contar_aciertos(esperados, predichos)
        porcentaje = 100 * aciertos / len(esperados)
        print(f"Aciertos: {aciertos} de {len(esperados)} ({porcentaje:.1f}%)")
        if aciertos == len(esperados):
            print("Estos pesos separan todos los puntos.")
        else:
            print("Estos pesos no separan todos los puntos.")

        titulo = f"b = {sesgo}, w = {pesos}, {nombre}, aciertos = {porcentaje:.1f}%"
        graficar(entradas, esperados, predichos, clases, titulo)

        if not pedir_si_no("¿Deseas probar otros pesos? (s/n): "):
            break


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\nPrograma terminado.")
