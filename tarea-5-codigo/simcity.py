# Equipo (nombre del grupo): Resuelve
# Integrantes (nombre y cédula, uno por línea):
# - Tomás Paraco, 30497481
# - Arturo Pulgar, 29920034
# - Christian Goncalves, 30224303

# Tarea 5 - Simplified SimCity (cellular automaton) in text mode.
#
# Design decisions (points the assignment leaves open):
#  1. Distance "10 cells": Chebyshev distance (max of the column and row difference)
#     between the top-left cells of the two zones. All zones are 3x3, so this is the
#     same as measuring between their centers.
#  2. Votes: growth votes count 1 each, decline votes count 2 each.
#     balance = growth - decline. If balance > 0 the zone goes up 1 level,
#     if balance < 0 it goes down 1 level, if it is 0 it stays the same.
#  3. Levels are always kept between 0 and 27.
#  4. Traffic of a road cell = sum of sqrt(level) of the distinct zones touching it
#     (next to a level 16 zone: 4; between two level 16 zones: 8).
#     A road cell that touches no zone takes the traffic of its busiest adjacent road
#     cell (the adjacent road cell counts as its level). Only the traffic that comes from
#     zones is passed on, so traffic spreads one cell along the road and never feeds
#     on itself.
#  5. Traffic is shown rounded to an integer and only when it is greater than 1.
#     Horizontal: "=5=" or "=12"; vertical: "|5|" or "|12". Crossings always show ":+:".
#  6. In the first tick (or right after loading) nothing "decreased in the previous tick".
#  7. A road can only cross another road perpendicularly (the cell becomes ":+:").
#     Overlapping a parallel road, a zone, or going outside the grid is an error.
#     Zones must fit completely inside the grid and only on empty cells.
#  8. "Adjacent road cells" of a zone are the road cells right next to its border
#     (up, down, left, right). The rule about road cells whose traffic decreased
#     (commercial zones) also uses only the adjacent road cells.
#  9. Each tick: (a) every zone votes using the current state, (b) all zones change at
#     the same time, (c) traffic is recalculated with the new levels.
# 10. File format (ciudad.txt): one line per element.
#       zone <type> <column> <row> <level>
#       road <column> <row> <end>      (same syntax as the "a" command)
#       road <column> <row> <end> <h|v>  (one road cell, used by "guardar")
#     Lines starting with "#" are comments.

import math
import sys

COLUMNS = 29
ROWS = 15
ZONE_SIZE = 3
MAX_LEVEL = 27
NEAR_DISTANCE = 10
ZONE_TYPES = {"r": "residential", "c": "commercial", "i": "industrial"}


# ---------- Coordinates ----------

def column_name(index):
    # 0 -> A, 25 -> Z, 26 -> AA ...
    name = ""
    index += 1
    while index > 0:
        index, rest = divmod(index - 1, 26)
        name = chr(ord("A") + rest) + name
    return name


def column_index(text):
    text = text.strip().upper()
    if not text.isalpha():
        return None
    index = 0
    for letter in text:
        index = index * 26 + (ord(letter) - ord("A") + 1)
    index -= 1
    return index if 0 <= index < COLUMNS else None


def row_index(text):
    if not text.strip().isdigit():
        return None
    index = int(text) - 1
    return index if 0 <= index < ROWS else None


def neighbors(col, row):
    for dc, dr in ((0, -1), (0, 1), (-1, 0), (1, 0)):
        c, r = col + dc, row + dr
        if 0 <= c < COLUMNS and 0 <= r < ROWS:
            yield c, r


# ---------- City ----------

def new_city():
    # cells: (col, row) -> ("zone", zone) or ("road", "h" | "v" | "x")
    return {"cells": {}, "zones": [], "traffic": {}, "base": {}, "traffic_dropped": set()}


def zone_cells(col, row):
    return [(col + dc, row + dr) for dr in range(ZONE_SIZE) for dc in range(ZONE_SIZE)]


def add_zone(city, kind, col, row, level=0):
    cells = zone_cells(col, row)
    if col + ZONE_SIZE > COLUMNS or row + ZONE_SIZE > ROWS:
        return "Error: la zona no cabe en la cuadrícula (ocupa 3x3 celdas)."
    if any(cell in city["cells"] for cell in cells):
        return "Error: la celda está ocupada."
    zone = {"kind": kind, "col": col, "row": row, "level": level, "shrank": False}
    city["zones"].append(zone)
    for cell in cells:
        city["cells"][cell] = ("zone", zone)
    update_traffic(city, track_drops=False)
    return None


def road_cells(col, row, end):
    # end is a column (horizontal road) or a row (vertical road)
    if end.isalpha():
        end_col = column_index(end)
        if end_col is None:
            return None, None
        first, last = sorted((col, end_col))
        return "h", [(c, row) for c in range(first, last + 1)]
    end_row = row_index(end)
    if end_row is None:
        return None, None
    first, last = sorted((row, end_row))
    return "v", [(col, r) for r in range(first, last + 1)]


def add_road(city, col, row, end):
    direction, cells = road_cells(col, row, end)
    if cells is None:
        return "Error: la arteria se sale de la cuadrícula."
    for cell in cells:
        current = city["cells"].get(cell)
        if current is None:
            continue
        if current[0] == "zone":
            return "Error: la celda está ocupada por una zona."
        if current[1] == direction or current[1] == "x":
            return "Error: la arteria se solapa con otra arteria."
    for cell in cells:
        current = city["cells"].get(cell)
        city["cells"][cell] = ("road", "x" if current else direction)
    update_traffic(city, track_drops=False)
    return None


def road_list(city):
    return [cell for cell, content in city["cells"].items() if content[0] == "road"]


def update_traffic(city, track_drops=True):
    roads = road_list(city)
    base = {}
    for cell in roads:
        touching = {}  # distinct zones next to this cell
        for near in neighbors(*cell):
            content = city["cells"].get(near)
            if content and content[0] == "zone":
                touching[id(content[1])] = content[1]["level"]
        base[cell] = sum(math.sqrt(level) for level in touching.values())

    traffic = {}
    for cell in roads:
        traffic[cell] = base[cell]
        if base[cell] == 0:
            traffic[cell] = max([base[near] for near in neighbors(*cell) if near in base], default=0)

    if track_drops:
        old = city["traffic"]
        city["traffic_dropped"] = {cell for cell in roads if cell in old and traffic[cell] < old[cell]}
    city["base"] = base
    city["traffic"] = traffic


# ---------- Rules ----------

def distance(zone_1, zone_2):
    return max(abs(zone_1["col"] - zone_2["col"]), abs(zone_1["row"] - zone_2["row"]))


def adjacent_roads(city, zone):
    own = set(zone_cells(zone["col"], zone["row"]))
    roads = set()
    for cell in own:
        for near in neighbors(*cell):
            content = city["cells"].get(near)
            if near not in own and content and content[0] == "road":
                roads.add(near)
    return roads


def votes(city, zone):
    near = [z for z in city["zones"] if z is not zone and distance(z, zone) <= NEAR_DISTANCE]
    residential = [z for z in near if z["kind"] == "r"]
    commercial = [z for z in near if z["kind"] == "c"]
    industrial = [z for z in near if z["kind"] == "i"]
    roads = adjacent_roads(city, zone)
    busy_roads = [cell for cell in roads if city["traffic"].get(cell, 0) > 5]
    root = math.sqrt(zone["level"])
    grow = decline = 0

    if zone["kind"] == "r":
        grow += len(roads)
        grow += sum(1 for z in industrial if z["level"] < 18)
        grow += sum(1 for z in commercial if z["level"] > root)
        decline += 2 * len(busy_roads)
        decline += 2 * max(0, len(industrial) - 10)  # from the 11th industrial zone on
        decline += 2 * sum(1 for z in commercial if z["shrank"])
    elif zone["kind"] == "c":
        grow += len(busy_roads)
        grow += len(residential)
        grow += sum(1 for z in industrial if z["level"] > root)
        decline += 2 * sum(1 for cell in roads if cell in city["traffic_dropped"])
        decline += 2 * sum(1 for z in residential if z["shrank"])
        decline += 2 * sum(1 for z in industrial if z["shrank"])
    else:
        grow += len(roads)
        grow += len(residential)
        grow += sum(1 for z in commercial if z["level"] > root)
        decline += 2 * sum(1 for z in near if z["shrank"])
    return grow, decline


def tick(city):
    changes = []
    for zone in city["zones"]:
        grow, decline = votes(city, zone)
        changes.append((zone, grow - decline))
    for zone, balance in changes:
        old = zone["level"]
        if balance > 0:
            zone["level"] = min(MAX_LEVEL, old + 1)
        elif balance < 0:
            zone["level"] = max(0, old - 1)
        zone["shrank"] = zone["level"] < old
    update_traffic(city)


# ---------- Drawing ----------

def draw_road(city, cell, direction):
    if direction == "x":
        return ":+:"
    value = round(city["traffic"].get(cell, 0))
    text = str(min(value, 99)) if value > 1 else ""
    if direction == "h":
        return "=" * (3 - len(text)) + text if len(text) == 2 else (f"={text}=" if text else "===")
    return f"|{text}" if len(text) == 2 else (f"|{text}|" if text else "| |")


def draw_zone_cell(zone, col, row):
    letters = []
    for i in range(3):
        position = (row - zone["row"]) * 9 + (col - zone["col"]) * 3 + i  # reading order
        letter = zone["kind"]
        letters.append(letter.upper() if position < zone["level"] else letter)
    return "".join(letters)


def draw(city):
    lines = [("  " + "".join(f"{column_name(c):^3}" for c in range(COLUMNS))).rstrip()]
    for row in range(ROWS):
        line = f"{row + 1:<2}"
        for col in range(COLUMNS):
            content = city["cells"].get((col, row))
            if content is None:
                line += "   "
            elif content[0] == "zone":
                line += draw_zone_cell(content[1], col, row)
            else:
                line += draw_road(city, (col, row), content[1])
        lines.append(line.rstrip())
    print("\n".join(lines))


# ---------- Save and load ----------

def save(city, path):
    lines = ["# SimCity - ciudad guardada"]
    for zone in city["zones"]:
        lines.append(f"zone {zone['kind']} {column_name(zone['col'])} {zone['row'] + 1} {zone['level']}")
    # roads are saved cell by cell with their direction, so crossings are kept
    for (col, row), content in sorted(city["cells"].items(), key=lambda item: (item[0][1], item[0][0])):
        if content[0] == "road":
            directions = ["h", "v"] if content[1] == "x" else [content[1]]
            for direction in directions:
                end = column_name(col) if direction == "h" else str(row + 1)
                lines.append(f"road {column_name(col)} {row + 1} {end} {direction}")
    with open(path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")


def load(path):
    city = new_city()
    with open(path, encoding="utf-8") as file:
        lines = file.read().splitlines()
    for number, line in enumerate(lines, start=1):
        parts = line.split()
        if not parts or parts[0].startswith("#"):
            continue
        error = None
        if parts[0] == "zone" and len(parts) == 5 and parts[1] in ZONE_TYPES:
            col, row = column_index(parts[2]), row_index(parts[3])
            if col is None or row is None or not parts[4].isdigit():
                error = "coordenada inválida"
            else:
                error = add_zone(city, parts[1], col, row, min(MAX_LEVEL, int(parts[4])))
        elif parts[0] == "road" and len(parts) in (4, 5):
            col, row = column_index(parts[1]), row_index(parts[2])
            if col is None or row is None:
                error = "coordenada inválida"
            elif len(parts) == 5:
                error = add_road_cell(city, col, row, parts[4])
            else:
                error = add_road(city, col, row, parts[3])
        else:
            error = "línea no reconocida"
        if error:
            raise ValueError(f"línea {number}: {error}")
    update_traffic(city, track_drops=False)
    return city


def add_road_cell(city, col, row, direction):
    # one road cell saved with its direction ("h" or "v"); two of them make a crossing
    if direction not in ("h", "v"):
        return "dirección de arteria inválida"
    current = city["cells"].get((col, row))
    if current and (current[0] == "zone" or current[1] in (direction, "x")):
        return "celda ocupada"
    city["cells"][(col, row)] = ("road", "x" if current else direction)
    return None


# ---------- Commands ----------

def run_command(city, text):
    # returns (city, changed); changed = True when the grid must be printed
    parts = text.split()
    command = parts[0].lower()

    if command in ZONE_TYPES:
        if len(parts) != 3:
            print(f"Error: uso: {command} <columna> <fila>  (por ejemplo: {command} c 4)")
            return city, False
        col, row = column_index(parts[1]), row_index(parts[2])
        if col is None or row is None:
            print("Error: la celda no existe en la cuadrícula.")
            return city, False
        error = add_zone(city, command, col, row)
    elif command == "a":
        if len(parts) != 4:
            print("Error: uso: a <columna> <fila> <fin>  (por ejemplo: a c 7 p  o  a f 4 9)")
            return city, False
        col, row = column_index(parts[1]), row_index(parts[2])
        if col is None or row is None:
            print("Error: la celda no existe en la cuadrícula.")
            return city, False
        error = add_road(city, col, row, parts[3])
    elif command == "guardar":
        if len(parts) != 2:
            print("Error: uso: guardar <ruta_archivo>")
            return city, False
        try:
            save(city, parts[1])
            print(f"Ciudad guardada en {parts[1]}")
        except OSError:
            print("Error: no se pudo guardar el archivo.")
        return city, False
    elif command == "cargar":
        if len(parts) != 2:
            print("Error: uso: cargar <ruta_archivo>")
            return city, False
        try:
            return load(parts[1]), True
        except FileNotFoundError:
            print("Error: el archivo no existe.")
        except (OSError, ValueError) as problem:
            print(f"Error: no se pudo cargar la ciudad ({problem}).")
        return city, False
    else:
        print("Error: comando inválido. Comandos: r, c, i, a, guardar, cargar, salir o Enter para un tick.")
        return city, False

    if error:
        print(error)
        return city, False
    return city, True


def main():
    city = new_city()
    draw(city)
    while True:
        try:
            text = input("> ").strip()
        except EOFError:
            print()
            break
        if text == "":
            tick(city)
            draw(city)
            continue
        if text.lower() == "salir":
            break
        city, changed = run_command(city, text)
        if changed:
            draw(city)
    print("Fin del juego.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nFin del juego.")
    sys.exit(0)
