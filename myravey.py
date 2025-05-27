import numpy as np
import random
import folium
from geopy.distance import geodesic
import time
import os
from typing import List, Dict, Tuple

# Конфигурация
DATA_FILE = 'points.csv'  # Файл с точками в формате: id,lat,lon,weight
SPEEDS = {
    'pedestrian': 5,
    'cyclist': 15,
    'car': 50
}

# Параметры муравьиного алгоритма
ANTS_COUNT = 50
ITERATIONS = 100
EVAPORATION_RATE = 0.5
ALPHA = 1.0  # Влияние феромона
BETA = 2.0  # Влияние эвристической информации
Q = 100  # Константа для обновления феромона


class Point:
    def __init__(self, id: int, lat: float, lon: float, weight: float):
        self.id = id
        self.lat = lat
        self.lon = lon
        self.weight = weight

    def __str__(self):
        return f"Point(id={self.id}, lat={self.lat}, lon={self.lon}, weight={self.weight})"


class Route:
    def __init__(self, points: List[Point], order: List[int] = None):
        self.points = points
        self.order = order if order else list(range(len(points)))
        self.total_time = 0
        self.total_weight = 0

    def calculate_metrics(self, max_time: float, transport: str) -> Tuple[float, float]:
        """Вычисляет время и вес маршрута"""
        self.total_time = 0
        self.total_weight = 0

        for i in range(len(self.order)):
            point_idx = self.order[i]
            self.total_weight += self.points[point_idx].weight

            if i > 0:
                prev_point_idx = self.order[i - 1]
                prev_point = self.points[prev_point_idx]
                curr_point = self.points[point_idx]

                distance = geodesic(
                    (prev_point.lat, prev_point.lon),
                    (curr_point.lat, curr_point.lon)
                ).km

                time_h = distance / SPEEDS[transport]
                self.total_time += time_h

        return self.total_time, self.total_weight

    def __len__(self):
        return len(self.order)

    def __getitem__(self, index):
        return self.points[self.order[index]]

    def __repr__(self):
        return f"Route(order={self.order}, time={self.total_time:.2f}h, weight={self.total_weight:.2f})"


class AntColonyOptimizer:
    def __init__(self, points: List[Point], max_time: float, transport: str):
        self.points = points
        self.max_time = max_time
        self.transport = transport
        self.num_points = len(points)
        self.pheromone = np.ones((self.num_points, self.num_points))
        self.distances = self._calculate_distances()
        self.heuristic = 1 / (self.distances + 1e-10)  # Эвристическая информация (обратная расстоянию)
        self.best_route = None
        self.best_weight = -1

    def _calculate_distances(self) -> np.ndarray:
        """Вычисляет матрицу расстояний между точками"""
        distances = np.zeros((self.num_points, self.num_points))
        for i in range(self.num_points):
            for j in range(self.num_points):
                if i != j:
                    distances[i][j] = geodesic(
                        (self.points[i].lat, self.points[i].lon),
                        (self.points[j].lat, self.points[j].lon)
                    ).km
        return distances

    def _select_next_point(self, current_point: int, visited: set) -> int:
        """Выбирает следующую точку для муравья"""
        probabilities = []
        total = 0.0

        for next_point in range(self.num_points):
            if next_point not in visited:
                pheromone = self.pheromone[current_point][next_point] ** ALPHA
                heuristic = self.heuristic[current_point][next_point] ** BETA
                prob = pheromone * heuristic
                probabilities.append((next_point, prob))
                total += prob

        # Нормализуем вероятности
        if total > 0:
            probabilities = [(point, prob / total) for point, prob in probabilities]
            probabilities.sort(key=lambda x: -x[1])
            points, probs = zip(*probabilities)
            return np.random.choice(points, p=probs)
        else:
            return random.choice([p for p in range(self.num_points) if p not in visited])

    def _construct_routes(self) -> List[Route]:
        """Строит маршруты для всех муравьев"""
        routes = []
        for _ in range(ANTS_COUNT):
            visited = {0}  # Начинаем с первой точки
            order = [0]

            while len(visited) < self.num_points:
                next_point = self._select_next_point(order[-1], visited)
                order.append(next_point)
                visited.add(next_point)

            route = Route(self.points, order)
            route.calculate_metrics(self.max_time, self.transport)
            routes.append(route)
        return routes

    def _update_pheromone(self, routes: List[Route]):
        """Обновляет феромоны на основе маршрутов муравьев"""
        # Испарение феромона
        self.pheromone *= (1 - EVAPORATION_RATE)

        # Добавление нового феромона
        for route in routes:
            if route.total_time <= self.max_time:
                delta_pheromone = Q * route.total_weight
                for i in range(len(route.order) - 1):
                    from_point = route.order[i]
                    to_point = route.order[i + 1]
                    self.pheromone[from_point][to_point] += delta_pheromone

    def run(self) -> Route:
        """Запускает муравьиный алгоритм"""
        for iteration in range(ITERATIONS):
            routes = self._construct_routes()
            self._update_pheromone(routes)

            # Находим лучший маршрут в текущей итерации
            valid_routes = [r for r in routes if r.total_time <= self.max_time]
            if valid_routes:
                current_best = max(valid_routes, key=lambda x: x.total_weight)
                if current_best.total_weight > self.best_weight:
                    self.best_route = current_best
                    self.best_weight = current_best.total_weight

            print(f"Iteration {iteration + 1}/{ITERATIONS}, Best weight: {self.best_weight:.2f}")

        return self.best_route


def load_points(filename: str) -> List[Point]:
    """Загружает точки из файла"""
    points = []
    try:
        if not os.path.exists(filename):
            print(f"Файл {filename} не найден. Создаю тестовые данные...")
            return create_test_data(filename)

        with open(filename, 'r') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split(',')
                    if len(parts) >= 4:
                        id = int(parts[0])
                        lat = float(parts[1])
                        lon = float(parts[2])
                        weight = float(parts[3])
                        points.append(Point(id, lat, lon, weight))
        if not points:
            raise ValueError("Файл пустой")
        return points
    except Exception as e:
        print(f"Ошибка при загрузке файла: {e}")
        return None



def plot_route_on_map(route: Route, transport: str, filename: str = 'route.html'):
    """Визуализирует маршрут на карте с использованием Folium"""
    if route is None or len(route) == 0:
        print("Нет точек для отображения")
        return

    # Создаем карту с центром в первой точке маршрута
    m = folium.Map(location=[route[0].lat, route[0].lon], zoom_start=13)

    # Добавляем точки маршрута
    for i, point in enumerate(route):
        folium.Marker(
            location=[point.lat, point.lon],
            popup=f"Точка {point.id}\nВес: {point.weight}",
            icon=folium.Icon(color='red' if i == 0 else 'blue')
        ).add_to(m)

    # Получаем координаты точек в порядке маршрута
    coords = [(point.lat, point.lon) for point in route]
    # Добавляем линии маршрута
    folium.PolyLine(
        locations=coords,
        color='blue',
        weight=3,
        opacity=0.8
    ).add_to(m)

    # Добавляем информацию о маршруте
    route_info = f"""
        <div style="position: fixed; bottom: 50px; left: 50px; width: 300px; height: 120px; 
                    background-color: white; z-index:9999; padding: 10px; border-radius: 5px;
                    box-shadow: 0 0 5px grey;">
            <b>Маршрут ({transport})</b><br>
            Всего точек: {len(route)}<br>
            Общий вес: {route.total_weight:.2f}<br>
            Общее время: {route.total_time:.2f} ч
        </div>
        """
    m.get_root().html.add_child(folium.Element(route_info))

    # Сохраняем карту
    m.save(filename)
    print(f"Карта сохранена в {filename}")


def main():
    # Загрузка точек
    points = load_points(DATA_FILE)

    if points is None:
        print("Не удалось загрузить или создать данные. Программа завершена.")
        return

    print(f"\nЗагружено {len(points)} точек:")
    for p in points:
        print(p)

    # Параметры задачи
    max_time_hours = 3.0  # Максимальное время в часах
    transport = 'pedestrian'  # Способ передвижения

    # Запуск муравьиного алгоритма
    print("\nЗапуск муравьиного алгоритма...")
    start_time = time.time()

    aco = AntColonyOptimizer(points, max_time_hours, transport)
    best_route = aco.run()

    if best_route is None:
        print("Не удалось найти подходящий маршрут с заданными ограничениями.")
        return

    print(f"\nОптимальный маршрут найден за {time.time() - start_time:.2f} секунд")
    print(best_route)

    # Визуализация
    plot_route_on_map(best_route, transport)


if __name__ == "__main__":
    main()
