import random
import math

# Попытка импорта matplotlib; если библиотека отсутствует — графики строиться не будут
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("[Предупреждение] matplotlib не найден. Установите: pip install matplotlib")

# ========================== КОНСТАНТЫ ЗАДАЧИ ==========================
X_MIN = 9                     # Минимальное допустимое значение x
X_MAX = 14                    # Максимальное допустимое значение x
X_COUNT = X_MAX - X_MIN + 1   # Количество допустимых значений (6)
CHROMOSOME_LENGTH = math.ceil(math.log2(X_COUNT))  # Длина хромосомы = 3 бита

# Глобальные параметры алгоритма (переопределяются при вводе с клавиатуры)
POPULATION_SIZE = 10
CROSSOVER_PROB = 0.70
MUTATION_PROB = 0.20
GENERATION_COUNT = 50


# ========================== ЦЕЛЕВАЯ ФУНКЦИЯ ==========================
def target_function(x):
    """Целевая функция: f(x) = x^2 + 0.1*x - 23  ->  максимизируется."""
    return x ** 2 + 0.1 * x - 23


# ===================== КОДИРОВАНИЕ / ДЕКОДИРОВАНИЕ =====================
def decode_chromosome(chromosome):
    """
    Преобразует бинарную хромосому в целое значение x.
    Хромосома хранит смещение относительно X_MIN.
    Возвращает None, если декодированное значение выходит за X_MAX.
    """
    offset = 0
    for bit in chromosome:
        offset = offset * 2 + bit
    x = X_MIN + offset
    return x if x <= X_MAX else None


def encode_chromosome(x):
    """Кодирует целое значение x в бинарную хромосому длины CHROMOSOME_LENGTH."""
    offset = x - X_MIN
    chrom = []
    for _ in range(CHROMOSOME_LENGTH):
        chrom.append(offset % 2)
        offset //= 2
    chrom.reverse()
    return chrom


# ===================== СОЗДАНИЕ ОСОБИ (СЛОВАРЬ) =====================
def make_individual(chromosome=None, x=None):
    """Создаёт особь в виде словаря с полями 'chromosome', 'x', 'fitness'."""
    if chromosome is not None:
        chrom = chromosome[:]
        xv = decode_chromosome(chrom)
    elif x is not None:
        xv = x
        chrom = encode_chromosome(x)
    else:
        chrom = [0] * CHROMOSOME_LENGTH
        xv = X_MIN

    return {
        'chromosome': chrom,
        'x': xv,
        'fitness': target_function(xv) if xv is not None else -float('inf')
    }


def copy_individual(ind):
    """Создаёт независимую копию особи-словаря."""
    return make_individual(chromosome=ind['chromosome'])


def ind_to_str(ind):
    """Строковое представление особи для отладки."""
    bits = ''.join(str(b) for b in ind['chromosome'])
    return f"Ind(x={ind['x']}, chrom={bits}, fit={ind['fitness']:.2f})"


# =================== СТРАТЕГИИ НАЧАЛЬНОЙ ПОПУЛЯЦИИ ===================
def init_population_blanket(size):
    """
    II(A) — Стратегия «одеяла»:
    равномерное случайное распределение по всему диапазону [X_MIN, X_MAX].
    """
    return [make_individual(x=random.randint(X_MIN, X_MAX)) for _ in range(size)]


def init_population_focusing(size):
    """
    II(C) — Стратегия фокусировки:
    начальные особи генерируются без крайних значений, т.е. из [X_MIN+1, X_MAX-1].
    """
    return [make_individual(x=random.randint(X_MIN + 1, X_MAX - 1)) for _ in range(size)]


# ===================== СЕЛЕКЦИЯ РОДИТЕЛЕЙ =====================
def select_parents_elitist(population, num_pairs):
    """
    III(C) — Элитная селекция:
    лучшая особь скрещивается со случайно выбранными остальными.
    """
    sorted_pop = sorted(population, key=lambda ind: ind['fitness'], reverse=True)
    best = copy_individual(sorted_pop[0])
    pairs = []
    for _ in range(num_pairs):
        other = copy_individual(random.choice(sorted_pop[1:]))
        pairs.append((best, other))
    return pairs


def hamming_distance(chrom1, chrom2):
    """Расстояние Хэмминга между двумя бинарными хромосомами."""
    return sum(a != b for a, b in zip(chrom1, chrom2))


def select_parents_inbreeding(population, num_pairs):
    """
    III(E) — Инбридинг:
    в качестве родителей выбираются наиболее похожие особи
    (с минимальным расстоянием Хэмминга).
    """
    pairs = []
    available = list(range(len(population)))

    for _ in range(num_pairs):
        if len(available) < 2:
            break

        min_dist = float('inf')
        best_pair = (available[0], available[1])

        for idx_i in range(len(available)):
            for idx_j in range(idx_i + 1, len(available)):
                i, j = available[idx_i], available[idx_j]
                dist = hamming_distance(population[i]['chromosome'],
                                        population[j]['chromosome'])
                if dist < min_dist:
                    min_dist = dist
                    best_pair = (i, j)

        pairs.append((copy_individual(population[best_pair[0]]),
                      copy_individual(population[best_pair[1]])))
        available.remove(best_pair[0])   # исключаем использованную особь
        available.remove(best_pair[1])

    return pairs


# ===================== ОПЕРАТОРЫ КРОССИНГОВЕРА =====================
def crossover_standard_one_point(parent1, parent2):
    """
    IV(A) — Стандартный одноточечный кроссинговер.
    """
    if random.random() > CROSSOVER_PROB:
        return []

    cut = random.randint(1, CHROMOSOME_LENGTH - 1)
    c1 = parent1['chromosome'][:cut] + parent2['chromosome'][cut:]
    c2 = parent2['chromosome'][:cut] + parent1['chromosome'][cut:]

    offspring = []
    for chrom in (c1, c2):
        x = decode_chromosome(chrom)
        if x is not None:
            offspring.append(make_individual(chromosome=chrom))
    return offspring


def crossover_ordering_one_point(parent1, parent2):
    """
    IV(E) — Упорядочивающий одноточечный кроссинговер:
    после стандартного разреза биты потомка упорядочиваются (сортируются).
    """
    if random.random() > CROSSOVER_PROB:
        return []

    cut = random.randint(1, CHROMOSOME_LENGTH - 1)
    c1 = parent1['chromosome'][:cut] + parent2['chromosome'][cut:]
    c2 = parent2['chromosome'][:cut] + parent1['chromosome'][cut:]

    # Упорядочивание — сортировка битов по возрастанию
    c1 = sorted(c1)
    c2 = sorted(c2)

    offspring = []
    for chrom in (c1, c2):
        x = decode_chromosome(chrom)
        if x is not None:
            offspring.append(make_individual(chromosome=chrom))
    return offspring


def crossover_golden_ratio(parent1, parent2):
    """
    IV(L) — Кроссинговер на основе «Золотого сечения»:
    точка разреза определяется пропорцией 0.618.
    """
    if random.random() > CROSSOVER_PROB:
        return []

    cut = round(0.618 * CHROMOSOME_LENGTH)
    cut = max(1, min(cut, CHROMOSOME_LENGTH - 1))

    c1 = parent1['chromosome'][:cut] + parent2['chromosome'][cut:]
    c2 = parent2['chromosome'][:cut] + parent1['chromosome'][cut:]

    offspring = []
    for chrom in (c1, c2):
        x = decode_chromosome(chrom)
        if x is not None:
            offspring.append(make_individual(chromosome=chrom))
    return offspring


# ===================== ОПЕРАТОРЫ МУТАЦИИ =====================
def mutate_simple(individual):
    """
    V(A) — Простая мутация:
    инверсия (замена 0 на 1 и наоборот) одного случайного гена.
    """
    if random.random() > MUTATION_PROB:
        return

    idx = random.randint(0, CHROMOSOME_LENGTH - 1)
    individual['chromosome'][idx] ^= 1
    individual['x'] = decode_chromosome(individual['chromosome'])
    if individual['x'] is not None:
        individual['fitness'] = target_function(individual['x'])


def mutate_translocation(individual):
    """
    V(H) — Транслокация:
    вырезание непрерывного сегмента хромосомы и вставка его в другую позицию.
    """
    if random.random() > MUTATION_PROB:
        return

    i, j = sorted([random.randint(0, CHROMOSOME_LENGTH - 1),
                   random.randint(0, CHROMOSOME_LENGTH - 1)])
    if i == j:
        return

    segment = individual['chromosome'][i:j]
    remaining = individual['chromosome'][:i] + individual['chromosome'][j:]

    if not remaining:
        return

    insert_pos = random.randint(0, len(remaining))
    new_chrom = remaining[:insert_pos] + segment + remaining[insert_pos:]

    if len(new_chrom) == CHROMOSOME_LENGTH:
        individual['chromosome'] = new_chrom
        individual['x'] = decode_chromosome(new_chrom)
        if individual['x'] is not None:
            individual['fitness'] = target_function(individual['x'])


# ===================== ПРОПОРЦИОНАЛЬНЫЙ ОТБОР =====================
def selection_proportional(pool, target_size):
    """
    VI(A) — Пропорциональный отбор (рулеточное колесо).
    Особи выбираются с вероятностью, пропорциональной их приспособленности.
    """
    total = sum(ind['fitness'] for ind in pool)
    if total <= 0:
        return [copy_individual(random.choice(pool)) for _ in range(target_size)]

    weights = [ind['fitness'] / total for ind in pool]
    chosen = random.choices(pool, weights=weights, k=target_size)
    return [copy_individual(ind) for ind in chosen]


# ===================== ПОСТРОЕНИЕ ГРАФИКА =====================
def plot_results(best_values, filename='genetic_algorithm_results.png'):
    """
    Строит график лучшего значения целевой функции по поколениям.
    Промежуточные точки — синие, финальная (оптимальная) — красная и крупная.
    """
    if not MATPLOTLIB_AVAILABLE:
        print("[Инфо] matplotlib недоступен — график не построен.")
        return

    gens = list(range(1, len(best_values) + 1))

    plt.figure(figsize=(12, 7))

    if len(best_values) > 1:
        plt.scatter(gens[:-1], best_values[:-1], c='blue', s=40,
                    label='Промежуточные значения', zorder=3)

    plt.scatter(gens[-1], best_values[-1], c='red', s=150,
                label='Оптимальное значение', zorder=5,
                edgecolors='black', linewidths=1)

    plt.plot(gens, best_values, 'b--', alpha=0.4, linewidth=1, zorder=2)

    plt.xlabel('Номер поколения', fontsize=12)
    plt.ylabel('Значение целевой функции f(x)', fontsize=12)
    plt.title(f'Генетический алгоритм:  max f(x) = x² + 0.1x − 23,  x ∈ [9, 14]; p_cr={CROSSOVER_PROB}, p_m={MUTATION_PROB}',
              fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"График сохранён: {filename}")
    plt.show()


# ===================== ВВОД ПАРАМЕТРОВ =====================
def safe_input(prompt):
    """Безопасный ввод: при EOF возвращает пустую строку."""
    try:
        return input(prompt)
    except EOFError:
        return ""


def input_parameters():
    """Диалоговый ввод параметров алгоритма с проверкой корректности.
    При недоступности stdin (например, запуск из скрипта) используются значения по умолчанию."""
    global POPULATION_SIZE, CROSSOVER_PROB, MUTATION_PROB, GENERATION_COUNT

    print("=" * 60)
    print("       НАСТРОЙКА ПАРАМЕТРОВ ГЕНЕТИЧЕСКОГО АЛГОРИТМА")
    print("=" * 60)

    # --- Размер популяции ---
    ans = safe_input("\nЗадать размер популяции вручную? (1 — да, 2 — нет, по умолчанию 10): ").strip()
    if ans == '1':
        while True:
            try:
                v = int(safe_input("  Размер популяции [10; 100]: "))
                if 10 <= v <= 100:
                    POPULATION_SIZE = v
                    break
                print("  Ошибка: значение вне диапазона [10; 100]")
            except ValueError:
                print("  Ошибка: требуется целое число")
    else:
        print(f"  Оставлено по умолчанию: {POPULATION_SIZE}")

    # --- Вероятность кроссинговера ---
    ans = safe_input("\nЗадать вероятность кроссинговера? (1 — да, 2 — нет, по умолчанию 0.7): ").strip()
    if ans == '1':
        while True:
            try:
                v = float(safe_input("  Вероятность кроссинговера [0.1; 1.0]: ").replace(',', '.'))
                if 0.1 <= v <= 1.0:
                    CROSSOVER_PROB = v
                    break
                print("  Ошибка: значение вне диапазона [0.1; 1.0]")
            except ValueError:
                print("  Ошибка: требуется число")
    else:
        print(f"  Оставлено по умолчанию: {CROSSOVER_PROB}")

    # --- Вероятность мутации ---
    ans = safe_input("\nЗадать вероятность мутации? (1 — да, 2 — нет, по умолчанию 0.2): ").strip()
    if ans == '1':
        while True:
            try:
                v = float(safe_input("  Вероятность мутации [0.1; 1.0]: ").replace(',', '.'))
                if 0.1 <= v <= 1.0:
                    MUTATION_PROB = v
                    break
                print("  Ошибка: значение вне диапазона [0.1; 1.0]")
            except ValueError:
                print("  Ошибка: требуется число")
    else:
        print(f"  Оставлено по умолчанию: {MUTATION_PROB}")

    # --- Число поколений ---
    ans = safe_input("\nЗадать число поколений? (1 — да, 2 — нет, по умолчанию 50): ").strip()
    if ans == '1':
        while True:
            try:
                v = int(safe_input("  Число поколений [50; 1000]: "))
                if 50 <= v <= 1000:
                    GENERATION_COUNT = v
                    break
                print("  Ошибка: значение вне диапазона [50; 1000]")
            except ValueError:
                print("  Ошибка: требуется целое число")
    else:
        print(f"  Оставлено по умолчанию: {GENERATION_COUNT}")

    print("\n" + "=" * 60)


# ===================== ГЛАВНЫЙ ЦИКЛ =====================
def main():
    random.seed()
    input_parameters()

    # --- Выбор стратегии начальной популяции ---
    print("\nСТРАТЕГИЯ СОЗДАНИЯ НАЧАЛЬНОЙ ПОПУЛЯЦИИ:")
    print("  1 — Стратегия 'одеяла' (равномерное распределение)")
    print("  2 — Стратегия фокусировки (центр диапазона)")

    while True:
        choice = safe_input("Выберите (1 или 2): ").strip()
        if choice == '1':
            population = init_population_blanket(POPULATION_SIZE)
            print("Выбрана стратегия 'одеяла'.\n")
            break
        elif choice == '2':
            population = init_population_focusing(POPULATION_SIZE)
            print("Выбрана стратегия фокусировки.\n")
            break
        else:
            # При недоступности stdin выбираем 'одеяло' по умолчанию
            if choice == "":
                population = init_population_blanket(POPULATION_SIZE)
                print("Выбрана стратегия 'одеяла' (по умолчанию).\n")
                break
            print("Неверный ввод. Повторите.")

    best_history = []

    print("-" * 60)
    print("Запуск эволюции (микроэволюция)...")
    print("-" * 60)

    # ===================== VII(A) — МИКРОЭВОЛЮЦИЯ =====================
    for generation in range(GENERATION_COUNT):
        # 1) Селекция родителей: элитная + инбридинг
        parent_pairs = []
        parent_pairs.extend(select_parents_elitist(population, 2))
        parent_pairs.extend(select_parents_inbreeding(population, 2))

        # 2) Кроссинговер: стандартный, упорядочивающий, золотое сечение
        offspring = []
        for p1, p2 in parent_pairs:
            offspring.extend(crossover_standard_one_point(p1, p2))
            offspring.extend(crossover_ordering_one_point(p1, p2))
            offspring.extend(crossover_golden_ratio(p1, p2))

        # 3) Мутация: простая + транслокация
        for child in offspring:
            mutate_simple(child)
            mutate_translocation(child)

        # Отбираем только валидных потомков
        valid_offspring = [ch for ch in offspring if ch['x'] is not None]

        # 4) Формирование нового поколения (микроэволюция + пропорциональный отбор)
        pool = population + valid_offspring
        best_in_pool = max(pool, key=lambda ind: ind['fitness'])

        # Пропорциональный отбор на (POPULATION_SIZE − 1) мест;
        # последнее место гарантированно занимает лучшая особь (элитизм)
        new_population = selection_proportional(pool, POPULATION_SIZE - 1)
        new_population.append(copy_individual(best_in_pool))
        population = new_population

        # Сохраняем лучшее значение текущего поколения для графика
        gen_best = max(ind['fitness'] for ind in population)
        best_history.append(gen_best)
        print(f"  Поколение {generation + 1:3d}: best_fitness = {gen_best:.4f}, "
              f"разнообразие x = {sorted(set(ind['x'] for ind in population))}")

    print("-" * 60)
    print("Эволюция завершена.\n")

    # ===================== РЕЗУЛЬТАТЫ =====================
    best_ind = max(population, key=lambda ind: ind['fitness'])

    print("=" * 60)
    print("РЕЗУЛЬТАТЫ РАБОТЫ АЛГОРИТМА")
    print("=" * 60)
    print(f"Лучшее найденное решение:")
    print(f"  x        = {best_ind['x']}")
    print(f"  f(x)     = {best_ind['fitness']:.4f}")
    print(f"  Хромосома: {''.join(str(b) for b in best_ind['chromosome'])}")

    # Теоретический максимум для справки
    theoretical_x = max(range(X_MIN, X_MAX + 1), key=lambda xv: target_function(xv))
    theoretical_f = target_function(theoretical_x)
    print(f"\nТеоретический максимум:")
    print(f"  x        = {theoretical_x}")
    print(f"  f(x)     = {theoretical_f:.4f}")

    # Построение графика
    print("\nПостроение графика...")
    plot_results(best_history)

    print("\nПрограмма завершена.")


if __name__ == "__main__":
    main()
