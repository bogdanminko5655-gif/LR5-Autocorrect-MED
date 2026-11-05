"""ЛР №5 — автокорекція тексту, ймовірності, мінімальна відстань редагування.

Python 3.10+. Виконання: python lab5.py
Використовує наданий shakespeare.txt; якщо його немає, завантажує твори з NLTK Gutenberg.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np

ALPHABET = 'abcdefghijklmnopqrstuvwxyz'
TOKEN_RE = re.compile(r"[a-z]+", flags=re.IGNORECASE)
CORPUS_FILES = ('shakespeare-hamlet.txt', 'shakespeare-macbeth.txt', 'shakespeare-caesar.txt')
SOURCE_URL = 'https://www.gutenberg.org/ebooks/100'


def process_data(file_name: str | Path) -> list[str]:
    """Reads and tokenizes a UTF-8 English language corpus."""
    text = Path(file_name).read_text(encoding='utf-8-sig', errors='replace').lower()
    return TOKEN_RE.findall(text)


def get_count(word_l: Iterable[str]) -> dict[str, int]:
    return dict(Counter(word_l))


def get_probs(word_count_dict: dict[str, int]) -> dict[str, float]:
    total = sum(word_count_dict.values())
    return {word: count / total for word, count in word_count_dict.items()} if total else {}


def delete_letter(word: str, verbose: bool = False) -> list[str]:
    result = [word[:i] + word[i + 1:] for i in range(len(word))]
    if verbose: print('delete:', result)
    return result


def switch_letter(word: str, verbose: bool = False) -> list[str]:
    result = [word[:i] + word[i + 1] + word[i] + word[i + 2:]
              for i in range(len(word) - 1) if word[i] != word[i + 1]]
    if verbose: print('switch:', result)
    return result


def replace_letter(word: str, verbose: bool = False) -> list[str]:
    result = [word[:i] + c + word[i + 1:]
              for i in range(len(word)) for c in ALPHABET if c != word[i]]
    if verbose: print('replace:', result[:20])
    return result


def insert_letter(word: str, verbose: bool = False) -> list[str]:
    result = [word[:i] + c + word[i:]
              for i in range(len(word) + 1) for c in ALPHABET]
    if verbose: print('insert:', result[:20])
    return result


def edit_one_letter(word: str, allow_switches: bool = True) -> set[str]:
    return (set(delete_letter(word)) | set(replace_letter(word)) |
            set(insert_letter(word)) | (set(switch_letter(word)) if allow_switches else set()))


def edit_two_letters(word: str, allow_switches: bool = True,
                     vocab: set[str] | None = None) -> set[str]:
    """Two successive edits; optional vocab filter avoids retaining all strings."""
    if vocab is None:
        result: set[str] = set()
        for candidate in edit_one_letter(word, allow_switches):
            result.update(edit_one_letter(candidate, allow_switches))
        return result
    result = set()
    for candidate in edit_one_letter(word, allow_switches):
        result.update(edit_one_letter(candidate, allow_switches) & vocab)
    return result


def get_corrections(word: str, probs: dict[str, float], vocab: set[str],
                    n: int = 3, verbose: bool = False,
                    allow_switches: bool = True) -> list[tuple[str, float]]:
    """Prefer known word, then one-edit candidates, then two-edit candidates.

    Frequency P(c) ranks candidates *within* the nearest edit tier; this is
    a simplified language model, not a measured P(typed|correct) error model.
    """
    word = word.lower()
    if word in vocab:
        candidates = {word}
    else:
        candidates = edit_one_letter(word, allow_switches) & vocab
        if not candidates:
            candidates = edit_two_letters(word, allow_switches, vocab)
        if not candidates:
            candidates = {word}
    result = sorted(((w, probs.get(w, 0.0)) for w in candidates),
                    key=lambda x: (-x[1], x[0]))[:max(0, n)]
    if verbose: print(f'{word}: {result}')
    return result


def correct_text(text: str, probs: dict[str, float], vocab: set[str]) -> str:
    """Corrects alphabetic tokens while retaining punctuation and letter case."""
    def replace(match: re.Match[str]) -> str:
        original = match.group(0)
        candidate = get_corrections(original, probs, vocab, n=1)[0][0]
        if original.isupper(): return candidate.upper()
        if original.istitle(): return candidate.capitalize()
        return candidate
    return TOKEN_RE.sub(replace, text)


def min_edit_distance(source: str, target: str,
                      ins_cost: int = 1, del_cost: int = 1,
                      rep_cost: int = 2) -> tuple[np.ndarray, int]:
    """Wagner-Fischer weighted edit distance (insert/delete/replace).

    Adjacent transpose is part of candidate generation but is NOT a primitive
    of this traditional Levenshtein-style distance recurrence.
    """
    if min(ins_cost, del_cost, rep_cost) < 0:
        raise ValueError('Вартість редагування має бути невід’ємною')
    m, n = len(source), len(target)
    D = np.zeros((m + 1, n + 1), dtype=np.int64)
    D[:, 0] = np.arange(m + 1) * del_cost
    D[0, :] = np.arange(n + 1) * ins_cost
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if source[i - 1] == target[j - 1] else rep_cost
            D[i, j] = min(D[i - 1, j] + del_cost,
                          D[i, j - 1] + ins_cost,
                          D[i - 1, j - 1] + cost)
    return D, int(D[m, n])


def ensure_corpus(path: str | Path) -> tuple[Path, str]:
    """Uses user corpus if present, otherwise NLTK Gutenberg Shakespeare plays.

    NLTK downloads its corpus over the internet on first use (e.g. Colab).
    """
    path = Path(path)
    if path.exists() and path.stat().st_size > 0:
        return path, 'Локальний файл користувача: ' + str(path)
    try:
        import nltk
        from nltk.corpus import gutenberg
        try:
            nltk.data.find('corpora/gutenberg')
        except LookupError:
            if not nltk.download('gutenberg', quiet=True):
                raise RuntimeError('Не вдалося отримати корпус Gutenberg через NLTK')
        text_parts = [gutenberg.raw(book) for book in CORPUS_FILES]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('\n\n'.join(text_parts), encoding='utf-8')
        return path, 'NLTK Gutenberg: Hamlet, Macbeth, Julius Caesar (Shakespeare)'
    except Exception as exc:
        raise RuntimeError(
            'Не вдалося завантажити корпус Шекспіра. Перевір інтернет у Colab; '
            'або завантаж текст із https://www.gutenberg.org/ebooks/100 '
            'і збережи його як data/shakespeare.txt. '
            f'Технічна причина: {exc}'
        ) from exc


def osa_distance(source: str, target: str) -> int:
    """Restricted Damerau–Levenshtein distance, with adjacent swap cost 1.

    Used only to validate that synthetic typos truly have 1 or 2 edits.
    Main task's MED uses the assignment's weighted costs and remains separate.
    """
    m, n = len(source), len(target)
    D = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1): D[i][0] = i
    for j in range(n + 1): D[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            D[i][j] = min(D[i - 1][j] + 1, D[i][j - 1] + 1,
                          D[i - 1][j - 1] + (source[i - 1] != target[j - 1]))
            if (i > 1 and j > 1 and source[i - 1] == target[j - 2]
                    and source[i - 2] == target[j - 1]):
                D[i][j] = min(D[i][j], D[i - 2][j - 2] + 1)
    return D[m][n]


def mutate(word: str, rng: random.Random, edits: int) -> str:
    """Introduces artificial spelling errors for reproducible evaluation."""
    s = word
    for _ in range(edits):
        operations = ["delete", "switch", "replace", "insert"]
        rng.shuffle(operations)
        for operation in operations:
            if operation == 'delete' and len(s) > 2:
                pos = rng.randrange(len(s)); t = s[:pos] + s[pos + 1:]
            elif operation == 'switch' and len(s) > 1:
                indices = [i for i in range(len(s) - 1) if s[i] != s[i + 1]]
                if not indices: continue
                pos = rng.choice(indices); t = s[:pos] + s[pos+1] + s[pos] + s[pos+2:]
            elif operation == 'replace':
                pos = rng.randrange(len(s)); letter = rng.choice(ALPHABET.replace(s[pos], ''))
                t = s[:pos] + letter + s[pos+1:]
            elif operation == 'insert':
                pos = rng.randrange(len(s)+1); t = s[:pos] + rng.choice(ALPHABET) + s[pos:]
            else: continue
            if t != s:
                s = t
                break
    return s


def generate_test_cases(counts: dict[str, int], sample_size: int = 100,
                        two_edit_fraction: float = .25,
                        seed: int = 42) -> list[dict]:
    """Choose frequent known words and inject OOV misspellings (not real-world errors).

    Results measure synthetic correction accuracy, not general spelling quality.
    """
    rng = random.Random(seed)
    vocabulary = set(counts)
    eligible = [w for w, c in counts.items()
                if 4 <= len(w) <= 9 and c >= 3 and w.isalpha()]
    rng.shuffle(eligible)
    if not eligible: raise ValueError('Корпус замалий для тестування: бракує частотних слів')
    sample_size = min(sample_size, len(eligible))
    cases = []
    # Distinct targets keep results easy to audit.
    for word in eligible:
        if len(cases) >= sample_size: break
        edits = 2 if (len(cases) % 4 == 3 and two_edit_fraction >= 0.24) else 1
        # General fractions use a seeded RNG; leave default deterministic 1 in 4.
        if two_edit_fraction != .25:
            edits = 2 if rng.random() < two_edit_fraction else 1
        for _ in range(30):
            wrong = mutate(word, rng, edits)
            if (wrong != word and wrong not in vocabulary and wrong.isalpha()
                    and osa_distance(word, wrong) == edits):
                cases.append({'typed': wrong, 'expected': word, 'edits': edits,
                              'target_count': counts[word]})
                break
    if not cases: raise ValueError('Не вдалося сформувати тестові опечатки')
    return cases


def evaluate(cases: list[dict], probs: dict[str, float], vocab: set[str]) -> tuple[dict, list[dict]]:
    rows = []
    for i, case in enumerate(cases, 1):
        start = time.perf_counter()
        suggestions = get_corrections(case['typed'], probs, vocab, n=3)
        elapsed_ms = (time.perf_counter() - start) * 1000
        got = suggestions[0][0] if suggestions else case['typed']
        rows.append({**case, 'case': i, 'predicted': got,
                     'top3': ' | '.join(w for w, _ in suggestions),
                     'correct': int(got == case['expected']),
                     'top3_correct': int(case['expected'] in [w for w, _ in suggestions]),
                     'time_ms': round(elapsed_ms, 3)})
    n = len(rows)
    one = [r for r in rows if r['edits'] == 1]
    two = [r for r in rows if r['edits'] == 2]
    def accuracy(rr: list[dict], key='correct') -> float | None:
        return round(sum(x[key] for x in rr) / len(rr), 6) if rr else None
    metrics = {
        'test_count': n,
        'correct_top1': sum(r['correct'] for r in rows),
        'accuracy_top1': accuracy(rows),
        'accuracy_top3': accuracy(rows, 'top3_correct'),
        'one_edit_count': len(one), 'accuracy_one_edit': accuracy(one),
        'two_edit_count': len(two), 'accuracy_two_edits': accuracy(two),
        'avg_correction_ms': round(sum(r['time_ms'] for r in rows) / n, 3),
        'median_correction_ms': round(float(np.median([r['time_ms'] for r in rows])), 3),
        'total_correction_seconds': round(sum(r['time_ms'] for r in rows) / 1000, 3),
        'evaluation_type': 'Штучно згенеровані опечатки у словах із корпусу; не незалежний набір реальних помилок',
    }
    return metrics, rows


def csv_write(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if not rows: return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(rows[0])
    with path.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_figures(metrics: dict, rows: list[dict], counts: dict[str, int],
                 result_dir: Path, docs_dir: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # Plot 1: scores by edit complexity.
    labels = ['Top-1', 'Top-3', '1 edit', '2 edits']
    scores = [metrics[k] for k in ('accuracy_top1', 'accuracy_top3',
                                  'accuracy_one_edit', 'accuracy_two_edits')]
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ['#3b82f6', '#60a5fa', '#14b8a6', '#fb923c']
    ax.bar(labels, [v * 100 if v is not None else 0 for v in scores], color=colors)
    ax.set_ylim(0, 105); ax.set_ylabel('Accuracy, %'); ax.set_title('Autocorrection — synthetic typo test')
    for i, v in enumerate(scores):
        ax.text(i, (v or 0) * 100 + 1.7, f'{(v or 0)*100:.1f}%', ha='center')
    ax.grid(axis='y', alpha=.2); fig.tight_layout()
    fig.savefig(result_dir / 'accuracy.png', dpi=160); fig.savefig(docs_dir / 'accuracy.png', dpi=160)
    plt.close(fig)
    # Plot 2: most frequent words in corpus.
    common = Counter(counts).most_common(18)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh([w for w, _ in common][::-1], [c for _, c in common][::-1], color='#6366f1')
    ax.set_title('Most frequent words in Shakespeare corpus'); ax.set_xlabel('Occurrences')
    ax.grid(axis='x', alpha=.2); fig.tight_layout()
    fig.savefig(result_dir / 'frequency.png', dpi=160); fig.savefig(docs_dir / 'frequency.png', dpi=160)
    plt.close(fig)
    # Plot 3: MED dynamic programming matrix with instructional example.
    source, target = 'play', 'stay'
    D, dist = min_edit_distance(source, target)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(D, cmap='Blues')
    ax.set_xticks(range(len(target)+1), ['∅', *list(target)])
    ax.set_yticks(range(len(source)+1), ['∅', *list(source)])
    ax.set_xlabel('Target'); ax.set_ylabel('Source')
    ax.set_title(f'Minimum edit distance: {source} → {target} = {dist}')
    for y in range(D.shape[0]):
        for x in range(D.shape[1]):
            ax.text(x, y, str(D[y, x]), ha='center', va='center')
    fig.tight_layout()
    fig.savefig(result_dir / 'edit_distance.png', dpi=160); fig.savefig(docs_dir / 'edit_distance.png', dpi=160)
    plt.close(fig)
    csv_write(result_dir / 'med_matrix.csv', [dict(source_prefix='#' if i == 0 else source[i-1],
                 **{f'col_{j}': int(D[i,j]) for j in range(D.shape[1])}) for i in range(D.shape[0])])


def run(corpus: str | Path = 'data/shakespeare.txt',
        result_dir: str | Path = 'results', docs_dir: str | Path = 'docs',
        sample_size: int = 100, seed: int = 42) -> dict:
    corpus, source = ensure_corpus(corpus)
    words = process_data(corpus)
    if not words: raise ValueError('Порожній корпус')
    counts = get_count(words); probs = get_probs(counts); vocab = set(counts)
    cases = generate_test_cases(counts, sample_size=sample_size, seed=seed)
    scores, predictions = evaluate(cases, probs, vocab)
    result_dir = Path(result_dir); docs_dir = Path(docs_dir)
    result_dir.mkdir(parents=True, exist_ok=True); docs_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        'status': 'completed', 'corpus_source': source,
        'corpus_url_reference': SOURCE_URL if source.startswith('NLTK Gutenberg:') else None,
        'total_tokens': len(words), 'unique_words': len(vocab),
        'sum_probabilities': round(sum(probs.values()), 8),
        'random_seed': seed, 'requested_test_cases': sample_size,
        **scores
    }
    (result_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding='utf-8')
    csv_write(result_dir / 'predictions.csv', predictions)
    med_examples = [{'source': a, 'target': b, 'distance': min_edit_distance(a, b)[1]}
                    for a, b in [('play', 'stay'), ('sunday', 'saturday'),
                                 ('intention', 'execution')]]
    csv_write(result_dir / 'med_examples.csv', med_examples)
    metrics['med_examples'] = med_examples
    (result_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding='utf-8')
    top = Counter(counts).most_common(100)
    csv_write(result_dir / 'word_frequency_top100.csv',
              [{'word': w, 'count': c, 'probability': round(probs[w], 9)} for w, c in top])
    examples = [{'input': w, 'suggestions': [{'word': t, 'probability': p}
                 for t, p in get_corrections(w, probs, vocab, n=3)]}
                for w in ('hllo', 'mony', 'comming', 'happpy', 'xylophonne')]
    (result_dir / 'example_corrections.json').write_text(json.dumps(examples, indent=2, ensure_ascii=False), encoding='utf-8')
    make_figures(metrics, predictions, counts, result_dir, docs_dir)
    display_predictions = [{k: row[k] for k in ('typed', 'expected', 'predicted', 'correct', 'edits')}
                           for row in predictions[:30]]
    docs_data = {'status': 'completed', 'metrics': metrics, 'examples': examples,
                 'sample_predictions': display_predictions,
                 'top_words': [{'word': w, 'count': c} for w, c in top[:15]]}
    (docs_dir / 'results.json').write_text(json.dumps(docs_data, indent=2, ensure_ascii=False), encoding='utf-8')
    two_acc_text = (f"{scores['accuracy_two_edits']:.2%}"
                    if scores['accuracy_two_edits'] is not None else 'немає')
    report = f'''# Звіт з лабораторної роботи №5

**Тема:** Автокорекція тексту з використанням мінімальної відстані редагування.

## Мета

Реалізувати частотну модель слів, операції редагування, автокорекцію
і алгоритм мінімальної відстані редагування; виміряти точність та час роботи.

## Дані

- Джерело корпусу: {source}.
- Токенів (усі входження слів): {len(words):,}.
- Унікальних слів: {len(vocab):,}.
- Сума ймовірностей: {sum(probs.values()):.8f}.

## Реалізація

Виконано операції видалення, заміни, вставки та перестановки сусідніх літер.
Кандидати шукаються на відстані одного, а за відсутності кандидатів — двох
редагувань. Серед кандидатів одного рівня обирається слово з найбільшою
емпіричною ймовірністю P(c)=count(c)/N. Динамічне програмування обчислює
MED з ins_cost=1, del_cost=1, rep_cost=2. Перестановка є операцією генерації
кандидатів, але не окремою операцією в MED.

## Результати контрольного експерименту

- Прикладів (штучно згенеровані опечатки): {scores['test_count']}.
- Top-1 accuracy: {scores['accuracy_top1']:.2%} ({scores['correct_top1']} правильних).
- Top-3 accuracy: {scores['accuracy_top3']:.2%}.
- Accuracy для однієї операції: {scores['accuracy_one_edit']:.2%} ({scores['one_edit_count']} тестів).
- Accuracy для двох операцій: {two_acc_text} ({scores['two_edit_count']} тестів).
- Середній час виправлення одного слова: {scores['avg_correction_ms']:.3f} мс.
- Загальний час виправлення тестових слів: {scores['total_correction_seconds']:.3f} с.

**Методика оцінювання:** із корпусу випадково вибрано частотні слова,
до яких внесено одну або дві синтетичні помилки. Тест містить лише опечатки,
яких немає в словнику. Це відтворюваний контрольний експеримент,
але не незалежний корпус реальних орфографічних помилок.

## Приклади

- Демонстраційні слова із завдання збережено у `example_corrections.json`.
- Повна таблиця передбачень: `predictions.csv`.
- Матриця MED: `med_matrix.csv`, графік `edit_distance.png`.
- MED для всіх трьох пар із методички: `med_examples.csv`.
- Графіки: `accuracy.png`, `frequency.png`.

## Висновок

Створено словникову систему автокорекції з генерацією кандидатів та
ймовірнісним ранжуванням. Точність залежить від корпусу та частоти слів:
рідкісні слова можуть програвати частішим альтернативам; коректно написані
слова не змінюються, а слова поза словником іноді не мають кандидатів.
Більші корпуси підвищують покриття, проте пошук кандидатів на відстані 2
має вищу обчислювальну вартість.
'''
    (result_dir / 'REPORT.md').write_text(report, encoding='utf-8')
    print('Готово. Корпус:', source)
    print('Токени:', len(words), 'Унікальні слова:', len(vocab))
    print('Top-1:', f"{scores['accuracy_top1']:.2%}", 'Top-3:', f"{scores['accuracy_top3']:.2%}")
    print('Середній час:', scores['avg_correction_ms'], 'мс')
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', default='data/shakespeare.txt')
    parser.add_argument('--out', default='results')
    parser.add_argument('--docs', default='docs')
    parser.add_argument('--tests', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    run(args.corpus, args.out, args.docs, args.tests, args.seed)
