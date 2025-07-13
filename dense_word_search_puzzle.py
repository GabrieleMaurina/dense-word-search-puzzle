import argparse
import dataclasses
import logging
import os.path
import random
import multiprocessing
import string
import time
import typing

DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WORD_LIST = os.path.join(DIR, 'words.txt')


@dataclasses.dataclass
class SolutionWord:
    word: str
    position: list[tuple[int, int]]


class DenseWordSearchPuzzle:
    def __init__(
        self,
        width: int = 10,
        height: int = 10,
        words: list | None = None,
        words_file: str | None = None,
        diagonal: bool = False,
        backwards: bool = False,
        seconds: int = 60,
        log_level: int = logging.INFO,
    ):
        self._log_level = log_level
        self._init_logger()
        if words is None and words_file is None:
            raise ValueError('Either "words" or "words_file" must be provided')
        if words is not None and words_file is not None:
            raise ValueError(
                'Only one of "words" or "words_file" can be provided')
        if words is not None and not words:
            raise ValueError('List "words" cannot be empty')
        if words_file is not None and (
            not words_file or not os.path.isfile(words_file)
        ):
            raise ValueError('File "words_file" must be a valid file path')
        self._width = width
        self._height = height
        self._words: list[str] = words or []
        self._words_file = words_file
        self._remaining_words: list[str] = []
        self._diagonal = diagonal
        self._backwards = backwards
        self._seconds = seconds
        self._solution_words: list[SolutionWord] = []
        self._lengths = {}
        self._max_lengths = {}
        self._puzzle = [[None for _ in range(
            self.height)] for _ in range(self.width)]

    @property
    def log_level(self) -> int:
        return self._log_level

    @property
    def logger(self) -> logging.Logger:
        return self._logger

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def words(self) -> list[str]:
        return self._words

    @property
    def words_file(self) -> str | None:
        return self._words_file

    @property
    def remaining_words(self) -> list[str]:
        return self._remaining_words

    @property
    def diagonal(self) -> bool:
        return self._diagonal

    @property
    def backwards(self) -> bool:
        return self._backwards

    @property
    def seconds(self) -> int:
        return self._seconds

    @property
    def solution_words(self) -> list[SolutionWord]:
        return self._solution_words

    @property
    def lengths(self) -> dict[int, list[str]]:
        return self._lengths

    @property
    def max_lengths(self) -> dict[int, list[str]]:
        return self._max_lengths

    @property
    def puzzle(self) -> list[list[str | None]]:
        return self._puzzle

    def _init_logger(self) -> None:
        self._logger = logging.getLogger('DenseWordSearchPuzzle')
        self.logger.setLevel(self.log_level)
        self.logger.propagate = False
        stream_handler = logging.StreamHandler()
        format = '[%(asctime)s] %(levelname)s %(pathname)s:%(lineno)d: %(message)s'
        datefmt = '%Y-%m-%d %H:%M:%S'
        formatter = logging.Formatter(format, datefmt=datefmt)
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)

    def _load_words(self) -> None:
        if not self.words:
            assert self.words_file is not None, (
                'Words file must be provided if words are empty'
            )
            self.logger.info(f'Loading words from file: {self.words_file}')
            if not os.path.isfile(self.words_file):
                raise FileNotFoundError(
                    f'Words file not found: {self.words_file}')
            with open(self.words_file, 'r') as file:
                self._words = [line.strip().lower()
                               for line in file if line.strip()]
        self._validate_words()
        self.logger.info(f'Loaded and validated {len(self._words)} words')
        if not self._words:
            raise ValueError('No valid words found to generate the puzzle')
        self._remaining_words = self._words.copy()

    def _is_valid_word(self, word: str) -> bool:
        valid = word.isalpha() and len(word) > 1
        if not valid:
            self.logger.warning(f'Skipping invalid word: {word}')
        return valid

    def _validate_words(self) -> None:
        self.logger.info('Validating words')
        self._words = list(set(filter(self._is_valid_word, self._words)))

    def _preprocess_words(self) -> None:
        self.logger.info('Preprocessing words for length categorization')
        for word in self.words:
            length = len(word)
            if length not in self.lengths:
                self.lengths[length] = []
            self.lengths[length].append(word)
        lengths = sorted(self.lengths.keys())
        for length in range(lengths[0], lengths[-1]):
            self.max_lengths[length] = []
        for length, words in self.lengths.items():
            for ml in self.max_lengths:
                if length <= ml:
                    self.max_lengths[ml].extend(words)

    def _get_random_row_or_column(self) -> tuple[str, int]:
        directions = ['row', 'column']
        if self.diagonal:
            directions.append('diagonal')
        direction = random.choice(directions)
        match direction:
            case 'row':
                row = random.randint(0, self.height - 1)
                return ('row', row)
            case 'column':
                column = random.randint(0, self.width - 1)
                return ('column', column)
            case 'diagonal':
                diagonal = random.randint(1, self.width + self.height - 2)
                return ('diagonal', diagonal)

    def _fit_word(self, current_values: list[str | None]) -> None:
        pass

    def _get_word_to_insert(self, args: dict[str, typing.Any]) -> SolutionWord:
        location = args['location']
        words = args['words']
        puzzle = args['puzzle']
        current_values = []
        match location[0]:
            case 'row':
                for x in range(self.width):
                    current_values.append(self._puzzle[x][location[1]])
            case 'column':
                for y in range(self.height):
                    current_values.append(self._puzzle[location[1]][y])
            case 'diagonal':
                x = max(0, location[1] - self.height + 1)
                y = min(location[1], self.height - 1)
                for i in range(min(self.width - x, y + 1)):
                    current_values.append(self._puzzle[x][y])
                    x += 1
                    y -= 1

    def _insert_word(self, SolutionWord: SolutionWord) -> bool:
        for i, (x, y) in enumerate(SolutionWord.position):
            current_char = self._puzzle[x][y]
            new_char = SolutionWord.word[i]
            if current_char is not None and current_char != new_char:
                return False
        for i, (x, y) in enumerate(SolutionWord.position):
            self._puzzle[x][y] = SolutionWord.word[i]
        return True

    def _insert_words(self) -> None:
        self.logger.info('Inserting words into the puzzle')
        start = time.time()
        with multiprocessing.Pool() as pool:
            while time.time() - start < self.seconds:
                locations = [{
                    'location': self._get_random_row_or_column(),
                    'words': self.remaining_words,
                    'puzzle': self.puzzle,
                } for _ in multiprocessing.cpu_count()]
                for word in pool.imap_unordered(self._get_word_to_insert, locations):
                    if word is None:
                        continue
                    if word.word not in self._remaining_words:
                        continue
                    inserted = self._insert_word(word)
                    if inserted:
                        self._solution_words.append(word)
                        self._remaining_words.remove(word.word)

    def _fill_remaining_spaces(self) -> None:
        self.logger.info('Filling remaining spaces with random letters')
        for x in range(self.width):
            for y in range(self.height):
                if self._puzzle[x][y] is None:
                    self._puzzle[x][y] = random.choice(string.ascii_lowercase)

    def generate_puzzle(self) -> None:
        self.logger.info('Starting Dense Word Search Puzzle Generator')
        self.logger.info(f'Puzzle dimensions: {self.width}x{self.height}')
        self._load_words()
        self._preprocess_words()
        self._insert_words()
        self._fill_remaining_spaces()

    def __str__(self) -> str:
        string = ''
        for y in range(self.height - 1, -1, -1):
            if y < self.height - 1:
                string += '\n'
            for x in range(self.width):
                if self._puzzle[x][y] is None:
                    string += ' '
                else:
                    string += self._puzzle[x][y]  # type: ignore
        return string


def existing_file(path):
    if not os.path.isfile(path):
        raise argparse.ArgumentTypeError(f'File "{path}" does not exist.')
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Dense Word Search Puzzle Generator')
    parser.add_argument(
        '-x', '--width', type=int, default=10, help='Width of the puzzle grid'
    )
    parser.add_argument(
        '-y', '--height', type=int, default=10, help='Height of the puzzle grid'
    )
    parser.add_argument(
        '-d', '--diagonal', default=False, action='store_true', help='Allow diagonal word placement'
    )
    parser.add_argument(
        '-b', '--backwards', default=False, action='store_true', help='Allow backwards word placement'
    )
    parser.add_argument(
        '-s', '--seconds', type=int, default=60, help='Max seconds to run the puzzle generator (default: 60 seconds)'
    )
    parser.add_argument(
        '-l',
        '--log-level',
        type=str,
        default='INFO',
        help='Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '-w',
        '--words',
        nargs='+',
        default=None,
        help='List of words to include in the puzzle (space-separated)',
    )
    group.add_argument(
        '-wf',
        '--words-file',
        type=existing_file,
        default=None,
        help='Path to a file containing the list of words (one word per line)',
    )
    args = parser.parse_args()
    if args.words_file is None and args.words is None:
        args.words_file = DEFAULT_WORD_LIST
    args.log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    return args


def main() -> None:
    args = parse_args()
    dwsp = DenseWordSearchPuzzle(
        width=args.width,
        height=args.height,
        words=args.words,
        words_file=args.words_file,
        diagonal=args.diagonal,
        backwards=args.backwards,
        seconds=args.seconds,
        log_level=args.log_level,
    )
    dwsp.generate_puzzle()
    print(dwsp)
    for i, word in enumerate(dwsp.solution_words, 1):
        print(i, word.word)


if __name__ == '__main__':
    main()
