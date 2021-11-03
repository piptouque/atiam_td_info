
from typing import Any, List, Union, Callable, Dict, Tuple
import numpy as np
import queue
import threading
import pretty_midi as midi
import copy
import random
import string


def swap_el(a: Union[np.ndarray, List], i: int, j: int) -> None:
    # see: https://stackoverflow.com/a/47951813
    if isinstance(a, np.ndarray):
        a[[i, j]] = a[[j, i]]
    elif isinstance(a, list):
        a[i], a[j] = a[j], a[i]


def find_el(arr: np.ndarray, el: Any) -> Tuple:
    # see: https://stackoverflow.com/a/41732691
    for idx, val in np.ndenumerate(arr):
        if val == el:
            return idx if arr.ndim > 1 else idx[0]
    return None


def get_random_word(size: int) -> str:
    # get random words
    # source: https://stackoverflow.com/a/2030081
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(size))


def quicksort(a: Union[np.ndarray, List], start_idx: int, stop_idx: int, comp: Callable[[Any, Any], int], use_threads: bool = False) -> None:
    """Recursive step
    Args:
        a (np.ndarray): The (n-1-axis)D-sub-array of input array to be sorted.
        start_idx ([type]): Start index of sub-array to be sorted
        stop_idx ([type]): Stop index of sub-array to be sorted (inclusive)
        pivot_idx ([type]): Index of pivot, should be in [start_idx, stop_idx]
        use_threads(bool): Defaults to False.
    """

    def quicksort_partition(a: Union[np.ndarray, List], start_idx: int, stop_idx: int, pivot_idx: int) -> int:
        """Recursive step of the quicksort algorithm.
        Args:
            a (np.ndarray): [description]
            start_idx (int): [description]
            stop_idx (int): [description]
            pivot_idx (int): [description]

        Returns:
            int: -1 if done, new pivot_idx value otherwise.
        """
        sub_arr_len = stop_idx-start_idx+1
        if sub_arr_len <= 1:
            # nothing to do
            return -1
        elif sub_arr_len == 2 and comp(a[start_idx], a[stop_idx]) > 0:
            swap_el(a, start_idx, stop_idx)
            return -1
        # if pivot is not the starting element, swap em
        if pivot_idx != start_idx:
            swap_el(a, start_idx, pivot_idx)
        # sort_start_idx should now point to the first element to be processed
        # after pivot_idx.
        sort_start_idx, pivot_idx = pivot_idx+1, start_idx
        # Using pivot to partition the array
        left_idx, right_idx = sort_start_idx, stop_idx
        # for it in range(sort_start_idx, stop_idx+1):
        while left_idx <= right_idx:
            while left_idx <= right_idx and comp(a[left_idx], a[pivot_idx]) <= 0:
                left_idx += 1
            while right_idx >= left_idx and comp(a[right_idx], a[pivot_idx]) >= 0:
                right_idx -= 1
            if left_idx < right_idx:
                # swap the two
                swap_el(a, left_idx, right_idx)
        # Finally, swap right with pivot
        swap_el(a, pivot_idx, right_idx)
        pivot_idx, right_idx = right_idx, pivot_idx
        return pivot_idx
    q = queue.Queue(maxsize=-1)  # no limit

    def loop():
        start_idx, stop_idx = q.get()
        pivot_idx = start_idx
        pivot_idx = quicksort_partition(
            a, start_idx, stop_idx, pivot_idx)
        if pivot_idx >= 0:
            q.put((start_idx, pivot_idx-1))
            q.put((pivot_idx+1, stop_idx))
        q.task_done()

    q.put((start_idx, stop_idx))
    # todo: for some reason using threads is slower. I'm probably not doing it right
    if use_threads:
        def worker():
            while True:
                loop()
        threading.Thread(target=worker, daemon=True).start()
    else:
        while not q.empty():
            loop()
    q.join()


def pipsort_list(seq: List, comp: Callable[[Any, Any], int], kind: Union['quicksort'], use_threads: bool) -> None:
    """In-place quick-sorting of input 1D-list.
    Args:
        seq (List): input 1D-list to be sorted.
        use_threads(bool): Defaults to False. Not used.
    """
    if kind == 'quicksort':
        quicksort(seq, 0, len(seq)-1, comp=comp, use_threads=use_threads)
    else:
        raise AttributeError(f"Sorting method not supported: {kind}")


def pipsort_array(arr: np.ndarray, axis: int, comp: Callable[[Any, Any], int], kind: Union['quicksort'], use_threads: bool) -> None:
    """In-place quick-sorting of input nD-array along axis.
    Args:
        array (np.ndarray): input nD-array to be sorted.
        axis (int, optional): Defaults to -1 for last axis.
        use_threads(bool): Defaults to False.
    """
    Ni = np.shape(arr)[:axis]
    Nk = np.shape(arr)[axis+1:]
    alg = None
    if kind == 'quicksort':
        alg = quicksort
    else:
        raise AttributeError(f"Sorting method not supported: {kind}")
    np.apply_along_axis(lambda _a: alg(_a, 0, np.shape(
        _a)[0]-1, comp=comp, use_threads=use_threads), axis=axis, arr=arr)


def pipsort(a: Union[np.ndarray, List], axis: int = -1, order: Union[None, str, List[str], Callable[[Any, Any], int]] = None, kind: Union['quicksort'] = 'quicksort', use_threads: bool = False) -> None:
    """In-place quick-sorting of input nD-array, according to input comparison function.
    Args:
        array (np.ndarray): input nD-array to be sorted.
        order (Union[str, List[str]]: see numpy doc.
        axis (int, optional): Defaults to -1 for last axis.
        use_threads(bool): Defaults to False.
    """
    def comp(a, b) -> int:
        if callable(order):
            return order(a, b)
        elif order is not None:
            def comp_rec(a, b, i: int) -> int:
                comp = a[order[i]] - b[order[i]]
                if comp > 0:
                    return 1
                elif comp < 0:
                    return -1
                elif i >= len(order)-1:
                    return 0
                else:
                    return comp_rec(a, b, i+1)
            return comp_rec(a, b, 0)
        else:
            return a - b

    if axis == None:
        raise AttributeError(
            "Use out-of-place version to sort a flattened version of the array.")
    if isinstance(a, np.ndarray):
        pipsort_array(a, axis=axis, comp=comp,
                      kind=kind, use_threads=use_threads)
    elif isinstance(a, list):
        pipsort_list(a, comp=comp, kind=kind, use_threads=use_threads)
    else:
        raise AttributeError(f"Type not recognised: {type(a)}")


def pipsorted(a: Union[np.ndarray, List], axis: Union[int, None] = -1, order: Union[None, str, List[str], Callable[[Any, Any], int]] = None, kind: Union['quicksort'] = 'quicksort', use_threads: bool = False) -> np.ndarray:
    """Out-of-place quick-sorting of nD-array, according to input predicate.
    See pipsort for documentation.
    """
    a_sorted = None
    if axis is None:
        a_sorted = np.flatten(arr)
        axis = 0
    else:
        a_sorted = copy.deepcopy(a)
    pipsort(a_sorted, axis=axis, order=order,
            kind=kind, use_threads=use_threads)
    return a_sorted


def get_instrument_notes(part: midi.PrettyMIDI) -> Dict[int, List[midi.Note]]:
    """Get played notes by instruments (indexed by programme)

    Args:
        part (midi.PrettyMIDI):

    Returns:
        Dict[int, List[midi.Note]]: Dictionary of notes indexed by instrument programme.
    """
    return dict([(instr.program, instr.notes.copy()) for instr in part.instruments])


def load_similarity_matrix(path: str, delimiter: str = ' ', comments: str = '#') -> Tuple[np.ndarray, np.ndarray]:
    """[summary]

    Args:
        path (str): [description]
        delimiter (str, optional): [description]. Defaults to ' '.
        comments (str, optional): [description]. Defaults to '#'.

    Returns:
        np.ndarray: [description]
    """
    def loadtxt_skip(path: str, **kargs):
        # The letters in the first row and first column should be the same.
        # Apparently numpy.loadtxt includes comment lines in skiprows,
        # so we need to skip those first.
        # see: https://stackoverflow.com/a/17151323
        with open(path) as f:
            lines = (line for line in f if not line.startswith(comments))
            return np.loadtxt(lines, **kargs)
    # Load file a first time to get the letters
    letters = loadtxt_skip(path, dtype=str, max_rows=1)
    letters_2 = loadtxt_skip(path, dtype=str, skiprows=1, usecols=0)
    if not np.array_equal(letters, letters_2):
        raise AttributeError(
            f"Letters are inconsistent: {letters}!={letters_2}")
    cols = list(range(1, len(letters)+1))  #  skip first column
    sim_mat = loadtxt_skip(path, dtype=np.int64, skiprows=1, usecols=cols)
    return sim_mat, letters


def get_directions(dim: int) -> List[Tuple]:
    """Get (sorted) list of vectors of 'strictly positive' directions.

    Args:
        dim (int): [description]

    Returns:
        [type]: [description]
    """
    # Binary representation to string to tuple.
    # see: https://stackoverflow.com/a/21732313
    return [tuple([int(c) for c in format(i, 'b').zfill(dim)]) for i in range(1, 2 ** dim)]


def scores_to_grad(left_score, top_score, top_left_score) -> Tuple[int, str]:
    score_list = [left_score, top_score, top_left_score]
    grad = ''
    max_score = np.nanmax(score_list)
    for (i, score) in enumerate(score_list):
        grad = grad + str(int(score == max_score))
    return max_score, grad


def get_similarity(idx_s: Tuple[int, int], s_1: str, s_2: str, sim_mat: np.ndarray, letters: np.ndarray) -> int:
    # warning: the indices are shifted (1, 1),
    # as can be seen in the definition of path_mat and grad_mat
    l_1, l_2 = s_1[idx_s[0]-1], s_2[idx_s[1]-1]
    idx_l = find_el(letters, l_1), find_el(letters, l_2)
    if np.any(np.array(idx_l) == None):
        raise AttributeError(
            f"Letter similarity not found: '{l_1}' and '{l_2}'")
    return sim_mat[idx_l]
# for reference on affine gap implementation,
# see: https://www.cs.cmu.edu/~ckingsf/bioinfo-lectures/gaps.pdf


def get_path_matrices(s_1: str, s_2: str, sim_mat: np.ndarray, letters: np.ndarray, gap_penalty: int) -> Tuple[np.ndarray, np.ndarray]:
    """Get the filled path matrices used in the Needleman-Wunsch algorithm

    Args:
        s_1 (str): [description]
        s_2 (str): [description]
        sim_mat (np.ndarray): [description]

    Returns:
        Tuple[np.ndarray, np.ndarray]: [description]
    """
    n, m = len(s_1), len(s_2)
    path_mat = np.empty((n+1, m+1), dtype=np.int64)
    grad_mat = np.empty((n+1, m+1), dtype=np.dtype('U3'))

    # first we have to fill the first line and column
    # for both path and gradient matrices!
    path_mat[:, 0] = np.arange(path_mat.shape[0]) * gap_penalty
    path_mat[0, :] = np.arange(path_mat.shape[1]) * gap_penalty
    grad_mat[:, 0] = np.full_like(grad_mat[:, 0], fill_value='010')
    grad_mat[0, :] = np.full_like(grad_mat[0, :], fill_value='100')
    dirs = get_directions(path_mat.ndim)

    with np.nditer(path_mat, op_flags=['readwrite'], flags=['multi_index']) as it:
        for x in it:
            idx = it.multi_index
            if np.all(np.array(idx)):
                # tuple arithmetic
                # see: https://stackoverflow.com/a/17418273
                left_idx = tuple(np.subtract(idx, dirs[0]))
                top_idx = tuple(np.subtract(idx, dirs[1]))
                top_left_idx = tuple(np.subtract(idx, dirs[2]))
                # indel
                left_score = path_mat[left_idx] + gap_penalty
                top_score = path_mat[top_idx] + gap_penalty
                # match or mismatch, depends on similarity
                top_left_score = path_mat[top_left_idx] + \
                    get_similarity(idx, s_1, s_2, sim_mat, letters)
                # signal whether each direction is optimal
                # by '1', else '0'
                # ex: '100', '101'
                max_score, grad_mat[idx] = scores_to_grad(
                    left_score, top_score, top_left_score)
                x[...] = max_score
    # path_mat = path_mat[1:, 1:]
    # grad_mat = grad_mat[1:, 1:]
    return path_mat, grad_mat


def get_path_matrices_affine(s_1: str, s_2: str, sim_mat: np.ndarray, letters: np.ndarray, gap_penalty: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """Get the filled path matrices used in the Needleman-Wunsch algorithm

    Args:
        s_1 (str): [description]
        s_2 (str): [description]
        sim_mat (np.ndarray): [description]

    Returns:
        Tuple[np.ndarray, np.ndarray]: [description]
    """

    n, m = len(s_1), len(s_2)
    path_mat = np.empty((n+1, m+1), dtype=np.int64)
    grad_mat = np.empty((n+1, m+1), dtype=np.dtype('U3'))
    m_mat = np.empty((n+1, m+1), dtype=np.float32)
    x_mat = np.empty_like(m_mat)
    y_mat = np.empty_like(m_mat)

    # first we have to fill the first line and column
    # for both path and gradient matrices!
    #
    m_mat[:, 0] = np.full(m_mat.shape[0], fill_value=-np.inf)
    m_mat[0, :] = np.full(m_mat.shape[1], fill_value=-np.inf)
    x_mat[:, 0] = np.arange(x_mat.shape[0]) * gap_penalty[1] + gap_penalty[0]
    x_mat[0, :] = np.full(x_mat.shape[1], fill_value=-np.inf)
    y_mat[:, 0] = np.full(y_mat.shape[0], fill_value=-np.inf)
    y_mat[0, :] = np.arange(y_mat.shape[1]) * gap_penalty[1] + gap_penalty[0]
    x_mat[0, 0] = 0
    y_mat[0, 0] = 0
    #
    path_mat[:, 0] = x_mat[:, 0]
    path_mat[0, :] = y_mat[0, :]
    #
    # signals whether each direction is optimal
    # by '1', else '0'
    # ex: '100', '101'
    grad_mat[:, 0] = np.full_like(grad_mat[:, 0], fill_value='010')
    grad_mat[0, :] = np.full_like(grad_mat[0, :], fill_value='100')
    dirs = get_directions(path_mat.ndim)

    with np.nditer(path_mat, op_flags=['readwrite'], flags=['multi_index']) as it:
        for el in it:
            idx = it.multi_index
            if np.all(np.array(idx)):
                # tuple arithmetic
                # see: https://stackoverflow.com/a/17418273
                left_idx = tuple(np.subtract(idx, dirs[0]))
                top_idx = tuple(np.subtract(idx, dirs[1]))
                top_left_idx = tuple(np.subtract(idx, dirs[2]))
                # match-mismatch
                m_mat[idx] = np.amax(np.array(
                    [m_mat[top_left_idx], x_mat[top_left_idx], y_mat[top_left_idx]]) + get_similarity(idx, s_1, s_2, sim_mat, letters))
                # indel
                x_mat[idx] = np.amax(np.array([m_mat[left_idx] + gap_penalty[0],
                                     x_mat[left_idx] + gap_penalty[1], y_mat[left_idx] + gap_penalty[0]]))
                y_mat[idx] = np.amax(np.array(
                    [m_mat[top_idx] + gap_penalty[0], x_mat[top_idx] + gap_penalty[0], y_mat[top_idx] + gap_penalty[1]]))
                max_score, grad_mat[idx] = scores_to_grad(
                    x_mat[idx], y_mat[idx], m_mat[idx])
                el[...] = max_score
    return path_mat, grad_mat


def get_aligned_strings(s_1: str, s_2: str, path_mat: np.ndarray, grad_mat: np.ndarray, gap_symbol: str = '-') -> Dict[List[Tuple[str, str]], int]:
    # start from the bottom right
    last_idx = tuple(np.subtract(path_mat.shape, (1, 1)))
    #
    score = path_mat[last_idx]
    #
    path_list = [[last_idx, '', '']]
    paths_done = []
    #
    dirs = get_directions(path_mat.ndim)

    def get_next_letters(next_idx: Tuple[int, int]):
        # indices are offset by (1, 1)
        # since grad and path are (n+1, m+1)
        idx = tuple(np.subtract(next_idx, dirs[2]))
        if i == 0:
            return [gap_symbol, s_2[idx[1]]]
        elif i == 1:
            return [s_1[idx[0]], gap_symbol]
        else:
            return [s_1[idx[0]], s_2[idx[1]]]
    while not len(path_list) == 0:
        path = path_list.pop()
        idx = path[0]
        if np.all(np.array(idx) == 0):
            # done!
            paths_done.append(path)
        else:
            grad = grad_mat[idx]
            # check for one or more optimal steps
            have_optimal_steps = np.array(
                [bool(int(g)) for g in grad], dtype=bool)
            assert(np.any(have_optimal_steps))
            nb_paths = np.count_nonzero(have_optimal_steps)
            next_ids = [tuple(np.subtract(idx, d)) for d in dirs]
            for i in range(len(dirs)):
                # first option is continued by this path
                next_idx = next_ids[i]
                if have_optimal_steps[i] and np.all(np.array(next_idx) >= 0):
                    next_letters = get_next_letters(idx)

                    next_step = [next_idx, path[1] +
                                 next_letters[0], path[2] + next_letters[1]]
                    path_list.append(next_step)
    # now we can return the paths in reverse order
    # they will give us the actual aligned strings.
    aligned_strings = [(path[1][::-1], path[2][::-1]) for path in paths_done]
    return dict({'string_list': aligned_strings, 'score': score})


def pipman(s_1: str, s_2: str, sim_mat: np.ndarray, letters: np.ndarray, gap_penality: int) -> Tuple[List[Tuple[str, str]], int]:
    """[summary]

    Args:
        s_1 (str): [description]
        s_2 (str): [description]
        sim_mat (np.ndarray): [description]
        gap_penality (int, optional): [description]. Defaults to -5.

    Returns:
        np.ndarray: [description]
    """
    path_mat, grad_mat = get_path_matrices(
        s_1, s_2, sim_mat, letters, gap_penalty=gap_penality)
    aligned = get_aligned_strings(s_1, s_2, path_mat, grad_mat)
    return aligned


def pipman_affine(s_1: str, s_2: str, sim_mat: np.ndarray, letters: np.ndarray, gap_penality: Union[Tuple[int, int], int]) -> Tuple[List[Tuple[str, str]], int]:
    """[summary]

    Args:
        s_1 (str): [description]
        s_2 (str): [description]
        sim_mat (np.ndarray): [description]
        gap_penality (int, optional): [description]. Defaults to -5.

    Returns:
        np.ndarray: [description]
    """
    path_mat, grad_mat = get_path_matrices_affine(
        s_1, s_2, sim_mat, letters, gap_penalty=gap_penality)
    aligned = get_aligned_strings(s_1, s_2, path_mat, grad_mat)
    return aligned
