
from typing import Any, List, Union, Callable, Dict, Tuple
import numpy as np
import queue
import threading
import pretty_midi as midi


def log_error(expected: Any, answer: Any, fail: bool = True) -> None:
    print("-- ERROR --")
    print(f"expected:  {expected}")
    print(f"got:       {answer}")
    if fail:
        raise AttributeError("Failed")


def swap_el(a: Union[np.ndarray, List], i: int, j: int) -> None:
    # see: https://stackoverflow.com/a/47951813
    a[[i, j]] = a[[j, i]]


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


def pip_sort_list(seq: List, comp: Callable[[Any, Any], int], kind: Union['quicksort'], use_threads: bool) -> None:
    """In-place quick-sorting of input 1D-list.
    Args:
        seq (List): input 1D-list to be sorted.
        use_threads(bool): Defaults to False. Not used.
    """
    # l = []
    # if np.ndim(seq) == 1:
    #     l = pip_sorted(np.array(seq), axis=axis, order=comp,
    #         way=way, kind=kind, use_threads=use_threads).tolist()
    # elif np.ndim(seq) == 2:
    # else:
    #    raise NotImplementedError()
    arr = pip_sort_array(np.array(seq), axis=-1, comp=comp,
                         way='default', kind=kind, use_threads=use_threads)
    # stupid hack to copy new elements to same address, since this is an in-place version.
    # todo: better.
    seq.clear()
    seq.extend(arr)


def pip_sort_array(arr: np.ndarray, axis: int, comp: Callable[[Any, Any], int], way: Union['default', 'above', 'below'], kind: Union['quicksort'], use_threads: bool) -> None:
    """In-place quick-sorting of input nD-array along axis.
    Args:
        array (np.ndarray): input nD-array to be sorted.
        axis (int, optional): Defaults to -1 for last axis.
        way (Union[, optional):
            If 'default', will sort recursively as numpy does,
            If 'below', will sort the (n-1-axis)D-sub-arrays.
            If 'above', will sort the (axis)D-sub-arrays.
            Defaults to 'default'.
        use_threads(bool): Defaults to False.
    """
    Ni = np.shape(arr)[:axis]
    Nk = np.shape(arr)[axis+1:]
    if way == 'default':
        np.apply_along_axis(lambda _a: quicksort(_a, 0, np.shape(
            _a)[0]-1, comp=comp, use_threads=use_threads), axis=axis, arr=arr)
    elif way == 'below':
        for ii in np.ndindex(Ni):
            quicksort(arr[ii + np.s_[:, ]], 0, np.shape(arr)[axis]-1,
                      comp=comp, use_threads=use_threads)
    elif way == 'above':
        for kk in np.ndindex(Nk):
            quicksort(arr[np.s_[:, ] + kk], 0, np.shape(arr)
                      [axis]-1, comp=comp, use_threads=use_threads)
    else:
        raise AttributeError(f"Way not recognised: {way}")


def pip_sort(a: Union[np.ndarray, List], axis: int = -1, order: Union[None, str, List[str], Callable[[Any, Any], int]] = None,  way: Union['default', 'above', 'below'] = 'default', kind: Union['quicksort'] = 'quicksort', use_threads: bool = False) -> None:
    """In-place quick-sorting of input nD-array, according to input comparison function.
    Args:
        array (np.ndarray): input nD-array to be sorted.
        order (Union[str, List[str]]: see numpy doc.
        axis (int, optional): Defaults to -1 for last axis.
        way (Union[, optional):
            If 'default', will sort recursively as numpy does,
            If 'below', will sort the (n-1-axis)D-sub-arrays.
            If 'above', will sort the (axis)D-sub-arrays.
            Defaults to 'default'.
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
        pip_sort_array(a, axis=axis, comp=comp,
                       way=way, kind=kind, use_threads=use_threads)
    elif isinstance(a, list):
        pip_sort_list(a, comp=comp, kind=kind, use_threads=use_threads)
    else:
        raise AttributeError(f"Type not recognised: {type(a)}")


def pip_sorted(a: Union[np.ndarray, List], axis: Union[int, None] = -1, order: Union[None, str, List[str], Callable[[Any, Any], int]] = None, way: Union['default', 'above', 'below'] = 'default', kind: Union['quicksort'] = 'quicksort', use_threads: bool = False) -> np.ndarray:
    """Out-of-place quick-sorting of nD-array, according to input predicate.
    See pip_sort for documentation.
    """
    a_sorted = None
    if axis is None:
        a_sorted = np.flatten(arr)
        axis = 0
    else:
        a_sorted = np.copy(a)
    pip_sort(a_sorted, axis=axis, order=order, way=way,
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
    letters = loadtxt_skip(path, dtype=np.dtype('U1'), max_rows=1)
    letters_2 = loadtxt_skip(path, dtype=np.dtype('U1'), skiprows=1, usecols=0)
    if not np.array_equal(letters, letters_2):
        raise AttributeError(
            f"Letters are inconsistent: {letters}!={letters_2}")
    cols = list(range(1, len(letters)+1))  #  skip first column
    pam = loadtxt_skip(path, dtype=np.int8, skiprows=1, usecols=cols)
    return letters, pam


def get_path_matrices(s1: str, s2: str, sim: np.ndarray, gap_penalty: Union[Tuple[int, int], int]) -> Tuple[np.ndarray, np.ndarray]:
    """Get the filled path matrices used in the Needleman-Wunsch algorithm

    Args:
        s1 (str): [description]
        s2 (str): [description]
        sim (np.ndarray): [description]

    Returns:
        Tuple[np.ndarray, np.ndarray]: [description]
    """
    n, m = len(s1), len(s2)
    path_mat = np.empty((n+1, m+1), dtype=np.int64)
    grad_mat = np.full((n+1, m+1, 3, 2), fill_value=-1, dtype=np.int64)

    # mat[:, 0] = -1
    # mat[0, :] = -1
    with np.nditer(path_mat, op_flags=['readwrite'], flags=['multi_index']) as it:
        for x in it:
            idx = it.multi_index
            if idx[0] == 0 or idx[1] == 0:
                dist = max(idx[0], idx[1])
                # Fill the first column and row
                x[...] = - dist * gap_penalty
            else:
                # tuple arithmetics
                # see: https://stackoverflow.com/a/17418273
                left_idx = tuple(np.subtract(idx, (1, 0)))
                top_idx = tuple(np.subtract(idx, (0, 1)))
                top_left_idx = tuple(np.subtract(idx, (1, 1)))
                # indel
                left_score = path_mat[left_idx] - gap_penalty
                top_score = path_mat[top_idx] - gap_penalty
                # match or mismatch, depends on similarity
                top_left_score = path_mat[top_left_idx] + sim[idx]
                max_score = np.amax([left_score, top_score, top_left_score])
                if left_score == max_score:
                    grad_mat[idx, 0] = left_idx
                if top_score == max_score:
                    grad_mat[idx, 1] = top_idx
                if top_left_score == max_score:
                    grad_mat[idx, 2] = top_left_idx
                x[...] = max_score
    path_mat = path_mat[1:, 1:]
    grad_mat = grad_mat[1:, 1:]
    return path_mat, grad_mat


def compute_optimal_paths(path_mat: np.ndarray, grad_mat: np.ndarray) -> List[Tuple[str, str]]:
    # start from the bottom right
    idx = tuple(np.subtract(path_mat.shape, (1, 1))
    path_l=[idx]
    while idx
    pass


def pipman(s1: str, s2: str, sim: np.ndarray, gap_penality: int) -> np.ndarray:
    """[summary]

    Args:
        s1 (str): [description]
        s2 (str): [description]
        sim (np.ndarray): [description]
        gap_penality (int, optional): [description]. Defaults to -5.

    Returns:
        np.ndarray: [description]
    """
    score=0
    ################
    # YOUR CODE HERE
    ################
    mat=get_path_matrix(s1, s2, sim, gap_penalty)
    return mat
