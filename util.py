
from typing import Any, List, Union, Callable
import numpy as np
import queue
import threading


def log_error(expected: Any, answer: Any, fail: bool = True) -> None:
    print("-- ERROR --")
    print(f"expected:  {expected}")
    print(f"got:       {answer}")
    if fail:
        raise AttributeError("Failed")


def swap_el(a: np.ndarray, i: int, j: int) -> None:
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


def pip_sort_list(seq: List, axis: int, order: Union[str, List[str]], way: Union['default', 'above', 'below'], kind: Union['quick'], use_threads: bool) -> None:
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
    raise NotImplementedError()


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
    # set comp: increasing order

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
                elif order >= len(order)-1:
                    return 0
                else:
                    return comp_rec(a, b, i+1)
            return comp_rec(a, b, 0)
        else:
            return a - b

    if axis == None:
        raise AttributeError(
            "Use out-of-place version for sorting flattened version of the array")
    if isinstance(a, np.ndarray):
        pip_sort_array(a, axis=axis, comp=comp,
                       way=way, kind=kind, use_threads=use_threads)
    elif isinstance(a, list):
        pip_sort_list(a, axis=axis, comp=comp,
                      way=way, kind=kind, use_threads=use_threads)
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
