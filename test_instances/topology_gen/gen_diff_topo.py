# Import libraries
from typing import List, Dict, Any
import math
import random


# Classes
class OperationMode:
    """
    代表一個操作在單一機器上的執行方式 (一個模式)。
    它是處理時間和機器ID的來源，支持機器的不同處理時間。
    """

    def __init__(self, machine_id: int, processing_time: int):
        self.machine_id: int = machine_id  # 機器 ID
        self.processing_time: int = processing_time  # 該模式下的處理時間

    def __str__(self):
        return f"(m={self.machine_id}, t={self.processing_time})"


class Operation:
    """代表一個工作中的單一加工步驟。
    包含該操作的所有可行模式清單，這是彈性的核心。"""

    def __init__(self, op_id: int, job_id: int, pos_in_job: int, modes: list[OperationMode]):
        self.op_id: int = op_id  # 全局唯一 ID
        self.job_id: int = job_id  # 所屬工作 ID
        self.pos_in_job: int = pos_in_job  # 在工作中的順序位置 (1-based)
        self.modes: list[OperationMode] = modes  # 可行模式清單 (List[OperationMode])

    def __str__(self):
        modes_str = ", ".join(str(m) for m in self.modes)
        return (
            f"Operation(op_id={self.op_id}, job_id={self.job_id}, pos_in_job={self.pos_in_job}, "
            f"modes=[{modes_str}])"
        )


class Job:
    """代表一個工作，包含其所有操作的有序序列。"""

    def __init__(self, job_id: int, operations: list[Operation]):
        self.job_id: int = job_id  # 工作 ID
        self.operations: list[Operation] = operations  # 有序的操作清單

    def __str__(self):
        # Only print crucial information: op_id, pos_in_job, and modes count
        ops_str = ", ".join("(" + ", ".join(str(mode) for mode in op.modes) + ")" for op in self.operations)
        return f"Job(j_id={self.job_id}, op_cnt == {len(self.operations)}, ops == [{ops_str}])"


# From Tai's JSSP format to abstract FJSSP structure
def format_tai_to_fjssp(file_content: str) -> Dict[str, Any]:
    """
    Parses content in Tai's JSSP format and converts it into an abstract FJSSP Job structure.

    Each line represents a job:
    - The first line contains two integers: job_cnt and machine_cnt.
    - Each subsequent line (one per job) contains pairs of numbers: (machine_id, processing_time) in order, representing all possible (machine, time) combinations for each operation.

    Args:
        file_content: str - The complete textual content of an FJSSP file.

    Returns:
        A dictionary containing:
            - 'jobs': List[Job] - the list of Job objects
            - 'machine_cnt': int - the number of machines
            - 'operations': List[Operation] - all operations across all jobs
    """

    # 分行
    lines = [line.strip() for line in file_content.strip().split("\n") if line.strip()]
    if len(lines) < 1:
        raise ValueError("內容缺少參數行")

    # 第一行：取出 job_cnt 與 machine_cnt
    top_tokens = lines[0].split()
    if len(top_tokens) < 2:
        raise ValueError("首行缺少 job_cnt 與 machine_cnt 參數")
    job_cnt = int(top_tokens[0])
    machine_cnt = int(top_tokens[1])

    # 從第二行開始，逐行處理每個工作
    all_jobs_data = []
    current_op_id = 0

    for job_idx, line in enumerate(lines[1:]):
        tokens = line.strip().split()
        if len(tokens) < 1:
            continue
        job_operations = []

        for i in range(0, len(tokens), 2):
            pos = i // 2 + 1
            machine_id = int(tokens[i])
            processing_time = int(tokens[i + 1])

            mode = OperationMode(machine_id=machine_id, processing_time=processing_time)

            operation = Operation(op_id=current_op_id, job_id=job_idx, pos_in_job=pos, modes=[mode])
            job_operations.append(operation)
            current_op_id += 1

        all_jobs_data.append(Job(job_id=job_idx, operations=job_operations))

    # Compute the union of all jobs' operations
    all_operations = []
    for job in all_jobs_data:
        all_operations.extend(job.operations)

    return {
        "jobs": all_jobs_data,
        "machine_cnt": machine_cnt,
        "operations": all_operations,
    }


# From abstract FJSSP structure to BRD format
def format_fjssp_to_brd(jobs: List[Job], machine_cnt: int, flex: int) -> str:
    """
    Convert abstract FJSSP jobs to BRD format.

    Args:
        jobs: List[Job], List of Job objects
        machine_cnt: int, total machine count
        flex: int, flexibility
    Returns:
        str, BRD format string
    """

    job_cnt = len(jobs)
    machine_cnt = machine_cnt

    brd_str = f"{job_cnt} {machine_cnt} {flex}\n"
    for job in jobs:
        job_line_str = f"{len(job.operations)} "
        for op in job.operations:
            job_line_str += f"{len(op.modes)} "
            for mode in op.modes:
                job_line_str += f"{mode.machine_id} {mode.processing_time} "
        brd_str += job_line_str + "\n"
    return brd_str


# Parallel, closed, linked
def process_parallel(data: Dict[str, Any], flex: int) -> Dict[str, Any]:
    """
    For each machine_id, collect its operations, create (flex-1) new machines,
    and for each operation, add the new machines (with the same processing time) to its modes list.

    Args:
        data: Dictionary containing 'jobs', 'machine_cnt', etc.
        flex: The degree of parallelism (number of parallel machines per original machine).
    Returns:
        jobs: List[Job], List of jobs with modified operations
        machine_cnt: int, new total machine count
    """

    jobs = data["jobs"]
    M_orig = data["machine_cnt"]
    next_machine_id = M_orig  # Start assigning new IDs after the originals

    for machine_id in range(M_orig):
        new_machine_ids = [next_machine_id + i for i in range(flex - 1)]
        next_machine_id += flex - 1
        # For each op whose first mode's machine_id equals machine_id, append new modes for the new machines
        for op in [op for op in data["operations"] if op.modes[0].machine_id == machine_id]:
            orig_time = op.modes[0].processing_time
            # Append new modes for each new machine (flex-1 replicas)
            for new_machine_id in new_machine_ids:
                new_mode = OperationMode(machine_id=new_machine_id, processing_time=orig_time)
                op.modes.append(new_mode)

    return {
        "jobs": jobs,
        "machine_cnt": next_machine_id,
    }


def process_closed(data: Dict[str, Any], flex: int) -> None:
    """
    將 JSSP 的 instance 轉換成 closed FJSSP 的 instance
    會直接更改 data 的內容

    Param:
        data: Dict[str, Any] - 原始的 JSSP 的 instance
            - 'jobs': List[Job] - 工作列表
            - 'machine_cnt': int - 機器數量
            - 'operations': List[Operation] - 所有操作
        flex: int - 彈性度
    Returns:
        None
    """

    # Partitioning
    machine_cnt = data.get("machine_cnt", 0)
    set_size = flex
    set_cnt = math.ceil(machine_cnt / flex)
    machines = list(range(set_cnt * set_size))
    data["machine_cnt"] = set_cnt * set_size
    random.shuffle(machines)
    machine_sets = [set(machines[i : i + set_size]) for i in range(0, len(machines), set_size)]

    new_machine_set_dict: Dict[int, set[int]] = {}
    for s in machine_sets:
        for m in s:
            new_machine_set_dict[m] = s.copy()

    for operation in data["operations"]:
        processing_time = operation.modes[0].processing_time
        cur_machine = operation.modes[0].machine_id

        for m in new_machine_set_dict[cur_machine]:
            if m == cur_machine:
                continue
            new_mode = OperationMode(machine_id=m, processing_time=processing_time)
            operation.modes.append(new_mode)


def process_linked(data: Dict[str, Any], flex: int) -> None:
    """
    將 JSSP 的 instance 轉換成 linked FJSSP 的 instance

    Param:
        data: Dict[str, Any] - 原始的 JSSP 的 instance
            - 'jobs': List[Job] - 工作列表
            - 'machine_cnt': int - 機器數量
            - 'operations': List[Operation] - 所有操作
        flex: int - 彈性度
    Returns:
        None
    """

    machine_cnt = data["machine_cnt"]
    for op in data["operations"]:
        cur_machine = op.modes[0].machine_id
        processing_time = op.modes[0].processing_time
        new_machines = [(cur_machine + i) % machine_cnt for i in range(flex)]
        for m in new_machines:
            if m == cur_machine:
                continue
            new_mode = OperationMode(machine_id=m, processing_time=processing_time)
            op.modes.append(new_mode)


def transform_to_FJSSP(input_path: str, flex: int, output_path: str, closed: bool) -> None:
    """
    Transform the instance to FJSSP format

    Param:
        input_path: str - the path of the input file
        flex: int - the flexibility
        output_path: str - the path of the output file
        closed: bool - True for closed topology, False for parallel topology
    Returns:
        None
    """

    with open(input_path, "r") as f:
        content = f.read()
        data = format_tai_to_fjssp(content)
    if closed:
        process_closed(data, flex)
    else:
        process_linked(data, flex)
    brd_str = format_fjssp_to_brd(data["jobs"], data["machine_cnt"], flex)
    with open(output_path, "w") as f:
        f.write(brd_str)
