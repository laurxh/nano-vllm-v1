from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
import types


ROOT = Path(__file__).resolve().parents[1]


def load_module(module_name: str, relative_path: str):
    spec = spec_from_file_location(module_name, ROOT / relative_path)
    module = module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


# Avoid importing nanovllm/__init__.py, which pulls heavyweight runtime deps.
nanovllm_pkg = sys.modules.setdefault("nanovllm", types.ModuleType("nanovllm"))
nanovllm_pkg.__path__ = [str(ROOT / "nanovllm")]
engine_pkg = sys.modules.setdefault("nanovllm.engine", types.ModuleType("nanovllm.engine"))
engine_pkg.__path__ = [str(ROOT / "nanovllm" / "engine")]

load_module("nanovllm.sampling_params", "nanovllm/sampling_params.py")
sequence_module = load_module("nanovllm.engine.sequence", "nanovllm/engine/sequence.py")
block_manager_module = load_module("nanovllm.engine.block_manager", "nanovllm/engine/block_manager.py")

Sequence = sequence_module.Sequence
BlockManager = block_manager_module.BlockManager


def make_seq(token_ids: list[int]) -> Sequence:
    seq = Sequence(token_ids)
    seq.block_table = []
    seq.num_cached_tokens = 0
    seq.num_new_tokens = 0
    return seq


def test_prefix_reuse_after_deallocate():
    manager = BlockManager(num_blocks=4, block_size=4)

    seq1 = make_seq([1, 2, 3, 4, 5, 6, 7, 8])
    used, free, new = manager.get_token_layout(seq1)
    assert (used, free, new) == (0, 0, 8)
    seq1.num_new_tokens = new
    manager.allocate(seq1)
    manager.deallocate(seq1)

    assert manager.num_evictable_blocks == 2
    assert len(manager.free_block_ids) == 2

    seq2 = make_seq([1, 2, 3, 4, 9, 10, 11, 12])
    used, free, new = manager.get_token_layout(seq2)
    assert used == 0
    assert free == 4
    assert new == 4

    seq2.num_new_tokens = new
    manager.allocate(seq2)
    assert seq2.num_cached_tokens == 4
    assert len(seq2.block_table) == 2
    assert manager.num_evictable_blocks == 1


def test_evictable_block_can_be_reused_for_new_prompt():
    manager = BlockManager(num_blocks=2, block_size=4)

    seq1 = make_seq([1, 2, 3, 4])
    _, _, new = manager.get_token_layout(seq1)
    seq1.num_new_tokens = new
    manager.allocate(seq1)
    manager.deallocate(seq1)

    assert manager.num_evictable_blocks == 1
    assert len(manager.free_block_ids) == 1

    seq2 = make_seq([5, 6, 7, 8, 9, 10, 11, 12])
    _, _, new = manager.get_token_layout(seq2)
    seq2.num_new_tokens = new
    manager.allocate(seq2)

    assert len(seq2.block_table) == 2
    assert len(manager.used_block_ids) == 2
    assert manager.num_evictable_blocks == 0
    assert len(manager.free_block_ids) == 0


def test_can_append_counts_evictable_blocks_as_available():
    manager = BlockManager(num_blocks=2, block_size=4)

    seq1 = make_seq([1, 2, 3, 4])
    _, _, new = manager.get_token_layout(seq1)
    seq1.num_new_tokens = new
    manager.allocate(seq1)
    manager.deallocate(seq1)

    seq2 = make_seq([20, 21, 22, 23])
    _, _, new = manager.get_token_layout(seq2)
    seq2.num_new_tokens = new
    manager.allocate(seq2)

    seq2.append_token(24)
    assert manager.can_append(seq2, 1)


def main():
    original_block_size = Sequence.block_size
    Sequence.block_size = 4
    try:
        test_prefix_reuse_after_deallocate()
        test_evictable_block_can_be_reused_for_new_prompt()
        test_can_append_counts_evictable_blocks_as_available()
    finally:
        Sequence.block_size = original_block_size
    print("block_manager tests passed")


if __name__ == "__main__":
    main()
