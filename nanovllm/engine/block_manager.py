from collections import deque
from time import perf_counter

from nanovllm.engine.sequence import Sequence


class RadixNode:

    def __init__(self, token_ids=(), parent=None):
        self.token_ids = tuple(token_ids)
        self.parent: "RadixNode | None" = parent
        self.children: dict[tuple[int, ...], "RadixNode"] = {}
        self.block_id = -1


class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.token_ids = []
        self.node: RadixNode | None = None
        self.prev_free_block: "Block | None" = None
        self.next_free_block: "Block | None" = None
        self.is_evictable = False

    def update(self, token_ids: list[int]):
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.token_ids = []
        self.prev_free_block = None
        self.next_free_block = None
        self.is_evictable = False


class BlockManager:
    """
    Blocks (or tokens) layout:

    ----------------------------------------------------------------------
    | < computed > | < new_computed > |       < new >       |
    ----------------------------------------------------------------------
    |     < Prefix-cached tokens >    |  < to be computed > |
    ----------------------------------------------------------------------
                                      | < to be allocated > |
    ----------------------------------------------------------------------
                                      |   < to be cached >  |
    ----------------------------------------------------------------------

    """

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()
        self.root = RadixNode()
        self.evictable_head: Block | None = None
        self.evictable_tail: Block | None = None
        self.num_evictable_blocks = 0
        self.reset_stats()

    def reset_stats(self):
        self.stats = {
            "can_allocate_time": 0.0,
            "can_allocate_calls": 0,
            "get_token_layout_time": 0.0,
            "get_token_layout_calls": 0,
            "allocate_time": 0.0,
            "allocate_calls": 0,
            "deallocate_time": 0.0,
            "deallocate_calls": 0,
            "can_append_time": 0.0,
            "can_append_calls": 0,
            "may_append_time": 0.0,
            "may_append_calls": 0,
        }

    def get_stats(self) -> dict[str, float | int]:
        stats = dict(self.stats)
        stats["total_time"] = (
            stats["can_allocate_time"]
            + stats["get_token_layout_time"]
            + stats["allocate_time"]
            + stats["deallocate_time"]
            + stats["can_append_time"]
            + stats["may_append_time"]
        )
        return stats

    def _get_child(self, node: RadixNode, token_ids: list[int]) -> RadixNode | None:
        return node.children.get(tuple(token_ids))

    def _get_or_create_child(self, node: RadixNode, token_ids: list[int]) -> RadixNode:
        key = tuple(token_ids)
        child = node.children.get(key)
        if child is None:
            child = RadixNode(token_ids=key, parent=node)
            node.children[key] = child
        return child

    def _prune_node(self, node: RadixNode | None):
        while node is not None and node is not self.root and node.block_id == -1 and not node.children:
            parent = node.parent
            if parent is not None:
                parent.children.pop(node.token_ids, None)
            node.parent = None
            node = parent

    def _detach_block_from_tree(self, block: Block):
        node = block.node
        if node is None:
            return
        if node.block_id == block.block_id:
            node.block_id = -1
        block.node = None
        self._prune_node(node)

    def _attach_block_to_tree(self, block: Block, node: RadixNode):
        if node.block_id not in (-1, block.block_id):
            # Initial Radix Tree version keeps the first cached block for a path.
            return
        if block.node is not None and block.node is not node:
            self._detach_block_from_tree(block)
        node.block_id = block.block_id
        block.node = node

    def _add_evictable_block(self, block: Block):
        assert block.ref_count == 0
        assert block.node is not None
        if block.is_evictable:
            return
        block.prev_free_block = self.evictable_tail
        block.next_free_block = None
        if self.evictable_tail is not None:
            self.evictable_tail.next_free_block = block
        else:
            self.evictable_head = block
        self.evictable_tail = block
        block.is_evictable = True
        self.num_evictable_blocks += 1

    def _remove_evictable_block(self, block: Block):
        if not block.is_evictable:
            return
        prev_block = block.prev_free_block
        next_block = block.next_free_block
        if prev_block is not None:
            prev_block.next_free_block = next_block
        else:
            self.evictable_head = next_block
        if next_block is not None:
            next_block.prev_free_block = prev_block
        else:
            self.evictable_tail = prev_block
        block.prev_free_block = None
        block.next_free_block = None
        block.is_evictable = False
        self.num_evictable_blocks -= 1

    def _pop_evictable_block(self) -> Block | None:
        block = self.evictable_head
        if block is None:
            return None
        self._remove_evictable_block(block)
        return block

    def _evict_block(self, block: Block) -> Block:
        assert block.ref_count == 0
        self._remove_evictable_block(block)
        self._detach_block_from_tree(block)
        block.reset()
        return block

    def _allocate_new_block(self) -> Block:
        if self.free_block_ids:
            block_id = self.free_block_ids.popleft()
            block = self.blocks[block_id]
            assert block.ref_count == 0
            block.reset()
        else:
            block = self._pop_evictable_block()
            assert block is not None
            block = self._evict_block(block)
        self.used_block_ids.add(block.block_id)
        return block

    def _activate_cached_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count == 0
        self._remove_evictable_block(block)
        self.used_block_ids.add(block_id)
        block.ref_count = 1
        return block

    def _deallocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count == 0
        self.used_block_ids.remove(block_id)
        if block.node is not None:
            self._add_evictable_block(block)
        else:
            block.token_ids = []
            self.free_block_ids.append(block_id)

    def _num_available_blocks(self) -> int:
        return len(self.free_block_ids) + self.num_evictable_blocks

    def _match_prefix(self, seq: Sequence) -> list[tuple[int, list[int], RadixNode]]:
        matched = []
        node = self.root
        # Preserve the current behavior: the last block is always recomputed.
        num_cacheable_blocks = max(seq.num_blocks - 1, 0)
        for i in range(num_cacheable_blocks):
            token_ids = seq.block(i)
            if len(token_ids) != self.block_size:
                break
            node = self._get_child(node, token_ids)
            if node is None or node.block_id == -1:
                break
            block = self.blocks[node.block_id]
            if block.token_ids != token_ids:
                break
            matched.append((block.block_id, token_ids, node))
        return matched

    def _node_from_block_id(self, block_id: int) -> RadixNode:
        if block_id == -1:
            return self.root
        block = self.blocks[block_id]
        return block.node if block.node is not None else self.root

    def can_allocate(self, num_tokens: int) -> bool:
        """
        Only for seq in the waiting queue.
        """
        start = perf_counter()
        try:
            return self._num_available_blocks() >= (num_tokens + self.block_size - 1) // self.block_size
        finally:
            self.stats["can_allocate_time"] += perf_counter() - start
            self.stats["can_allocate_calls"] += 1

    def get_token_layout(self, seq: Sequence):
        """
        Only for seq in the waiting queue.
        """
        start = perf_counter()
        try:
            assert not seq.block_table
            num_new_tokens = 0
            num_new_computed_tokens_in_used = 0
            num_new_computed_tokens_in_free = 0
            matched = self._match_prefix(seq)
            matched_ids = {block_id for block_id, _, _ in matched}

            for i in range(seq.num_blocks):
                token_ids = seq.block(i)
                if i < len(matched):
                    block_id = matched[i][0]
                    if block_id in self.used_block_ids:
                        num_new_computed_tokens_in_used += len(token_ids)
                    else:
                        num_new_computed_tokens_in_free += len(token_ids)
                    continue
                if i == seq.num_blocks - 1 and len(matched_ids) == seq.num_blocks:
                    num_new_tokens += len(token_ids)
                    continue
                num_new_tokens += len(token_ids)
            return num_new_computed_tokens_in_used, num_new_computed_tokens_in_free, num_new_tokens
        finally:
            self.stats["get_token_layout_time"] += perf_counter() - start
            self.stats["get_token_layout_calls"] += 1

    def allocate(self, seq: Sequence):
        """
        Only for seq in the waiting queue.
        """
        start = perf_counter()
        try:
            assert not seq.block_table
            parent_node = self.root
            matched = self._match_prefix(seq)

            for block_id, token_ids, node in matched:
                seq.num_cached_tokens += self.block_size
                if block_id in self.used_block_ids:
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    block = self._activate_cached_block(block_id)
                block.update(token_ids)
                seq.block_table.append(block_id)
                parent_node = node

            for i in range(seq.num_cached_tokens, seq.num_cached_tokens + seq.num_new_tokens, self.block_size):
                token_ids = seq[i: min(i + self.block_size, seq.num_cached_tokens + seq.num_new_tokens)]
                block = self._allocate_new_block()
                block.update(token_ids)
                if len(token_ids) == self.block_size:
                    parent_node = self._get_or_create_child(parent_node, token_ids)
                    self._attach_block_to_tree(block, parent_node)
                seq.block_table.append(block.block_id)
        finally:
            self.stats["allocate_time"] += perf_counter() - start
            self.stats["allocate_calls"] += 1

    def deallocate(self, seq: Sequence):
        """
        For finished seq or preempted seq in the running queue.
        """
        start = perf_counter()
        try:
            for block_id in reversed(seq.block_table):
                block = self.blocks[block_id]
                block.ref_count -= 1
                if block.ref_count == 0:
                    self._deallocate_block(block_id)
            seq.num_cached_tokens = 0
            seq.num_new_tokens = 0
            seq.block_table.clear()
        finally:
            self.stats["deallocate_time"] += perf_counter() - start
            self.stats["deallocate_calls"] += 1

    def can_append(self, seq: Sequence, num_new_tokens: int) -> bool:
        """
        Only for seq in the running queue.
        """
        start = perf_counter()
        try:
            last_computed_block_capacity = self.block_size - (seq.num_cached_tokens % self.block_size)
            if last_computed_block_capacity == self.block_size:
                last_computed_block_capacity = 0
            if (num_new_tokens - last_computed_block_capacity + self.block_size - 1) // self.block_size \
                <= self._num_available_blocks():
                return True
            return False
        finally:
            self.stats["can_append_time"] += perf_counter() - start
            self.stats["can_append_calls"] += 1

    def may_append(self, seq: Sequence):
        """
        Only for seq in the running queue.
        """
        start = perf_counter()
        try:
            for i in range(
                seq.num_cached_blocks * self.block_size,
                seq.num_cached_tokens + seq.num_new_tokens,
                self.block_size
            ):
                token_ids = seq[i: min(i + self.block_size, seq.num_cached_tokens + seq.num_new_tokens)]
                block_index = i // self.block_size
                current_block_id = seq.block_table[block_index] if block_index < len(seq.block_table) else -1
                current_block = self.blocks[current_block_id] if current_block_id != -1 else None

                if len(token_ids) == self.block_size:
                    previous_block_id = seq.block_table[block_index - 1] if block_index > 0 else -1
                    if current_block_id == -1:
                        current_block = self._allocate_new_block()
                        seq.block_table.append(current_block.block_id)
                    else:
                        # A running sequence may revisit a partially-filled block and
                        # extend it into a full cacheable block in a later step.
                        assert current_block.node is None
                        assert token_ids[:len(current_block.token_ids)] == current_block.token_ids
                    current_block.update(token_ids)
                    parent_node = self._node_from_block_id(previous_block_id)
                    node = self._get_or_create_child(parent_node, token_ids)
                    self._attach_block_to_tree(current_block, node)
                else:
                    if current_block_id == -1:
                        current_block = self._allocate_new_block()
                        seq.block_table.append(current_block.block_id)
                    else:
                        assert current_block.node is None
                    current_block.update(token_ids)
        finally:
            self.stats["may_append_time"] += perf_counter() - start
            self.stats["may_append_calls"] += 1
