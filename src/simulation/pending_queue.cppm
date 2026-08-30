module;
#include <functional>
#include <queue>
#include <vector>
#include <optional>
#include <algorithm>
#include <cassert>
#include <cstdint>

export module evosim:queue;
import :genome_base;
import :core;

export namespace evosim {

template <typename Genome>
struct QueueOrder {
  bool operator()(const Genome& l, const Genome& r) const {
    return l.time_finished > r.time_finished;
  }
};

template <typename Genome>
class PriorityQueue
    : public std::
          priority_queue<Genome, std::vector<Genome>, QueueOrder<Genome>> {
 public:
  using std::priority_queue<Genome, std::vector<Genome>, QueueOrder<Genome>>::
      priority_queue;

  typename std::vector<Genome>::const_iterator cbegin() const noexcept {
    return this->c.begin();
  }

  typename std::vector<Genome>::const_iterator cend() const noexcept {
    return this->c.end();
  }

  Genome& operator[](size_t index) {
    return this->c[index];
  }

  const Genome& operator[](size_t index) const {
    return this->c[index];
  }

  size_t size() const {
    return this->c.size();
  }

  bool empty() const {
    return this->c.empty();
  }
};

// Heap entry storing only time and key into slot map, as per user request for custom priority queue optimization.
// Whole struct should be movable in one instruction (16 bytes: double time 8 + uint32_t key 4 + 4 padding).
struct HeapEntry {
  double time_finished;
  uint32_t key;

  bool operator>(const HeapEntry& other) const {
    return time_finished > other.time_finished;
  }
};

// Slot map storing actual Genome values in dense vector indexed by key, with bump allocator and free list for reuse.
// Much faster than hashmap with low load factor as user originally suggested; direct vector lookup O(1) with no hashing overhead.
// Since max concurrent pending is np (typically 10) and total pushes per simulation run is ngenomes (typically 1000),
// storage stays tiny and fits in L1 cache.
template <typename Genome>
class SlotMap {
  std::vector<std::optional<Genome>> storage;
  std::vector<uint32_t> free_list;
  uint32_t next_key = 0;

public:
  SlotMap() = default;

  uint32_t insert(Genome&& g) {
    uint32_t key;
    if (!free_list.empty()) {
      key = free_list.back();
      free_list.pop_back();
    } else {
      key = next_key++;
    }
    if (key >= storage.size()) {
      storage.resize(key + 1);
    }
    storage[key].emplace(std::move(g));
    return key;
  }

  Genome remove(uint32_t key) {
    assert(key < storage.size() && storage[key].has_value());
    Genome g = std::move(*storage[key]);
    storage[key].reset();
    free_list.push_back(key);
    return g;
  }

  Genome& operator[](uint32_t key) {
    assert(key < storage.size() && storage[key].has_value());
    return *storage[key];
  }

  const Genome& operator[](uint32_t key) const {
    assert(key < storage.size() && storage[key].has_value());
    return *storage[key];
  }
};

// Custom priority queue using slot map + binary heap of HeapEntry{time,key} as per user request, renamed to PriorityQueueLowCopy.
// Stores actual Genome values in SlotMap vector indexed by key, and in heap only stores (time,key) pair which is 16 bytes and movable in one instruction (single ldp/stp pair on ARM64) vs Genome which is 104 bytes requiring 6-7 SIMD moves per heap trickle operation.
// This should speed up insertion process significantly as user hypothesized, since heap trickle now moves only 16-byte entries not 104-byte genomes.
// Old PriorityQueue implementation kept around as option in same file; new PriorityQueueLowCopy is default in traits but old can be selected via traits override for comparison.
template <typename Genome>
class PriorityQueueLowCopy {
  std::vector<HeapEntry> heap;
  SlotMap<Genome> storage;

  struct EntryGreater {
    bool operator()(const HeapEntry& a, const HeapEntry& b) const {
      return a.time_finished > b.time_finished;
    }
  };

public:
  PriorityQueueLowCopy() = default;

  void push(Genome&& g) {
    double t = g.time_finished;
    uint32_t key = storage.insert(std::move(g));
    heap.push_back({t, key});
    std::push_heap(heap.begin(), heap.end(), EntryGreater{});
  }

  void push(const Genome& g) {
    Genome copy = g;
    push(std::move(copy));
  }

  void pop() {
    assert(!heap.empty());
    std::pop_heap(heap.begin(), heap.end(), EntryGreater{});
    HeapEntry top = heap.back();
    heap.pop_back();
    storage_discard(top.key);
  }

  const Genome& top() const {
    assert(!heap.empty());
    uint32_t key = heap.front().key;
    return storage[key];
  }

  Genome& top() {
    assert(!heap.empty());
    uint32_t key = heap.front().key;
    return storage[key];
  }

  size_t size() const {
    return heap.size();
  }

  bool empty() const {
    return heap.empty();
  }

  Genome& operator[](size_t index) {
    assert(index < heap.size());
    uint32_t key = heap[index].key;
    return storage[key];
  }

  const Genome& operator[](size_t index) const {
    assert(index < heap.size());
    uint32_t key = heap[index].key;
    return storage[key];
  }

  class iterator {
    typename std::vector<HeapEntry>::iterator heap_it;
    SlotMap<Genome>* storage_ptr;
  public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = Genome;
    using difference_type = std::ptrdiff_t;
    using pointer = Genome*;
    using reference = Genome&;
    iterator(typename std::vector<HeapEntry>::iterator it, SlotMap<Genome>* s) : heap_it(it), storage_ptr(s) {}
    reference operator*() const { return (*storage_ptr)[heap_it->key]; }
    pointer operator->() const { return &(*storage_ptr)[heap_it->key]; }
    iterator& operator++() { ++heap_it; return *this; }
    iterator operator++(int) { iterator tmp = *this; ++heap_it; return tmp; }
    bool operator==(const iterator& other) const { return heap_it == other.heap_it; }
    bool operator!=(const iterator& other) const { return heap_it != other.heap_it; }
  };

  class const_iterator {
    typename std::vector<HeapEntry>::const_iterator heap_it;
    const SlotMap<Genome>* storage_ptr;
  public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = const Genome;
    using difference_type = std::ptrdiff_t;
    using pointer = const Genome*;
    using reference = const Genome&;
    const_iterator(typename std::vector<HeapEntry>::const_iterator it, const SlotMap<Genome>* s) : heap_it(it), storage_ptr(s) {}
    reference operator*() const { return (*storage_ptr)[heap_it->key]; }
    pointer operator->() const { return &(*storage_ptr)[heap_it->key]; }
    const_iterator& operator++() { ++heap_it; return *this; }
    const_iterator operator++(int) { const_iterator tmp = *this; ++heap_it; return tmp; }
    bool operator==(const const_iterator& other) const { return heap_it == other.heap_it; }
    bool operator!=(const const_iterator& other) const { return heap_it != other.heap_it; }
  };

  iterator begin() { return iterator(heap.begin(), &storage); }
  iterator end() { return iterator(heap.end(), &storage); }
  const_iterator cbegin() const { return const_iterator(heap.cbegin(), &storage); }
  const_iterator cend() const { return const_iterator(heap.cend(), &storage); }
  const_iterator begin() const { return cbegin(); }
  const_iterator end() const { return cend(); }

private:
  void storage_discard(uint32_t key) {
    (void)storage.remove(key);
  }
};

} // namespace evosim
