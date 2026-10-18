#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_PENDINGREADBACKS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_PENDINGREADBACKS_H_

#include <array>
#include <atomic>
#include <cstdint>
#include <mutex>

namespace Libs::Graphics {

// Guest-memory windows of asynchronous fault readbacks whose copies are in flight.
//
// A guest read fault on GPU-written memory records a copy on the readback queue and returns to
// the execution thread; the faulting thread waits for the copy and asks the execution thread to
// publish it. Between the two, anything that changes what the window holds or owns (new GPU
// writes, other downloads or backing writes of its bytes, the buffer's deletion) invalidates the
// window: its bytes are then dropped and the fault falls back to the synchronous readback, so a
// stale copy is never published.
//
// Thread-safe: backing writes are reported from any thread.
class PendingReadbackWindows {
public:
	// Entries are reported as a 32-bit mask (InvalidateOwner()).
	static constexpr uint32_t Capacity = 32;
	static constexpr uint32_t None     = UINT32_MAX;

	// Registers [begin, end) for `owner`; None when every entry is in use.
	uint32_t Add(uint64_t begin, uint64_t end, uint32_t owner) {
		std::lock_guard lock(m_mutex);
		for (uint32_t i = 0; i < Capacity; i++) {
			auto& entry = m_entries[i];
			if (!entry.active) {
				entry = {.begin = begin, .end = end, .owner = owner, .active = true, .valid = true};
				m_active.fetch_add(1, std::memory_order_release);
				return i;
			}
		}
		return None;
	}

	// Invalidates every window overlapping [begin, end).
	void Invalidate(uint64_t begin, uint64_t end) {
		if (m_active.load(std::memory_order_acquire) == 0 || begin >= end) {
			return;
		}
		std::lock_guard lock(m_mutex);
		for (auto& entry: m_entries) {
			if (entry.active && begin < entry.end && entry.begin < end) {
				entry.valid = false;
			}
		}
	}

	// Invalidates the windows of `owner`; returns a mask of their entries.
	uint32_t InvalidateOwner(uint32_t owner) {
		if (m_active.load(std::memory_order_acquire) == 0) {
			return 0;
		}
		std::lock_guard lock(m_mutex);
		uint32_t        mask = 0;
		for (uint32_t i = 0; i < Capacity; i++) {
			auto& entry = m_entries[i];
			if (entry.active && entry.owner == owner) {
				entry.valid = false;
				mask |= 1u << i;
			}
		}
		return mask;
	}

	// Frees the entry; true when no change invalidated it since Add().
	bool Remove(uint32_t index) {
		std::lock_guard lock(m_mutex);
		auto&           entry = m_entries[index];
		if (!entry.active) {
			return false;
		}
		const bool valid = entry.valid;
		entry            = {};
		m_active.fetch_sub(1, std::memory_order_release);
		return valid;
	}

	[[nodiscard]] bool IsActive(uint32_t index) const {
		std::lock_guard lock(m_mutex);
		return m_entries[index].active;
	}

	[[nodiscard]] uint32_t ActiveCount() const { return m_active.load(std::memory_order_acquire); }

private:
	struct Entry {
		uint64_t begin  = 0;
		uint64_t end    = 0;
		uint32_t owner  = 0;
		bool     active = false;
		bool     valid  = false;
	};
	mutable std::mutex           m_mutex;
	std::array<Entry, Capacity>  m_entries {};
	std::atomic_uint32_t         m_active {0};
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_PENDINGREADBACKS_H_
