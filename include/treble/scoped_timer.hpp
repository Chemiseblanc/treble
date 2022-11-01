#ifndef TREBLE_SCOPED_TIMER_H
#define TREBLE_SCOPED_TIMER_H
//          Copyright Matthew Gibson 2022.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE.txt or copy at
//          https://www.boost.org/LICENSE_1_0.txt)

#include <chrono>
#include <functional>

namespace treble {
/**
 * @brief Starts a timer on construction and invokes a callback on destruction
 */
template <typename Duration>
class scoped_timer {
  using clock = std::chrono::high_resolution_clock;
  using time_point = std::chrono::time_point<clock>;

  time_point start;
  std::function<void(Duration)> on_exit;

 public:
  explicit scoped_timer(std::function<void(Duration)> callback) noexcept
      : start{clock::now()}, on_exit{std::move(callback)} {}
  scoped_timer(const scoped_timer &) noexcept = default;
  scoped_timer(scoped_timer &&) noexcept = default;
  ~scoped_timer() noexcept(noexcept(
      on_exit(std::chrono::duration_cast<Duration>(clock::now() - start)))) {
    time_point end{clock::now()};
    on_exit(std::chrono::duration_cast<Duration>(end - start));
  }

  scoped_timer &operator=(const scoped_timer &) noexcept = default;
  scoped_timer &operator=(scoped_timer &&) noexcept = default;
};
}  // namespace treble

#endif  // TREBLE_SCOPED_TIMER_H